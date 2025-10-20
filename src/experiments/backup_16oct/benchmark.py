import torch
import time
import numpy as np
import yaml
import pandas as pd
import joblib

# Import all the models we want to compare
from src.models.multitask_model import MultitaskModel
from src.models.baselines.siloed_model import SiloedModel
from src.models.baselines.direct_vlm import DirectVLM
from transformers import AutoTokenizer
from src.data.dataloader import create_dataloaders

def count_parameters(model):
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_forward_latency(model, sample_input, device, num_runs=100):
    """ Measures the average latency of a single FORWARD pass (for non-generative models). """
    model.eval()
    latencies = []
    
    with torch.no_grad():
        for _ in range(10): _ = model(**sample_input)
        for _ in range(num_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            _ = model(**sample_input)
            end_event.record()
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))
    return np.mean(latencies), np.std(latencies)

def benchmark_generation_latency(model, sample_input, device, num_runs=50):
    """ Measures the average latency of the full GENERATE pass (for text generation models). """
    model.eval()
    latencies = []
    
    with torch.no_grad():
        for _ in range(5): _ = model.generate(sample_input['pixel_values'], max_length=128)
        for _ in range(num_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            _ = model.generate(sample_input['pixel_values'], max_length=128)
            end_event.record()
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))
    return np.mean(latencies), np.std(latencies)

def benchmark_tabular_latency(model, sample_input_df, num_runs=100):
    """ Measures the average CPU latency of a scikit-learn/XGBoost model. """
    latencies = []
    for _ in range(10): _ = model.predict(sample_input_df)
    for _ in range(num_runs):
        start_time = time.perf_counter()
        _ = model.predict(sample_input_df)
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)
    return np.mean(latencies), np.std(latencies)

def main():
    # --- SETUP ---
    MTL_CONFIG = 'configs/base_config.yaml'
    SILOED_ATTR_CONFIG = 'configs/exp1_1_siloed_attributes.yaml'
    SILOED_PRICE_CONFIG = 'configs/exp1_1_siloed_price.yaml'
    DIRECT_VLM_CONFIG = 'configs/exp2_1_direct_vlm.yaml'
    
    with open(MTL_CONFIG, 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- FIX: Call the dataloader function with the signature it expects. ---
    # We don't need the loaders themselves, but this is the correct way to call
    # the function to get the 'mappings' dictionary.
    _, _, mappings, _ = create_dataloaders(config)
    
    sample_pixel_values = torch.randn(1, 3, 224, 224).to(device)
    sample_vision_input = {'pixel_values': sample_pixel_values}
    sample_tabular_input = pd.DataFrame([dict.fromkeys(mappings.keys(), "Unknown")])

    print(f"--- Benchmarking on {device} ---")

    # --- 1. Benchmark Ours-MTL ---
    print("\n[1] Benchmarking Ours-MTL Model...")
    mtl_model = MultitaskModel(config, mappings).to(device)
    mtl_params = count_parameters(mtl_model)
    mtl_latency, mtl_std = benchmark_forward_latency(mtl_model, sample_vision_input, device)
    print(f"  - Total Parameters: {mtl_params / 1e6:.2f} M")
    print(f"  - FWD Pass Latency: {mtl_latency:.2f} ± {mtl_std:.2f} ms")

    # --- 2. Benchmark Baseline-Siloed ---
    print("\n[2] Benchmarking Baseline-Siloed Pipeline (Attributes + Price)...")
    with open(SILOED_ATTR_CONFIG, 'r') as f: attr_config = yaml.safe_load(f)
    attr_model = SiloedModel(attr_config, mappings).to(device)
    attr_params = count_parameters(attr_model)
    attr_latency, _ = benchmark_forward_latency(attr_model, sample_vision_input, device)
    
    with open(SILOED_PRICE_CONFIG, 'r') as f: price_config = yaml.safe_load(f)
    price_model = SiloedModel(price_config).to(device)
    price_params = count_parameters(price_model)
    price_latency, _ = benchmark_forward_latency(price_model, sample_vision_input, device)

    siloed_total_params = attr_params + price_params
    siloed_total_latency = attr_latency + price_latency
    print(f"  - Total Parameters (Sum): {siloed_total_params / 1e6:.2f} M")
    print(f"  - Total Inference Latency (Sum): {siloed_total_latency:.2f} ms")

    # --- 3. Benchmark Baseline-Hybrid ---
    print("\n[3] Benchmarking Baseline-Hybrid Pipeline (Vision -> XGBoost)...")
    hybrid_vision_params = attr_params
    hybrid_vision_latency = attr_latency
    
    xgb_model_path = 'models/hybrid_baseline/tabular_price_model.joblib'
    try:
        tabular_model = joblib.load(xgb_model_path)
        tabular_latency, _ = benchmark_tabular_latency(tabular_model, sample_tabular_input)
        
        hybrid_total_latency = hybrid_vision_latency + tabular_latency
        print(f"  - Total DL Parameters (Vision Stage): {hybrid_vision_params / 1e6:.2f} M")
        print(f"  - Total Inference Latency (Sum): {hybrid_total_latency:.2f} ms ({hybrid_vision_latency:.2f}ms GPU + {tabular_latency:.2f}ms CPU)")
    except FileNotFoundError:
        print(f"  - SKIPPING: XGBoost model not found at {xgb_model_path}. Run training first.")

    # --- 4. Benchmark Baseline-DirectVLM ---
    print("\n[4] Benchmarking Baseline-DirectVLM Model...")
    try:
        with open(DIRECT_VLM_CONFIG, 'r') as f:
            vlm_config = yaml.safe_load(f)
        vlm_model = DirectVLM(vlm_config).to(device)
        vlm_params = count_parameters(vlm_model)
        vlm_latency, vlm_std = benchmark_generation_latency(vlm_model.model, sample_vision_input, device)
        print(f"  - Total Parameters: {vlm_params / 1e6:.2f} M")
        print(f"  - Generation Latency: {vlm_latency:.2f} ± {vlm_std:.2f} ms")
    except FileNotFoundError:
        print(f"  - SKIPPING: Config not found at {DIRECT_VLM_CONFIG}. Make sure the file exists.")

if __name__ == '__main__':
    main()