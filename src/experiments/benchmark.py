import torch
import time
import numpy as np
import yaml
import pandas as pd
import joblib
from tqdm import tqdm

# --- FIX: Import the correct VLM class and its processor ---
from src.models.multitask_model import MultitaskModel
from src.models.baselines.siloed_model import SiloedModel
from src.models.baseline_git import GitBaselineModel
from src.models.baselines.direct_vlm_model import DirectVLM 
from transformers import GitProcessor, AutoTokenizer      
from src.data.dataloader import create_dataloaders

def count_parameters(model):
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_latency(model, inputs, device, num_runs=50):
    """
    --- FIX: A unified and robust latency benchmark function ---
    Measures the average latency of a model's end-to-end inference pass.
    """
    model.eval()
    model.to(device)
    latencies = []

    # Move tensor inputs to the correct device
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device)

    # Warm-up runs to stabilize GPU clocks and cache
    with torch.no_grad():
        for _ in range(10):
            _ = model(**inputs)

    # Timed runs
    with torch.no_grad():
        for _ in tqdm(range(num_runs), desc=f"Benchmarking {model.__class__.__name__}"):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            _ = model(**inputs)
            end_event.record()
            
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))
            
    return np.mean(latencies), np.std(latencies)

def benchmark_tabular_latency(model, sample_input_df, num_runs=100):
    """ Measures the average CPU latency of a scikit-learn/XGBoost model. (This was already correct) """
    latencies = []
    for _ in range(10): _ = model.predict(sample_input_df)
    for _ in range(num_runs):
        start_time = time.perf_counter()
        _ = model.predict(sample_input_df)
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)
    return np.mean(latencies), np.std(latencies)

# --- Wrapper classes to make different model methods compatible with the benchmark function ---
class MTLWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, pixel_values, tokenizer, attribute_mappers):
        return self.model.predict(pixel_values, tokenizer, attribute_mappers)

class VLMWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, pixel_values, processor):
        return self.model.predict(pixel_values, processor)

class DirectVLMWrapper(torch.nn.Module): # <-- ADDED
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, pixel_values, tokenizer):
        return self.model.generate(pixel_values, tokenizer)

def main():
    # --- SETUP ---
    MTL_CONFIG = 'configs/base_config.yaml'
    ATTR_CONFIG = 'configs/exp1_1_siloed_attributes.yaml'
    PRICE_CONFIG = 'configs/exp1_1_siloed_price.yaml'
    VLM_CONFIG = 'configs/git_base_config.yaml' # <-- Use the correct VLM config
    
    with open(MTL_CONFIG, 'r') as f:
        config = yaml.safe_load(f) # This is the base config for dataloaders
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    _, _, mappings, tokenizer = create_dataloaders(config)
    
    sample_pixel_values = torch.randn(1, 3, 224, 224)
    sample_tabular_input = pd.DataFrame([dict.fromkeys(mappings.keys(), "Unknown")])

    print(f"--- Benchmarking on {device} ---")
    results = []

    # --- 1. Benchmark Ours-MTL ---
    print("\n[1] Benchmarking Ours-MTL Model...")
    mtl_model = MultitaskModel(config, mappings)
    mtl_params = count_parameters(mtl_model)
    # --- FIX: Benchmark the full 'predict' method for true end-to-end latency ---
    mtl_inputs = {'pixel_values': sample_pixel_values, 'tokenizer': tokenizer, 'attribute_mappers': mappings}
    mtl_latency, mtl_std = benchmark_latency(MTLWrapper(mtl_model), mtl_inputs, device)
    results.append({'Model': 'Ours-MTL', 'Parameters (M)': mtl_params / 1e6, 'Latency (ms)': mtl_latency})
    print(f"  - Total Parameters: {mtl_params / 1e6:.2f} M")
    print(f"  - E2E Latency: {mtl_latency:.2f} ± {mtl_std:.2f} ms")

    # --- 2. Benchmark Baseline-Siloed ---
    print("\n[2] Benchmarking Baseline-Siloed Pipeline (Attributes + Price)...")
    with open(ATTR_CONFIG, 'r') as f:
        attr_config = yaml.safe_load(f)
    with open(PRICE_CONFIG, 'r') as f:
        price_config = yaml.safe_load(f)
        
    attr_model = SiloedModel(attr_config, mappings)
    price_model = SiloedModel(price_config)
    
    siloed_total_params = count_parameters(attr_model) + count_parameters(price_model)
    
    siloed_inputs = {'pixel_values': sample_pixel_values}
    attr_latency, _ = benchmark_latency(attr_model, siloed_inputs, device)
    price_latency, _ = benchmark_latency(price_model, siloed_inputs, device)
    siloed_total_latency = attr_latency + price_latency
    results.append({'Model': 'Baseline-Siloed', 'Parameters (M)': siloed_total_params / 1e6, 'Latency (ms)': siloed_total_latency})
    print(f"  - Total Parameters (Sum): {siloed_total_params / 1e6:.2f} M")
    print(f"  - Total Inference Latency (Sum): {siloed_total_latency:.2f} ms")

    # --- 3. Benchmark Baseline-Hybrid ---
    print("\n[3] Benchmarking Baseline-Hybrid Pipeline (Vision -> XGBoost)...")
    hybrid_vision_params = count_parameters(attr_model)
    hybrid_vision_latency = attr_latency
    
    xgb_model_path = 'models/hybrid_baseline/tabular_price_model.joblib'
    try:
        tabular_model = joblib.load(xgb_model_path)
        tabular_latency, _ = benchmark_tabular_latency(tabular_model, sample_tabular_input)
        hybrid_total_latency = hybrid_vision_latency + tabular_latency
        results.append({'Model': 'Baseline-Hybrid', 'Parameters (M)': hybrid_vision_params / 1e6, 'Latency (ms)': hybrid_total_latency})
        print(f"  - Total DL Parameters (Vision Stage): {hybrid_vision_params / 1e6:.2f} M")
        print(f"  - Total Inference Latency (Sum): {hybrid_total_latency:.2f} ms ({hybrid_vision_latency:.2f}ms GPU + {tabular_latency:.2f}ms CPU)")
    except FileNotFoundError:
        print(f"  - SKIPPING: XGBoost model not found at {xgb_model_path}. Run training first.")

    # --- 4. Benchmark Baseline-DirectVLM ---
    print("\n[4] Benchmarking Baseline-DirectVLM Model...")
    try:
        with open(VLM_CONFIG, 'r') as f:
            vlm_config = yaml.safe_load(f)
        # --- FIX: Load the correct model and processor ---
        vlm_model = GitBaselineModel(vlm_config)
        vlm_processor = GitProcessor.from_pretrained(vlm_config['model']['model_name'])
        
        vlm_params = count_parameters(vlm_model)
        vlm_inputs = {'pixel_values': sample_pixel_values, 'processor': vlm_processor}
        vlm_latency, vlm_std = benchmark_latency(VLMWrapper(vlm_model), vlm_inputs, device)
        results.append({'Model': 'Baseline-DirectVLM', 'Parameters (M)': vlm_params / 1e6, 'Latency (ms)': vlm_latency})
        print(f"  - Total Parameters: {vlm_params / 1e6:.2f} M")
        print(f"  - E2E Generation Latency: {vlm_latency:.2f} ± {vlm_std:.2f} ms")
    except FileNotFoundError:
        print(f"  - SKIPPING: Config not found at {VLM_CONFIG}. Make sure the file exists.")

    # --- 5. Benchmark Baseline-VLM (ViT-T5) --- 
    print("\n[5] Benchmarking Baseline-VLM (ViT-T5) Model...")
    T5_VLM_CONFIG = 'configs/config_direct_vlm.yaml'
    try:
        with open(T5_VLM_CONFIG, 'r') as f:
            t5_vlm_config = yaml.safe_load(f)

        t5_vlm_model = DirectVLM(
            vision_model_name=t5_vlm_config['model']['vision_model_name'],
            text_model_name=t5_vlm_config['model']['text_model_name']
        )
        t5_vlm_tokenizer = AutoTokenizer.from_pretrained(t5_vlm_config['model']['text_model_name'])
        
        t5_vlm_params = count_parameters(t5_vlm_model)
        t5_vlm_inputs = {'pixel_values': sample_pixel_values, 'tokenizer': t5_vlm_tokenizer}
        t5_vlm_latency, t5_vlm_std = benchmark_latency(DirectVLMWrapper(t5_vlm_model), t5_vlm_inputs, device)
        results.append({'Model': 'Baseline-VLM (ViT-T5)', 'Parameters (M)': t5_vlm_params / 1e6, 'Latency (ms)': t5_vlm_latency})
        print(f"  - Total Parameters: {t5_vlm_params / 1e6:.2f} M")
        print(f"  - E2E Generation Latency: {t5_vlm_latency:.2f} ± {t5_vlm_std:.2f} ms")
    except FileNotFoundError:
        print(f"  - SKIPPING: Config not found at {T5_VLM_CONFIG}. Make sure the file exists.")

    # --- SUMMARY ---
    print("\n" + "="*50)
    print("Benchmark Summary")
    print("="*50)
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))
    summary_df.to_csv("benchmark_results.csv", index=False)
    print("\nResults saved to benchmark_results.csv")
    print("="*50)

if __name__ == '__main__':
    main()