# src/evaluation/benchmark.py

import torch
import time
import numpy as np
import yaml
from PIL import Image

# --- FIX: Added missing import for dataloader creation. ---
# This is needed to load the `mappings` dictionary, which is required to initialize the models.
from src.data.dataloader import create_test_dataloaders
# ANNOTATION: Import all the models we want to compare.
from src.models.multitask_model import MultitaskModel
from src.models.baselines.siloed_model import SiloedModel
from transformers import ViTImageProcessor, AutoTokenizer
from src.data.dataloader import create_test_dataloaders

def count_parameters(model):
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_latency(model, sample_input, device, num_runs=100):
    """
    Measures the average inference latency of a model.
    
    Args:
        model: The model to benchmark.
        sample_input (dict): A dictionary of tensors to feed to the model.
        device: The device to run on.
        num_runs (int): Number of forward passes to average over.
    """
    model.eval()
    latencies = []
    
    # Warm-up runs to load model onto GPU cache
    with torch.no_grad():
        for _ in range(10):
            _ = model(**sample_input)

    with torch.no_grad():
        for _ in range(num_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            _ = model(**sample_input)
            end_event.record()
            
            torch.cuda.synchronize() # Wait for the events to complete
            latencies.append(start_event.elapsed_time(end_event)) # Time in milliseconds
            
    return np.mean(latencies), np.std(latencies)

def main():
    # --- SETUP ---
    MTL_CONFIG = 'configs/base_config.yaml'
    SILOED_ATTR_CONFIG = 'configs/exp1_1_siloed_attributes.yaml'
    SILOED_PRICE_CONFIG = 'configs/exp1_1_siloed_price.yaml'
    
    with open(MTL_CONFIG, 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load mappings from the dataloader function; we don't need the loaders themselves.
    _, _, mappings, _ = create_test_dataloaders(config)
    sample_pixel_values = torch.randn(1, 3, 224, 224).to(device)
    sample_input = {'pixel_values': sample_pixel_values}
    
    print(f"--- Benchmarking on {device} ---")

    # --- 1. Benchmark Ours-MTL ---
    print("\n[1] Benchmarking Ours-MTL Model...")
    mtl_model = MultitaskModel(config, mappings).to(device)
    mtl_params = count_parameters(mtl_model)
    # For latency, we time the `forward` pass, which is the core computation.
    mtl_latency, mtl_std = benchmark_latency(mtl_model, sample_input, device)
    print(f"  - Total Parameters: {mtl_params / 1e6:.2f} M")
    print(f"  - Inference Latency: {mtl_latency:.2f} ± {mtl_std:.2f} ms")

    # --- 2. Benchmark Baseline-Siloed ---
    print("\n[2] Benchmarking Baseline-Siloed Pipeline...")
    # Load attribute model
    with open(SILOED_ATTR_CONFIG, 'r') as f:
        attr_config = yaml.safe_load(f)
    attr_model = SiloedModel(attr_config, mappings).to(device)
    attr_params = count_parameters(attr_model)
    attr_latency, _ = benchmark_latency(attr_model, sample_input, device)
    
    # Load price model
    with open(SILOED_PRICE_CONFIG, 'r') as f:
        price_config = yaml.safe_load(f)
    price_model = SiloedModel(price_config).to(device)
    price_params = count_parameters(price_model)
    price_latency, _ = benchmark_latency(price_model, sample_input, device)

    # ANNOTATION: For a pipeline, we sum the parameters and latencies of its components.
    siloed_total_params = attr_params + price_params
    siloed_total_latency = attr_latency + price_latency
    print(f"  - Total Parameters (Attr + Price): {siloed_total_params / 1e6:.2f} M")
    print(f"  - Total Inference Latency (Sum of Stages): {siloed_total_latency:.2f} ms")

if __name__ == '__main__':
    main()