import torch
import yaml
import argparse
import os
import pandas as pd
from tqdm import tqdm
import copy

# --- Imports from your existing project structure ---
# Assumes the script is run from a location where 'src' is a visible package,
# or the project root is added to the PYTHONPATH.
from src.data.dataloader import create_dataloaders
from src.models.multitask_model import MultitaskModel
from src.training.loss import CompositeLoss
# We import the core training and validation functions directly from your train script
from src.train import train_one_epoch, validate_one_epoch

# --- Configuration for the Ablation Study ---
# Define the different weight combinations you want to test.
# Each dictionary represents one full training run.
WEIGHT_COMBINATIONS = [
    # --- 1. Balanced Scenarios ---
    {'price': 0.3, 'attributes': 0.3, 'text': 0.4}, # Perfectly even baseline
    {'price': 0.4, 'attributes': 0.4, 'text': 0.2},   # Balanced, prediction-focused
    
    # --- 2. Task-Focused Scenarios ---
    {'price': 0.6, 'attributes': 0.2, 'text': 0.2},   # Price-focused
    {'price': 0.2, 'attributes': 0.6, 'text': 0.2},   # Attribute-focused (from your list)
    {'price': 0.1, 'attributes': 0.1, 'text': 0.8},   # Text-focused (from your list)
    {'price': 0.1, 'attributes': 0.4, 'text': 0.5},   # Attribute + Text focused

    # --- 3. "Knock-Out" Ablation Scenarios (Crucial for the paper) ---
    {'price': 0.0, 'attributes': 0.5, 'text': 0.5},   # No Price Task: Does price-awareness help other tasks?
    {'price': 0.5, 'attributes': 0.0, 'text': 0.5},   # No Attribute Task: Can the model still perform without fine-grained labels?
    {'price': 0.5, 'attributes': 0.5, 'text': 0.0},   # No Text Task: Can we learn a good HVR without a generation objective?
]

def run_training_session(base_config, weights, root_output_dir):
    """
    Executes a complete training and validation loop for a single weight combination.

    Args:
        base_config (dict): The base configuration loaded from the YAML file.
        weights (dict): The specific loss weight combination for this run.
        root_output_dir (str): The main directory to save all ablation results.

    Returns:
        float: The best validation loss achieved during this training session.
    """
    # 1. Create a deep copy of the config to avoid side effects between runs
    config = copy.deepcopy(base_config)

    # 2. Dynamically modify the configuration for this specific run
    config['training']['loss_weights'] = weights
    
    # Create a unique, descriptive experiment name from the weights
    exp_name = f"ablation_P{weights['price']}_A{weights['attributes']}_T{weights['text']}"
    config['experiment_name'] = exp_name
    
    # Define the specific output directory for this run's artifacts
    run_output_dir = os.path.join(root_output_dir, exp_name)
    os.makedirs(run_output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"STARTING ABLATION RUN: {exp_name}")
    print(f"Weights: {weights}")
    print(f"Results will be saved in: {run_output_dir}")
    print("="*80 + "\n")

    # 3. Setup training components (device, dataloaders, model, etc.)
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # The create_dataloaders function is imported and reused
    train_loader, val_loader, mappings, _ = create_dataloaders(config)
    
    # Initialize a new model, loss function, and optimizer for each run
    model = MultitaskModel(config, mappings).to(device)
    loss_fn = CompositeLoss(config['training']['loss_weights'])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['training']['learning_rate']))

    best_val_loss = float('inf')

    # 4. Execute the training and validation loop
    for epoch in range(config['training']['epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['training']['epochs']} for {exp_name} ---")
        
        # Reuse the training logic from train.py
        train_log_data = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        
        # Reuse the validation logic from train.py
        val_metrics = validate_one_epoch(model, val_loader, loss_fn, device, mappings)
        
        current_val_loss = val_metrics['val_total_loss']
        print(f"Epoch {epoch+1} Summary: Train Loss = {train_log_data['train_total_loss']:.4f}, Val Loss = {current_val_loss:.4f}, Val MAE = {val_metrics['val_price_mae']:.2f}, Val F1 = {val_metrics['val_attribute_f1_macro']:.4f}")

        # 5. Track the best performance and save the model for this run
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            save_path = os.path.join(run_output_dir, 'model_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model for this run to {save_path} (Val Loss: {best_val_loss:.4f})")
            
    print(f"\nFINISHED ABLATION RUN: {exp_name}. Best validation loss: {best_val_loss:.4f}\n")
    return best_val_loss


def main(config_path, output_dir):
    """
    Main function to orchestrate the entire ablation study.
    """
    # Load the base configuration once
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # Prepare the root directory for all results
    os.makedirs(output_dir, exist_ok=True)
    
    summary_results = []

    # Main loop to iterate through each weight combination
    for weights in WEIGHT_COMBINATIONS:
        best_loss_for_run = run_training_session(base_config, weights, output_dir)
        
        # Log the final results for this run
        run_summary = {
            'price_weight': weights['price'],
            'attributes_weight': weights['attributes'],
            'text_weight': weights['text'],
            'best_validation_loss': best_loss_for_run
        }
        summary_results.append(run_summary)

    # After all runs are complete, create and save the summary CSV
    summary_df = pd.DataFrame(summary_results)
    summary_csv_path = os.path.join(output_dir, 'ablation_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    
    print("\n" + "*"*80)
    print("ABLATION STUDY COMPLETE!")
    print(f"Summary of results saved to: {summary_csv_path}")
    print("Final Results:")
    print(summary_df)
    print("*"*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a loss weight ablation study by training multiple models.")
    parser.add_argument('--config', type=str, required=True, help='Path to the BASE config file (e.g., configs/base_config.yaml).')
    parser.add_argument('--output_dir', type=str, default='ablation_training_results', help='Root directory to save all models and the final summary CSV.')
    args = parser.parse_args()
    
    main(args.config, args.output_dir)