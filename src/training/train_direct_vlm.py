# FILE: src/training/train_direct_vlm.py

import torch
import yaml
import argparse
import os
import sys
from tqdm import tqdm
from PIL import Image
import numpy as np

# ANNOTATION: This helper code makes the script runnable from anywhere.
# It adds the project's root directory to the Python path, allowing
# for absolute imports like `from src.data...` to work correctly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Add error handling for wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, skipping logging")

# ANNOTATION: Imports are updated to use the new absolute paths from the project root.
from src.data.dataloader_direct_vlm import create_dataloaders_for_direct_vlm
from src.models.baselines.direct_vlm_model import DirectVLM

def train_one_epoch(model, dataloader, optimizer, device):
    # --- This function's code remains the same ---
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return {'train_loss': avg_loss}

def validate_one_epoch(model, dataloader, device):
    # --- This function's code remains the same ---
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return {'val_loss': avg_loss}

def log_prediction_table(model, dataloader, tokenizer, device, num_examples=5):
    # --- This function's code remains the same ---
    if not WANDB_AVAILABLE: return None
    model.eval()
    try: batch = next(iter(dataloader))
    except StopIteration: return None
    num_examples = min(num_examples, batch['pixel_values'].shape[0])
    pixel_values = batch['pixel_values'][:num_examples].to(device)
    gt_jsons = batch['ground_truth_json'][:num_examples]
    generated_jsons = model.generate(pixel_values, tokenizer)
    table = wandb.Table(columns=["Image", "Generated JSON", "Ground-Truth JSON"])
    for i in range(num_examples):
        img_tensor = pixel_values[i].cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = img_tensor * std + mean
        img = Image.fromarray((img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        table.add_data(wandb.Image(img), generated_jsons[i], gt_jsons[i])
    return table

def main(config_path):
    # --- This function's code remains the same ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if WANDB_AVAILABLE:
        wandb.init(project=config.get('wandb', {}).get('project', 'multimodal-product-lister'), name=config['experiment_name'], config=config)

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, tokenizer = create_dataloaders_for_direct_vlm(config)
    model = DirectVLM(config['model']['vision_model_name'], config['model']['text_model_name']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['training']['learning_rate']))

    output_dir = os.path.join(config['output_dir'], config['experiment_name'])
    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(config['training']['epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['training']['epochs']} ---")
        train_log = train_one_epoch(model, train_loader, optimizer, device)
        val_log = validate_one_epoch(model, val_loader, device)
        if WANDB_AVAILABLE:
            wandb.log({**train_log, **val_log}, step=epoch)
        print(f"Epoch {epoch+1}: Train Loss = {train_log['train_loss']:.4f}, Val Loss = {val_log['val_loss']:.4f}")

        if val_log['val_loss'] < best_val_loss:
            best_val_loss = val_log['val_loss']
            save_path = os.path.join(output_dir, 'model_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model to {save_path}")

        if (epoch + 1) % config['training'].get('log_image_freq', 5) == 0:
            prediction_table = log_prediction_table(model, val_loader, tokenizer, device)
            if prediction_table and WANDB_AVAILABLE:
                wandb.log({"validation_predictions": prediction_table}, step=epoch)

    if WANDB_AVAILABLE: wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ANNOTATION: The help text now points to the new config location.
    parser.add_argument('--config', type=str, required=True, help='Path to the config file (e.g., configs/config_direct_vlm.yaml).')
    args = parser.parse_args()
    main(args.config)