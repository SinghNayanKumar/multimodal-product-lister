import torch
import torch.nn as nn
import yaml
import argparse
import os
from tqdm import tqdm

from src.data.dataloader import create_dataloaders
from src.models.vision_attribute_model import VisionAttributeModel

def calculate_attribute_loss(logits_dict, batch):
    """Calculates the average cross-entropy loss across all attributes."""
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    num_attrs = 0
    for attr_name, logits in logits_dict.items():
        target = batch[f"{attr_name}_target"]
        total_loss += loss_fn(logits, target)
        num_attrs += 1
    return total_loss / num_attrs if num_attrs > 0 else 0

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        for key in ['image_tensor'] + [k for k in batch if k.endswith('_target')]:
            batch[key] = batch[key].to(device)

        optimizer.zero_grad()
        outputs = model(image_tensor=batch['image_tensor'])
        loss = calculate_attribute_loss(outputs['attribute_logits'], batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            for key in ['image_tensor'] + [k for k in batch if k.endswith('_target')]:
                batch[key] = batch[key].to(device)
            outputs = model(image_tensor=batch['image_tensor'])
            loss = calculate_attribute_loss(outputs['attribute_logits'], batch)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, mappings, _ = create_dataloaders(config)
    
    model = VisionAttributeModel(config, mappings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['training']['learning_rate']))

    output_dir = os.path.join(config['output_dir'], f"{config['experiment_name']}_stage1")
    os.makedirs(output_dir, exist_ok=True)
    
    best_val_loss = float('inf')

    for epoch in range(config['training']['epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['training']['epochs']} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, device)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(output_dir, 'attribute_model_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best Stage 1 model to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()
    main(args.config)