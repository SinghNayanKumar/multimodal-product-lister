import torch
import torch.nn as nn
import yaml
import argparse
import os
from tqdm import tqdm

from src.data.dataloader import create_dataloaders
from src.models.baselines.siloed_model import SiloedModel # Import our new siloed model

def calculate_loss(outputs, batch, task):
    """
    A flexible loss calculator for the siloed models.
    Unlike the CompositeLoss which combines multiple losses, this function only
    calculates the loss for the single, active task.
    """
    if task == 'price':
        loss_fn = nn.MSELoss()
        return loss_fn(outputs['price_pred'], batch['price_target'])
    
    elif task == 'attributes':
        loss_fn = nn.CrossEntropyLoss()
        attr_logits = outputs['attribute_logits']
        total_attribute_loss = 0
        
        # Average the loss across all attributes
        for attr_name, logits in attr_logits.items():
            target = batch[f"{attr_name}_target"]
            total_attribute_loss += loss_fn(logits, target)
        
        return total_attribute_loss / len(attr_logits) if attr_logits else 0
    return 0

def train_one_epoch(model, dataloader, optimizer, device, task):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Training Siloed ({task})"):
        # Move batch to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
        
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = calculate_loss(outputs, batch, task)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, device, task):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validating Siloed ({task})"):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
            outputs = model(**batch)
            loss = calculate_loss(outputs, batch, task)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    task = config['model']['task']
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Starting siloed training for task '{task}' on device: {device}")

    train_loader, val_loader, mappings = create_dataloaders(config)
    
    # ANNOTATION: We only pass the mappings dictionary if the task is 'attributes',
    # as the price model doesn't need it.
    model = SiloedModel(config, mappings if task == 'attributes' else None).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    output_dir = os.path.join(config['output_dir'], config['experiment_name'])
    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(config['training']['epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['training']['epochs']} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, task)
        val_loss = validate_one_epoch(model, val_loader, device, task)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'model_best.pth'))
            print("Saved new best model.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a single-task (siloed) model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()
    main(args.config)
