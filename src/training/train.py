import torch
import yaml
import argparse
import os
from tqdm import tqdm

from src.data.dataloader import create_dataloaders
from src.models.multitask_model import MultitaskModel
from src.training.loss import CompositeLoss

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        # Move batch to device
        for key, value in batch.items():
            batch[key] = value.to(device)

        optimizer.zero_grad()
        outputs = model(**batch)
        loss_dict = loss_fn(outputs, batch)
        loss = loss_dict['total_loss']
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            for key, value in batch.items():
                batch[key] = value.to(device)
            
            outputs = model(**batch)
            loss_dict = loss_fn(outputs, batch)
            loss = loss_dict['total_loss']
            total_loss += loss.item()
    return total_loss / len(dataloader)


def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, mappings = create_dataloaders(config)
    
    model = MultitaskModel(config, mappings).to(device)
    loss_fn = CompositeLoss(config['training']['loss_weights'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    output_dir = os.path.join(config['output_dir'], config['experiment_name'])
    os.makedirs(output_dir, exist_ok=True)
    
    best_val_loss = float('inf')

    for epoch in range(config['training']['epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['training']['epochs']} ---")
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'model_best.pth'))
            print("Saved new best model.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()
    main(args.config)