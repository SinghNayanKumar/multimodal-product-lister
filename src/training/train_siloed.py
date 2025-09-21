import torch
import torch.nn as nn
import yaml
import argparse
import os
from tqdm import tqdm
import wandb
import numpy as np
from sklearn.metrics import mean_absolute_error, f1_score

from src.data.dataloader import create_dataloaders
from src.models.baselines.siloed_model import SiloedModel

def calculate_loss(outputs, batch, task):
    if task == 'price':
        loss_fn = nn.MSELoss()
        return loss_fn(outputs['price_pred'], batch['price_target'])
    elif task == 'attributes':
        loss_fn = nn.CrossEntropyLoss()
        attr_logits = outputs['attribute_logits']
        total_attribute_loss = 0
        for attr_name, logits in attr_logits.items():
            total_attribute_loss += loss_fn(logits, batch[f"{attr_name}_target"])
        return total_attribute_loss / len(attr_logits) if attr_logits else 0
    return 0

def train_one_epoch(model, dataloader, optimizer, device, task):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Training Siloed ({task})"):
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

def validate_one_epoch(model, dataloader, device, task, mappings=None):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    attr_preds, attr_targets = {k: [] for k in mappings.keys()} if mappings else {}, {k: [] for k in mappings.keys()} if mappings else {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validating Siloed ({task})"):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
            outputs = model(**batch)
            loss = calculate_loss(outputs, batch, task)
            total_loss += loss.item()
            
            if task == 'price':
                all_preds.extend(outputs['price_pred'].cpu().numpy())
                all_targets.extend(batch['price_target'].cpu().numpy())
            elif task == 'attributes':
                for attr_name, logits in outputs['attribute_logits'].items():
                    attr_preds[attr_name].extend(torch.argmax(logits, dim=-1).cpu().numpy())
                    attr_targets[attr_name].extend(batch[f"{attr_name}_target"].cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    metrics = {'val_loss': avg_loss}
    
    if task == 'price':
        preds_orig = np.expm1(np.array(all_preds))
        targets_orig = np.expm1(np.array(all_targets))
        metrics['val_price_mae'] = mean_absolute_error(targets_orig, preds_orig)
    elif task == 'attributes':
        f1_scores = [f1_score(attr_targets[k], attr_preds[k], average='macro', zero_division=0) for k in mappings.keys()]
        metrics['val_attribute_f1_macro'] = np.mean(f1_scores) if f1_scores else 0

    return metrics

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    task = config['model']['task']

    wandb.init(
        project=config.get('wandb', {}).get('project', 'multimodal-product-lister'),
        name=config['experiment_name'],
        config=config,
        group="siloed-baselines" # Group siloed runs together
    )

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Starting siloed training for task '{task}' on device: {device}")

    train_loader, val_loader, mappings, _ = create_dataloaders(config)
    
    # ANNOTATION: We only pass the mappings dictionary if the task is 'attributes',
    # as the price model doesn't need it.
    model = SiloedModel(config, mappings if task == 'attributes' else None).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['training']['learning_rate']))
    
    output_dir = os.path.join(config['output_dir'], config['experiment_name'])
    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(config['training']['epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['training']['epochs']} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, task)
        val_metrics = validate_one_epoch(model, val_loader, device, task, mappings if task == 'attributes' else None)
        
        log_data = {'train_loss': train_loss, **val_metrics}
        wandb.log(log_data, step=epoch)
        
        metric_str = f", Val MAE = {val_metrics.get('val_price_mae', 0):.2f}" if task == 'price' else f", Val F1 = {val_metrics.get('val_attribute_f1_macro', 0):.4f}"
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_metrics['val_loss']:.4f}{metric_str}")

        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            save_path = os.path.join(output_dir, 'model_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model to {save_path}")

    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a single-task (siloed) model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()
    main(args.config)