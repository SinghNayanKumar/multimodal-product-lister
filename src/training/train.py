import torch
import yaml
import argparse
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_absolute_error, f1_score
from PIL import Image

# Add error handling for wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, skipping logging")

from src.data.dataloader import create_dataloaders
from src.models.multitask_model import MultitaskModel
from src.training.loss import CompositeLoss

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    epoch_loss_dict = {"price_loss": 0, "attribute_loss": 0, "text_loss": 0}
    
    for batch in tqdm(dataloader, desc="Training"):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)

        optimizer.zero_grad()
        
        outputs = model(
            pixel_values=batch['pixel_values'],
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss_dict = loss_fn(outputs, batch)
        loss = loss_dict['total_loss']
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        for k in epoch_loss_dict.keys():
            if loss_dict.get(k) is not None:
                 epoch_loss_dict[k] += loss_dict[k]
            
    avg_total_loss = total_loss / len(dataloader)
    avg_loss_dict = {f"train_{k}": v / len(dataloader) for k, v in epoch_loss_dict.items()}
    avg_loss_dict['train_total_loss'] = avg_total_loss
    return avg_loss_dict

def validate_one_epoch(model, dataloader, loss_fn, device, attribute_mappers):
    model.eval()
    total_loss = 0
    
    all_price_preds, all_price_targets = [], []
    attr_preds, attr_targets = {k: [] for k in attribute_mappers.keys()}, {k: [] for k in attribute_mappers.keys()}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
            
            outputs = model(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )

            loss_dict = loss_fn(outputs, batch)
            loss = loss_dict['total_loss']
            total_loss += loss.item()

            # Collect predictions and targets for metrics
            all_price_preds.extend(outputs['price_pred'].cpu().numpy())
            all_price_targets.extend(batch['price_target'].cpu().numpy())
            
            for attr_name, logits in outputs['attribute_logits'].items():
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                targets = batch[f"{attr_name}_target"].cpu().numpy()
                attr_preds[attr_name].extend(preds)
                attr_targets[attr_name].extend(targets)

    # Calculate metrics - FIXED: Use predictions vs targets
    price_preds_orig_scale = np.expm1(np.array(all_price_preds))
    price_targets_orig_scale = np.expm1(np.array(all_price_targets))
    mae = mean_absolute_error(price_targets_orig_scale, price_preds_orig_scale)  # FIXED!

    f1_scores = []
    for attr_name in attribute_mappers.keys():
        f1 = f1_score(attr_targets[attr_name], attr_preds[attr_name], average='macro', zero_division=0)
        f1_scores.append(f1)
    avg_f1_macro = np.mean(f1_scores) if f1_scores else 0
    
    metrics = {
        'val_total_loss': total_loss / len(dataloader),
        'val_price_mae': mae,
        'val_attribute_f1_macro': avg_f1_macro
    }
    return metrics

def log_prediction_table(model, dataloader, tokenizer, attribute_mappers, device, num_examples=5):
    """Logs a wandb.Table with model predictions for visual inspection."""
    if not WANDB_AVAILABLE:
        return None
        
    model.eval()
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        return None

    num_examples = min(num_examples, batch['pixel_values'].shape[0])
    
    pixel_values = batch['pixel_values'][:num_examples].to(device)
    price_targets = batch['price_target'][:num_examples]

    # FIXED: Handle return type consistency
    outputs = model.predict(pixel_values, tokenizer, attribute_mappers)

    # --- ROBUSTNESS FIX ---
    # Check if the prediction output is valid. If it's None or an empty list,
    # it means something went wrong for this batch. Print a warning and skip logging.
    if not outputs:
        print("\n[Warning] `model.predict()` returned no output for the logging batch. Skipping wandb table creation for this epoch.\n")
        return None
    # --- END OF FIX ---

    if not isinstance(outputs, list):
        outputs = [outputs]
    
    table = wandb.Table(columns=["Image", "Generated Text", "Predicted Price", "True Price"])

    for i, output in enumerate(outputs[:num_examples]):
        img_tensor = pixel_values[i].cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = img_tensor * std + mean
        img = Image.fromarray((img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

        predicted_price = output.get('predicted_price', 0.0) #Fix
        true_price = np.expm1(price_targets[i].item())
        
        table.add_data(
            wandb.Image(img), 
            output.get('generated_text', '[GENERATION FAILED]'), #Fix 
            f"${predicted_price:.2f}", 
            f"${true_price:.2f}"
        )
        
    return table

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if WANDB_AVAILABLE:
        wandb.init(
            project=config.get('wandb', {}).get('project', 'multimodal-product-lister'),
            name=config['experiment_name'],
            config=config
        )

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, mappings, tokenizer = create_dataloaders(config)
    
    model = MultitaskModel(config, mappings).to(device)
    loss_fn = CompositeLoss(config['training']['loss_weights'])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['training']['learning_rate']))

    if WANDB_AVAILABLE:
        wandb.watch(model, loss_fn, log='all', log_freq=100)

    output_dir = os.path.join(config['output_dir'], config['experiment_name'])
    os.makedirs(output_dir, exist_ok=True)
    
    best_val_loss = float('inf')

    for epoch in range(config['training']['epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['training']['epochs']} ---")
        train_log_data = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_metrics = validate_one_epoch(model, val_loader, loss_fn, device, mappings)
        
        if WANDB_AVAILABLE:
            wandb.log({**train_log_data, **val_metrics}, step=epoch)

        print(f"Epoch {epoch+1}: Train Loss = {train_log_data['train_total_loss']:.4f}, Val Loss = {val_metrics['val_total_loss']:.4f}, Val MAE = {val_metrics['val_price_mae']:.2f}, Val F1 = {val_metrics['val_attribute_f1_macro']:.4f}")

        if val_metrics['val_total_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_total_loss']
            save_path = os.path.join(output_dir, 'model_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model to {save_path}")
            
            if WANDB_AVAILABLE:
                artifact = wandb.Artifact(f"{config['experiment_name']}-best", type='model')
                artifact.add_file(save_path)
                wandb.log_artifact(artifact)

        if (epoch + 1) % config['training'].get('log_image_freq', 5) == 0:
            prediction_table = log_prediction_table(model, val_loader, tokenizer, mappings, device)
            if prediction_table and WANDB_AVAILABLE:
                wandb.log({"validation_predictions": prediction_table}, step=epoch)

    if WANDB_AVAILABLE:
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()
    main(args.config)