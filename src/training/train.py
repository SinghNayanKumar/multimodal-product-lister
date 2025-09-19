import torch
import yaml
import argparse
import os
from tqdm import tqdm
import wandb
import numpy as np
from sklearn.metrics import mean_absolute_error, f1_score
from PIL import Image

from src.data.dataloader import create_dataloaders
from src.models.multitask_model import MultitaskModel
from src.training.loss import CompositeLoss

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train() #Sets the model to training mode. This enables features like dropout.
    total_loss = 0
    # Create a dictionary to accumulate losses for logging
    epoch_loss_dict = {"price_loss": 0, "attribute_loss": 0, "text_loss": 0}
    # ANNOTATION: tqdm provides a convenient progress bar for tracking progress through the batches.
    for batch in tqdm(dataloader, desc="Training"):
        # ANNOTATION: It's crucial to move all data tensors to the same device (CPU or GPU) as the model.
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)

        optimizer.zero_grad() #  Resets gradients from the previous step.
        
        outputs = model(
            pixel_values=batch['pixel_values'],
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss_dict = loss_fn(outputs, batch)
        loss = loss_dict['total_loss'] # ANNOTATION: We only need the combined total_loss for backpropagation.
        
        loss.backward() # ANNOTATION: Computes gradients for all model parameters.
        optimizer.step() # ANNOTATION: Updates model parameters based on the computed gradients.
        
        total_loss += loss.item() # .item() gets the scalar value of the loss tensor.
        # Accumulate component losses for logging
        for k in epoch_loss_dict.keys():
            epoch_loss_dict[k] += loss_dict[k]
            
    # ANNOTATION: Return the average losses for the epoch.
    avg_total_loss = total_loss / len(dataloader)
    avg_loss_dict = {f"train_{k}": v / len(dataloader) for k, v in epoch_loss_dict.items()}
    avg_loss_dict['train_total_loss'] = avg_total_loss
    return avg_loss_dict

def validate_one_epoch(model, dataloader, loss_fn, device, attribute_mappers):
    model.eval() # ANNOTATION: Sets the model to evaluation mode. This disables dropout and affects batch normalization layers.
    total_loss = 0
    
    # ANNOTATION: Lists to store all predictions and targets for metric calculation.
    all_price_preds, all_price_targets = [], []
    attr_preds, attr_targets = {k: [] for k in attribute_mappers.keys()}, {k: [] for k in attribute_mappers.keys()}

    # ANNOTATION: `torch.no_grad()` is a context manager that disables gradient calculation.
    # This is essential for validation as it reduces memory consumption and speeds up computation.
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

            # --- Collect predictions and targets for metrics ---
            # Price
            all_price_preds.extend(outputs['price_pred'].cpu().numpy())
            all_price_targets.extend(batch['price_target'].cpu().numpy())
            # Attributes
            for attr_name, logits in outputs['attribute_logits'].items():
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                targets = batch[f"{attr_name}_target"].cpu().numpy()
                attr_preds[attr_name].extend(preds)
                attr_targets[attr_name].extend(targets)

    # --- Calculate metrics ---
    # Price MAE (on the original scale)
    price_preds_orig_scale = np.expm1(np.array(all_price_preds))
    price_targets_orig_scale = np.expm1(np.array(all_price_targets))
    mae = mean_absolute_error(price_targets_orig_scale, price_preds_orig_scale)

    # Attribute F1-Score (Macro Average)
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

def log_prediction_table(model, dataloader, tokenizer, attribute_mappers, device):
    """Logs a wandb.Table with model predictions for visual inspection."""
    model.eval()
    # Get a single batch
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        return None

    pixel_values = batch['pixel_values'].to(device)
    
    # Use the model's own predict method for end-to-end generation
    outputs = model.predict(pixel_values, tokenizer, attribute_mappers)

    # For demonstration, we'll just log the first item of the batch
    img_tensor = pixel_values[0].cpu()
    # De-normalize for visualization if necessary (assuming standard normalization)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = img_tensor * std + mean
    img = Image.fromarray((img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

    predicted_price = outputs['predicted_price']
    true_price = np.expm1(batch['price_target'][0].item())
    
    table = wandb.Table(columns=["Image", "Generated Text", "Predicted Price", "True Price"])
    table.add_data(wandb.Image(img), outputs['generated_text'], f"${predicted_price:.2f}", f"${true_price:.2f}")
    return table

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- W&B Initialization ---
    wandb.init(
        project=config.get('wandb', {}).get('project', 'multimodal-product-lister'),
        name=config['experiment_name'],
        config=config
    )

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, mappings = create_dataloaders(config)
    model = MultitaskModel(config, mappings).to(device)
    loss_fn = CompositeLoss(config['training']['loss_weights'])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['training']['learning_rate']))

    wandb.watch(model, loss_fn, log='all', log_freq=100)

    output_dir = os.path.join(config['output_dir'], config['experiment_name'])
    os.makedirs(output_dir, exist_ok=True)
    
    best_val_loss = float('inf')

    for epoch in range(config['training']['epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['training']['epochs']} ---")
        train_log_data = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_metrics = validate_one_epoch(model, val_loader, loss_fn, device, mappings)
        
        # Log metrics to WandB
        wandb.log({**train_log_data, **val_metrics}, step=epoch)

        print(f"Epoch {epoch+1}: Train Loss = {train_log_data['train_total_loss']:.4f}, Val Loss = {val_metrics['val_total_loss']:.4f}, Val MAE = {val_metrics['val_price_mae']:.2f}, Val F1 = {val_metrics['val_attribute_f1_macro']:.4f}")

        if val_metrics['val_total_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_total_loss']
            save_path = os.path.join(output_dir, 'model_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model to {save_path}")
            # ANNOTATION: Saving best model as a W&B artifact for versioning and easy retrieval.
            artifact = wandb.Artifact(f"{config['experiment_name']}-best", type='model')
            artifact.add_file(save_path)
            wandb.log_artifact(artifact)

        # Log a table of qualitative predictions
        prediction_table = log_prediction_table(model, val_loader, wandb.config.tokenizer, mappings, device)
        if prediction_table:
            wandb.log({"validation_predictions": prediction_table}, step=epoch)

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()
    main(args.config)