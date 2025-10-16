import torch
import yaml
import argparse
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import evaluate

# Add error handling for wandb for robustness
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, skipping logging")

from src.data.dataloader_baseline import create_baseline_dataloaders
from src.models.baseline_git import GitBaselineModel

def train_one_epoch(model, dataloader, optimizer, device):
    """
    Performs one full training epoch.
    """
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        # Move all tensors to the correct device
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        
        # Forward pass to get loss
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs['loss']
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
            
    avg_loss = total_loss / len(dataloader)
    return {'train_loss': avg_loss}

def validate_one_epoch(model, dataloader, device):
    """
    --- FAST VALIDATION ---
    This function ONLY calculates validation loss. It is designed to be run every epoch.
    """
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            total_val_loss += outputs['loss'].item()

    return {'val_loss': total_val_loss / len(dataloader)}

def evaluate_model(model, dataloader, processor, device, rouge_metric):
    """
    --- SLOW EVALUATION ---
    Performs text generation on the entire validation set to calculate ROUGE scores.
    This should be called infrequently, e.g., only at the end of training.
    """
    model.eval()
    all_predictions = []
    all_references = []
    with torch.no_grad():
        val_df = dataloader.dataset.df
        for batch in tqdm(dataloader, desc="Evaluating ROUGE"):
            pixel_values = batch['pixel_values'].to(device)
            predictions = model.predict(pixel_values, processor)
            all_predictions.extend(predictions)
            
            # Since the dataloader is not shuffled, we can rely on its order
            # This part of the code is implicitly handled by iterating through the dataloader
            
        # Create references in the same order as the dataloader
        for _, row in val_df.iterrows():
            reference = f"title: {row['name']} | description: {row['description']}"
            all_references.append(reference)

    results = rouge_metric.compute(predictions=all_predictions, references=all_references)
    return {f"eval_{k}": v for k, v in results.items()}

def log_prediction_table(model, dataloader, processor, device, num_examples=5):
    """
    Logs a table of qualitative prediction examples to Weights & Biases.
    """
    if not WANDB_AVAILABLE: return None
    model.eval()
    try: batch = next(iter(dataloader))
    except StopIteration: return None 

    pixel_values = batch['pixel_values'][:num_examples].to(device)
    val_df = dataloader.dataset.df
    true_rows = val_df.iloc[:num_examples]

    predictions = model.predict(pixel_values, processor)

    table = wandb.Table(columns=["Image", "Generated Text", "Ground Truth Text"])
    for i, pred_text in enumerate(predictions):
        img_tensor = pixel_values[i].cpu()
        img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())
        img = Image.fromarray((img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        true_row = true_rows.iloc[i]
        reference = f"title: {true_row['name']} | description: {true_row['description']}"
        table.add_data(wandb.Image(img), pred_text, reference)
        
    return table

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = os.path.join(config['output_dir'], config['experiment_name'])
    os.makedirs(output_dir, exist_ok=True)

    if WANDB_AVAILABLE:
        # Use resume="auto" for the most seamless experience.
        # It will automatically find the run ID from the local wandb directory.
        wandb.init(
            project=config.get('wandb', {}).get('project', 'multimodal-product-lister'),
            name=config['experiment_name'],
            config=config,
            resume="auto",
            dir=output_dir # Store wandb logs in the experiment folder
        )

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, processor = create_baseline_dataloaders(config)
    model = GitBaselineModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['training']['learning_rate']))
    rouge = evaluate.load('rouge')
    
    # --- Checkpoint Loading Logic ---
    start_epoch = 0
    best_val_loss = float('inf')
    checkpoint_path = os.path.join(output_dir, 'checkpoint.pth')
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}. Best validation loss so far: {best_val_loss:.4f}")

    # --- Main Training Loop ---
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['training']['epochs']} ---")
        train_log_data = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = validate_one_epoch(model, val_loader, device)
        
        if WANDB_AVAILABLE:
            wandb.log({**train_log_data, **val_metrics}, step=epoch)

        print(f"Epoch {epoch+1}: Train Loss = {train_log_data['train_loss']:.4f}, Val Loss = {val_metrics['val_loss']:.4f}")

        # --- Checkpoint Saving Logic ---
        is_best = val_metrics['val_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['val_loss']
            best_model_path = os.path.join(output_dir, 'model_best.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model to {best_model_path} (Val Loss: {best_val_loss:.4f})")
        
        # Save a comprehensive checkpoint every epoch for resuming
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, checkpoint_path)

        # Log qualitative examples infrequently to avoid slowdown
        if (epoch + 1) % config['training'].get('log_image_freq', 5) == 0:
            print("Logging prediction examples to W&B...")
            prediction_table = log_prediction_table(model, val_loader, processor, device)
            if prediction_table and WANDB_AVAILABLE:
                wandb.log({"validation_predictions": prediction_table}, step=epoch)

    # --- Final Evaluation after Training ---
    print("\n--- Final Evaluation on Best Model ---")
    best_model_path = os.path.join(output_dir, 'model_best.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        final_rouge_scores = evaluate_model(model, val_loader, processor, device, rouge)
        print(f"Final ROUGE Scores: {final_rouge_scores}")
        if WANDB_AVAILABLE:
            wandb.log(final_rouge_scores)
    else:
        print("Best model not found, skipping final evaluation.")

    if WANDB_AVAILABLE:
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()
    main(args.config)