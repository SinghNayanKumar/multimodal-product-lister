import torch
import yaml
import argparse
import os
from tqdm import tqdm
import numpy as np
from PIL import Image

# Add error handling for wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, skipping logging")

from src.data.dataloader import create_test_dataloaders
from src.models.baselines.direct_vlm import DirectVLM

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training DirectVLM"):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)

        optimizer.zero_grad()
        
        # DirectVLM only needs pixel_values and labels
        outputs = model(
            pixel_values=batch['pixel_values'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
            
    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating DirectVLM"):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
            
            outputs = model(
                pixel_values=batch['pixel_values'],
                labels=batch['labels']
            )
            
            loss = outputs.loss
            total_loss += loss.item()

    return total_loss / len(dataloader)

def log_prediction_table(model, dataloader, tokenizer, device, num_examples=5):
    """Logs a wandb.Table with DirectVLM predictions for visual inspection."""
    if not WANDB_AVAILABLE:
        return None
        
    model.eval()
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        return None

    num_examples = min(num_examples, batch['pixel_values'].shape[0])
    
    pixel_values = batch['pixel_values'][:num_examples].to(device)
    true_labels = batch['labels'][:num_examples]

    # Generate predictions
    generated_texts = model.predict(pixel_values, tokenizer)
    
    # Decode true labels for comparison
    true_texts = tokenizer.batch_decode(true_labels, skip_special_tokens=True)
    
    table = wandb.Table(columns=["Image", "Generated Text", "True Text"])

    for i in range(len(generated_texts)):
        # De-normalize image for visualization
        img_tensor = pixel_values[i].cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = img_tensor * std + mean
        img = Image.fromarray((img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        
        table.add_data(
            wandb.Image(img), 
            generated_texts[i], 
            true_texts[i]
        )
        
    return table

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if WANDB_AVAILABLE:
        wandb.init(
            project=config.get('wandb', {}).get('project', 'multimodal-product-lister'),
            name=config['experiment_name'],
            config=config,
            group="direct-vlm-baseline"  # Group DirectVLM runs together
        )

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, mappings, tokenizer = create_test_dataloaders(config)
    
    model = DirectVLM(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['training']['learning_rate']))

    if WANDB_AVAILABLE:
        wandb.watch(model, log='all', log_freq=100)

    output_dir = os.path.join(config['output_dir'], config['experiment_name'])
    os.makedirs(output_dir, exist_ok=True)
    
    best_val_loss = float('inf')

    for epoch in range(config['training']['epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['training']['epochs']} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, device)
        
        if WANDB_AVAILABLE:
            wandb.log({'train_loss': train_loss, 'val_loss': val_loss}, step=epoch)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(output_dir, 'model_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model to {save_path}")
            
            if WANDB_AVAILABLE:
                artifact = wandb.Artifact(f"{config['experiment_name']}-best", type='model')
                artifact.add_file(save_path)
                wandb.log_artifact(artifact)

        # Log prediction examples
        if (epoch + 1) % config['training'].get('log_image_freq', 5) == 0:
            prediction_table = log_prediction_table(model, val_loader, tokenizer, device)
            if prediction_table and WANDB_AVAILABLE:
                wandb.log({"validation_predictions": prediction_table}, step=epoch)

    if WANDB_AVAILABLE:
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DirectVLM baseline model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()
    main(args.config)