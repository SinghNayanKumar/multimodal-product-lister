import torch
import yaml
import argparse
import os
from tqdm import tqdm


from src.data.dataloader import create_dataloaders
from src.models.multitask_model import MultitaskModel
from src.training.loss import CompositeLoss

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train() #Sets the model to training mode. This enables features like dropout.
    total_loss = 0
    # ANNOTATION: tqdm provides a convenient progress bar for tracking progress through the batches.
    for batch in tqdm(dataloader, desc="Training"):
        # ANNOTATION: It's crucial to move all data tensors to the same device (CPU or GPU) as the model.
        for key, value in batch.items():
            batch[key] = value.to(device)

        optimizer.zero_grad() #  Resets gradients from the previous step.
        
        # ANNOTATION: This is a clean way to pass data to the model. The `**batch` syntax unpacks the
        # dictionary into keyword arguments, assuming the keys in `batch` match the argument names
        # in the model's `forward` method (e.g., image_tensor=..., input_ids=...).
        outputs = model(**batch)
        
        loss_dict = loss_fn(outputs, batch)
        loss = loss_dict['total_loss'] # ANNOTATION: We only need the combined total_loss for backpropagation.
        
        loss.backward() # ANNOTATION: Computes gradients for all model parameters.
        optimizer.step() # ANNOTATION: Updates model parameters based on the computed gradients.
        
        total_loss += loss.item() # .item() gets the scalar value of the loss tensor.
    return total_loss / len(dataloader) # ANNOTATION: Returns the average loss for the epoch.

def validate_one_epoch(model, dataloader, loss_fn, device):
    model.eval() # ANNOTATION: Sets the model to evaluation mode. This disables dropout and affects batch normalization layers.
    total_loss = 0
    # ANNOTATION: `torch.no_grad()` is a context manager that disables gradient calculation.
    # This is essential for validation as it reduces memory consumption and speeds up computation.
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
    # ANNOTATION: Using a YAML config file is a best practice. It separates hyperparameters from code,
    # making experiments easy to configure, reproduce, and track.
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ANNOTATION: Setting up all the core components based on the configuration.
    train_loader, val_loader, mappings = create_dataloaders(config)
    model = MultitaskModel(config, mappings).to(device)
    loss_fn = CompositeLoss(config['training']['loss_weights'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # ANNOTATION: Defines a structured output directory for saving model checkpoints and logs.
    output_dir = os.path.join(config['output_dir'], config['experiment_name'])
    os.makedirs(output_dir, exist_ok=True)
    
    best_val_loss = float('inf')

    # --- The Main Training Loop ---
    for epoch in range(config['training']['epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['training']['epochs']} ---")
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # ANNOTATION: This is a standard checkpointing strategy. We monitor the validation loss
        # and save the model only when it improves. This ensures we keep the model that generalizes
        # best to unseen data, helping to prevent overfitting.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'model_best.pth'))
            print("Saved new best model.")

if __name__ == '__main__':
    # ANNOTATION: Using `argparse` makes the script runnable and configurable from the command line,
    # which is essential for server-based training and experimentation.
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()
    main(args.config)