import torch
import yaml
import argparse
import os
import pandas as pd
from tqdm import tqdm

from src.data.dataloader import create_dataloaders
from src.models.vision_attribute_model import VisionAttributeModel
from src.models.tabular_price_model import TabularPriceModel

def generate_attribute_predictions(model, dataloader, mappings, device):
    """Run inference with the Stage 1 model to create a tabular dataset."""
    model.eval()
    predictions = []
    true_prices = []
    
    # Create inverse mappings to convert class indices back to labels
    inverse_mappings = {attr: {i: label for label, i in class_map.items()} 
                        for attr, class_map in mappings.items()}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Attribute Predictions"):
            image_tensor = batch['image_tensor'].to(device)
            outputs = model(image_tensor=image_tensor)
            
            # Collect true prices for the target variable
            true_prices.extend(batch['price_target'].numpy())

            # For each item in the batch, get the predicted attribute
            batch_size = image_tensor.size(0)
            for i in range(batch_size):
                item_preds = {}
                for attr_name, logits in outputs['attribute_logits'].items():
                    pred_index = torch.argmax(logits[i]).item()
                    pred_label = inverse_mappings[attr_name][pred_index]
                    item_preds[attr_name] = pred_label
                predictions.append(item_preds)

    df = pd.DataFrame(predictions)
    df['price'] = true_prices
    return df

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    output_dir = os.path.join(config['output_dir'], f"{config['experiment_name']}_stage2")
    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: Generate predictions using the trained Stage 1 model ---
    print("--- Stage 1: Loading Attribute Model and Generating Predictions ---")
    train_loader, _, mappings, _  = create_dataloaders(config)
    
    # Load the trained attribute model
    attribute_model = VisionAttributeModel(config, mappings).to(device)
    stage1_model_path = os.path.join(config['output_dir'], f"{config['experiment_name']}_stage1", 'attribute_model_best.pth')
    attribute_model.load_state_dict(torch.load(stage1_model_path, map_location=device))

    # Generate the tabular dataset
    prediction_df = generate_attribute_predictions(attribute_model, train_loader, mappings, device)
    
    # --- Step 2: Train the Stage 2 tabular price model ---
    print("\n--- Stage 2: Training Tabular Price Model ---")
    attribute_cols = list(mappings.keys())
    X_train = prediction_df[attribute_cols]
    y_train = prediction_df['price']

    tabular_model = TabularPriceModel(categorical_features=attribute_cols)
    tabular_model.train(X_train, y_train)
    
    # Save the trained model
    model_save_path = os.path.join(output_dir, 'tabular_price_model.joblib')
    tabular_model.save(model_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()
    main(args.config)