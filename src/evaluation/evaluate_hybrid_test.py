import torch
import yaml
import pandas as pd
import numpy as np
import argparse
import os
import joblib
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data.test_dataloader import create_test_dataloader
from src.models.baselines.siloed_model import SiloedModel
from src.models.baselines.tabular_price_model import TabularPriceModel

def get_attribute_predictions(model, dataloader, device, mappings):
    """
    Runs inference with the siloed attribute model to generate predictions.
    """
    model.eval()
    all_preds = {attr: [] for attr in mappings.keys()}
    inverse_mappings = {attr: {i: label for label, i in mapping.items()} for attr, mapping in mappings.items()}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Stage 1: Generating attribute predictions"):
            pixel_values = batch['pixel_values'].to(device)
            outputs = model(pixel_values=pixel_values)
            
            for attr_name, logits in outputs['attribute_logits'].items():
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                decoded_preds = [inverse_mappings[attr_name].get(p, 'Unknown') for p in preds]
                all_preds[attr_name].extend(decoded_preds)
                
    return pd.DataFrame(all_preds)

def main(args):
    # --- 1. Load Config and Test Data ---
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')

    # --- FIX: Simplified to a single, efficient call to the dataloader function ---
    test_loader, _, mappings, _ = create_test_dataloader(config)
    
    test_df = pd.read_csv(os.path.join(config['data']['processed_dir'], config['data']['test_csv']))
    print(f"Loaded test dataloader with {len(test_df)} records.")

    # --- 2. Load Trained Models ---
    print(f"Loading Stage 1 Vision Model from: {args.vision_model_path}")
    vision_model = SiloedModel(config, mappings).to(device)
    vision_model.load_state_dict(torch.load(args.vision_model_path, map_location=device))

    print(f"Loading Stage 2 Tabular Model from: {args.tabular_model_path}")
    tabular_model = joblib.load(args.tabular_model_path)
    
    # --- 3. Run the Full Inference Pipeline ---
    X_test_tabular = get_attribute_predictions(vision_model, test_loader, device, mappings)
    
    print("Stage 2: Predicting prices with the tabular model...")
    y_test_pred_log = tabular_model.predict(X_test_tabular)
    y_test_pred = np.expm1(y_test_pred_log)

    y_test_true = test_df['price'].iloc[:len(y_test_pred)]

    # --- 4. Calculate and Report Final Metrics ---
    mae = mean_absolute_error(y_test_true, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    r2 = r2_score(y_test_true, y_test_pred)
    
    print("\n--- [FINAL] Hybrid Model (Vision -> XGBoost) TEST SET Performance ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R²): {r2:.4f}")
    print("--- This is the data to compare against Ours-MTL in your paper. ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the full hybrid baseline on the TEST set.")
    parser.add_argument('--config_path', type=str, default='configs/exp1_1_siloed_attributes.yaml', help='Path to the config file for the vision attribute model.')
    parser.add_argument('--vision_model_path', type=str, default='results/exp1_1_siloed_attributes/model_best.pth', help='Path to the trained Stage 1 vision model checkpoint (.pth).')
    parser.add_argument('--tabular_model_path', type=str, default='models/hybrid_baseline/tabular_price_model.joblib', help='Path to the trained Stage 2 tabular model (.joblib).')
    args = parser.parse_args()
    main(args)