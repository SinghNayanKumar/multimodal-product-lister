# src/evaluation/run_hybrid_baseline.py

import torch
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data.dataloader import create_dataloaders
from src.models.baselines.siloed_model import SiloedModel
from src.models.baselines.tabular_price_model import TabularPriceModel

def get_attribute_predictions(model, dataloader, device, mappings):
    """
    Runs inference with the siloed attribute model to generate a DataFrame of predicted attributes.
    """
    model.eval()
    all_preds = {attr: [] for attr in mappings.keys()}
    inverse_mappings = {attr: {i: label for label, i in mapping.items()} for attr, mapping in mappings.items()}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating attribute predictions"):
            pixel_values = batch['pixel_values'].to(device)
            outputs = model(pixel_values=pixel_values)
            
            for attr_name, logits in outputs['attribute_logits'].items():
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                decoded_preds = [inverse_mappings[attr_name].get(p, 'Unknown') for p in preds]
                all_preds[attr_name].extend(decoded_preds)
                
    return pd.DataFrame(all_preds)

def main():
    # --- CONFIGURATION ---
    SILOED_ATTR_CONFIG = 'configs/exp1_1_siloed_attributes.yaml'
    SILOED_ATTR_MODEL_PATH = 'results/exp1_1_siloed_attributes/model_best.pth'
    
    with open(SILOED_ATTR_CONFIG, 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    
    train_loader, val_loader, mappings, _ = create_dataloaders(config)
    
    model = SiloedModel(config, mappings).to(device)
    model.load_state_dict(torch.load(SILOED_ATTR_MODEL_PATH, map_location=device))

    # --- 2. Generate Predicted Attributes (The Features) ---
    print("Generating features for the tabular model using the trained vision model...")
    X_train_predicted = get_attribute_predictions(model, train_loader, device, mappings)
    X_val_predicted = get_attribute_predictions(model, val_loader, device, mappings)
    
    # --- 3. Load Ground-Truth Prices (The Target) ---
    train_df = pd.read_csv(os.path.join(config['data']['processed_dir'], config['data']['train_csv']))
    val_df = pd.read_csv(os.path.join(config['data']['processed_dir'], config['data']['val_csv']))
    
    y_train_log = np.log1p(train_df['price'])
    y_val_true = val_df['price'] # Original scale for final metric calculation

    # --- 4. Train the XGBoost model and SAVE it ---
    # The features are the columns from our generated predictions dataframe.
    ATTRIBUTE_COLS = list(X_train_predicted.columns)
    
    print("\nTraining Stage 2 (XGBoost model) on predicted attributes...")
    tabular_model = TabularPriceModel(categorical_features=ATTRIBUTE_COLS)
    
    # --- FIX: Train using the correctly separated features (X) and target (y) ---
    tabular_model.train(X_train_predicted, y_train_log)
    
    output_dir = 'models/hybrid_baseline'
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, 'tabular_price_model.joblib')
    joblib.dump(tabular_model, model_save_path)
    print(f"✅ Trained Hybrid Stage 2 model saved to: {model_save_path}")
    
    # --- 5. Evaluate on the VALIDATION set ---
    print("\nEvaluating the full hybrid pipeline on the VALIDATION set...")
    
    # --- FIX: Use the predicted attributes for the validation set as input ---
    y_val_pred_log = tabular_model.predict(X_val_predicted)
    y_val_pred = np.expm1(y_val_pred_log)
    
    mae = mean_absolute_error(y_val_true, y_val_pred)
    rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred))
    r2 = r2_score(y_val_true, y_val_pred)
    
    print("\n--- Hybrid Model (Vision -> XGBoost) VALIDATION Performance ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R²): {r2:.4f}")

if __name__ == '__main__':
    main()