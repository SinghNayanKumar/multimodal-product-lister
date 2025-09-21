import torch
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.data.dataloader import create_dataloaders

# --- FIX: Added missing import for dataloader creation. ---
from src.data.dataloader import create_dataloaders
from src.models.baselines.siloed_model import SiloedModel
# ANNOTATION: Import the dedicated tabular model class for this baseline.
from src.models.tabular_price_model import TabularPriceModel

def get_attribute_predictions(model, dataloader, device, mappings):
    """
    Runs inference with the siloed attribute model to generate predictions for a dataset.
    These predictions will become the features for the downstream XGBoost model.
    """
    model.eval()
    all_preds = {f"pred_{attr}": [] for attr in mappings.keys()}
    
    # --- FIX [Logical Bug]: Added inverse mappings. The original code would have failed ---
    # to decode the model's integer predictions (e.g., 5) back into meaningful labels (e.g., 'V-Neck').
    # This is essential for creating the tabular dataset for XGBoost.
    inverse_mappings = {
        attr: {i: label for label, i in mapping.items()} 
        for attr, mapping in mappings.items()
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating attribute predictions"):
            pixel_values = batch['pixel_values'].to(device)
            outputs = model(pixel_values=pixel_values)
            
            for attr_name, logits in outputs['attribute_logits'].items():
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                decoded_preds = [inverse_mappings[attr_name].get(p, 'Unknown') for p in preds]
                all_preds[f"pred_{attr_name}"].extend(decoded_preds)
                
    return pd.DataFrame(all_preds)

def main():
    # --- CONFIGURATION ---
    SILOED_ATTR_CONFIG = 'configs/exp1_1_siloed_attributes.yaml'
    SILOED_ATTR_MODEL_PATH = 'results/exp1_1_siloed_attributes/model_best.pth'
    
    # --- 1. Load trained siloed attribute model ---
    with open(SILOED_ATTR_CONFIG, 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    
    # Unpack the tokenizer into a placeholder `_`.
    train_loader, val_loader, mappings, _ = create_dataloaders(config)
    # --- NOTE: We pass `mappings` here so the siloed model knows how to build its attribute heads. ---
    model = SiloedModel(config, mappings).to(device)
    model.load_state_dict(torch.load(SILOED_ATTR_MODEL_PATH, map_location=device))

    # --- 2. Generate predictions for the training and validation sets ---
    # These predictions will be the features for our XGBoost model.
    train_preds_df = get_attribute_predictions(model, train_loader, device, mappings)
    val_preds_df = get_attribute_predictions(model, val_loader, device, mappings)
    
    # ANNOTATION: Combine predictions with the original ground truth data.
    # --- FIX [Best Practice]: Using os.path.join for robust path handling across different OS. ---
    train_df = pd.read_csv(os.path.join(config['data']['processed_dir'], config['data']['train_csv']))
    val_df = pd.read_csv(os.path.join(config['data']['processed_dir'], config['data']['val_csv']))
    
    train_hybrid_df = pd.concat([train_df.reset_index(drop=True), train_preds_df], axis=1)
    val_hybrid_df = pd.concat([val_df.reset_index(drop=True), val_preds_df], axis=1)

    # --- 3. Train the XGBoost model on the *predicted* attributes ---
    # ANNOTATION: Here's the core of the hybrid model. We use our dedicated TabularPriceModel
    # class for consistency and clarity.
    PREDICTED_ATTRIBUTE_COLS = list(train_preds_df.columns)
    PRICE_COL = 'price'

    print("\nTraining XGBoost model on predicted attributes...")
    tabular_model = TabularPriceModel(categorical_features=PREDICTED_ATTRIBUTE_COLS)
    # The model expects the log-transformed price, just like our main MTL model.
    y_train_log = np.log1p(train_hybrid_df[PRICE_COL])
    tabular_model.train(train_hybrid_df[PREDICTED_ATTRIBUTE_COLS], y_train_log)
    
    # --- 4. Evaluate the hybrid pipeline on the validation set ---
    print("\nEvaluating the full hybrid pipeline...")
    X_val = val_hybrid_df[PREDICTED_ATTRIBUTE_COLS]
    y_val_true = val_hybrid_df[PRICE_COL]
    
    # Predict log price and transform back to original scale
    y_val_pred_log = tabular_model.predict(X_val)
    y_val_pred = np.expm1(y_val_pred_log)
    
    mae = mean_absolute_error(y_val_true, y_val_pred)
    rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred))
    r2 = r2_score(y_val_true, y_val_pred)
    
    print("\n--- Hybrid Model (Vision -> XGBoost) Performance ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R²): {r2:.4f}")

if __name__ == '__main__':
    main()