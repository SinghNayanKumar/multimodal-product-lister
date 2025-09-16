
import pandas as pd
import ast
import json
import os
from sklearn.model_selection import train_test_split

# --- Configuration ---
RAW_DATA_PATH = 'data/raw/your_dataset.csv' # IMPORTANT: Change this to your raw CSV file name
PROCESSED_DATA_DIR = 'data/processed'

# TODO: Fill this list with the attributes you selected after EDA.
SELECTED_ATTRIBUTES = [
    'Neck', 'Sleeve Length', 'Print or Pattern Type', 'Type',
    'Hemline', 'Pattern', 'Length', 'Sleeve Styling',
    'Ornamentation', 'Occasion', 'Fabric', 'Fit'
]

def parse_attributes(attr_string):
    if not isinstance(attr_string, str): return {}
    try:
        return ast.literal_eval(attr_string)
    except (ValueError, SyntaxError):
        return {}

def main():
    print("Starting data processing...")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Loaded {len(df)} rows from raw data.")

    # --- Clean and Parse ---
    df.dropna(subset=['price', 'Image URL', 'name', 'p_attributes'], inplace=True)
    df['attributes_dict'] = df['p_attributes'].apply(parse_attributes)

    # --- Flatten Attributes ---
    for attr in SELECTED_ATTRIBUTES:
        df[attr] = df['attributes_dict'].apply(lambda x: x.get(attr))
        df[attr] = df[attr].replace('NA', None)
        df[attr] = df[attr].fillna('Unknown')
        df[attr] = df[attr].str.strip()

    # --- Final Cleanup ---
    # TODO: Make sure the column names match your dataset
    final_cols = ['ProductID', 'name', 'price', 'Image URL', 'Product Description', 'colour'] + SELECTED_ATTRIBUTES
    df_final = df[final_cols].copy()
    
    # We need a local image path column for the dataloader
    # For now, this is a placeholder. You should run your download script first.
    df_final['image_path'] = df_final['ProductID'].apply(lambda x: f"downloaded_images/{x}.jpg")
    
    print(f"Processed dataframe has {len(df_final)} rows.")

    # --- Create Attribute Mappings ---
    mappings = {}
    for attr in SELECTED_ATTRIBUTES + ['colour']: # Include base colour
        labels = df_final[attr].unique()
        mappings[attr] = {label: i for i, label in enumerate(labels)}
    
    with open(os.path.join(PROCESSED_DATA_DIR, 'attribute_mappings.json'), 'w') as f:
        json.dump(mappings, f, indent=4)
    print("Saved attribute mappings.")

    # --- Train/Val/Test Split ---
    train_val_df, test_df = train_test_split(df_final, test_size=0.1, random_state=42, stratify=df_final['Type'])
    train_df, val_df = train_test_split(train_val_df, test_size=0.11, random_state=42, stratify=train_val_df['Type'])

    # --- Save Processed Data ---
    train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'test.csv'), index=False)
    print("Saved train, validation, and test sets.")
    print("Data processing complete.")

if __name__ == '__main__':
    main()