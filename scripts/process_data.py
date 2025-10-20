
import pandas as pd
import ast
import json
import os
from sklearn.model_selection import train_test_split

# --- Configuration ---
RAW_DATA_PATH = 'data/raw/Fashion Dataset.csv' # IMPORTANT: Change this to your raw CSV file name
PROCESSED_DATA_DIR = 'data/processed'

# This list is selected from EDA, check notebooks/EDA.ipynb for more details.
SELECTED_ATTRIBUTES = [
    # --- Tier 1: Core Visuals ---
    'Neck',
    'Sleeve Length',
    'Print or Pattern Type',
    'Type',
    'Hemline',
    'Pattern', 
    
    # --- Tier 2: Detailed Visuals ---
    'Length',
    'Sleeve Styling',
    'Ornamentation',
    
    # --- Tier 3: Contextual for Text ---
    'Occasion',
    'Fabric',
    'Fit'
]

def parse_attributes(attr_string):
    
    """This function takes in attributes in string format 
        and converts it to dict."""

    # Check if the data is not a string (it might be a float NaN)
    if not isinstance(attr_string, str): 
        return {}
    try:
    # ast.literal_eval is the safe way to evaluate a string containing a Python literal
        return ast.literal_eval(attr_string)
    except (ValueError, SyntaxError):
        # If it fails, return an empty dict to avoid crashing
        return {}

def main():
    print("Starting data processing...")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Loaded {len(df)} rows from raw data.")

    # --- Mapping the Index to Images---
    df['image_path'] = df['p_id'].apply(lambda x: f'data/raw/Images/{x}.jpg')

    # --- Clean and Parse ---
    df['attributes_dict'] = df['p_attributes'].apply(parse_attributes)
    df.dropna(subset=['price', 'name', 'p_attributes','p_id'], inplace=True)
    df = df[df['image_path'].apply(os.path.exists)]

    # --- Flatten Attributes ---
    for attr in SELECTED_ATTRIBUTES:
        df[attr] = df['attributes_dict'].apply(lambda x: x.get(attr))
        df[attr] = df[attr].replace('NA', None)
        df[attr] = df[attr].fillna('Unknown')
        df[attr] = df[attr].str.strip()

    # --- Final Cleanup ---
    final_cols = ['name', 'price', 'description','brand', 'colour', 'avg_rating', 'ratingCount','image_path'] + SELECTED_ATTRIBUTES
    df_final = df[final_cols].copy()
    
    print(f"Processed dataframe has {len(df_final)} rows.")

    # --- Create Attribute Mappings ---
    mappings = {}
    for attr in SELECTED_ATTRIBUTES + ['colour']:  # Include base colour
        labels = list(df_final[attr].unique())
        if 'Unknown' not in labels:
            labels.append('Unknown')
        mappings[attr] = {label: i for i, label in enumerate(labels)}
    
    with open(os.path.join(PROCESSED_DATA_DIR, 'attribute_mappings.json'), 'w') as f:
        json.dump(mappings, f, indent=4)
    print("Saved attribute mappings.")

    # --- Train/Val/Test Split ---
    train_val_df, test_df = train_test_split(df_final, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.11, random_state=42)

    # --- Save Processed Data ---
    train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'test.csv'), index=False)
    print("Saved train, validation, and test sets.")
    print("Data processing complete.")

if __name__ == '__main__':
    main()