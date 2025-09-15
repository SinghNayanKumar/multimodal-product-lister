import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split

# --- Configuration for the Kaggle Dataset ---
RAW_DATA_PATH = 'data/raw/styles.csv'
IMAGE_DIR = 'data/processed/images'
PROCESSED_DATA_PATH = 'data/processed'

def process_dataset():
    """
    Reads the 'styles.csv', cleans it, and creates train/val/test splits.
    """
    print("Step 1: Reading the dataset...")
    try:
        df = pd.read_csv(RAW_DATA_PATH, on_bad_lines='skip') # 'on_bad_lines' handles potential errors in the csv
    except FileNotFoundError:
        print(f"Error: Make sure '{RAW_DATA_PATH}' exists. You need to download and place it there.")
        return

    print(f"Loaded {len(df)} records from Kaggle.")

    # --- Data Cleaning and Feature Engineering ---
    # Create the 'local_image_path' column
    df['local_image_path'] = df.apply(lambda row: os.path.join(IMAGE_DIR, f"{row['id']}.jpg"), axis=1)

    # Filter out rows where the image file doesn't actually exist
    df = df[df['local_image_path'].apply(os.path.exists)]
    print(f"Found {len(df)} records with corresponding images.")

    # Create a 'title' from existing fields
    df['title'] = df['productDisplayName'].fillna('Fashion Item')

    # Create a simple 'description' from attributes
    def create_description(row):
        return f"A stylish {row['masterCategory']} item in {row['baseColour']} for {row['season']} wear. This {row['subCategory']} piece is perfect for {row['usage']} occasions."
    df['description'] = df.apply(create_description, axis=1)
    
    # Create the structured 'attributes_json'
    def create_attributes(row):
        attrs = {
            "gender": row["gender"],
            "category": row["masterCategory"],
            "sub_category": row["subCategory"],
            "color": row["baseColour"],
            "season": row["season"],
            "usage": row["usage"]
        }
        return json.dumps(attrs)
    df['attributes_json'] = df.apply(create_attributes, axis=1)

    # Select and rename columns to match our project's expected format
    final_df = df[['title', 'description', 'local_image_path', 'attributes_json']]

    print("Step 2: Splitting data into train, validation, and test sets...")
    train_df, temp_df = train_test_split(final_df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")

    # Save the datasets
    train_df.to_csv(os.path.join(PROCESSED_DATA_PATH, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DATA_PATH, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DATA_PATH, 'test.csv'), index=False)
    
    print(f"\nProcessing complete! Datasets saved in '{PROCESSED_DATA_PATH}'")

if __name__ == '__main__':
    process_dataset()