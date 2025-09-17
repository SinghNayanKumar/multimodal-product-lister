import pandas as pd
import os
import json
import ast
from sklearn.model_selection import train_test_split

# --- Configuration for the Kaggle Dataset ---
RAW_DATA_PATH = 'data/raw/Fashion Dataset.csv'
IMAGE_DIR = 'data/raw/Images'
PROCESSED_DATA_PATH = 'data/processed'

def process_dataset():
    """
    Reads the '*.csv', cleans it, and creates train/val/test splits.
    """
    print("Step 1: Reading the dataset...")
    try:
        df = pd.read_csv(RAW_DATA_PATH, on_bad_lines='skip') # 'on_bad_lines' handles potential errors in the csv
    except FileNotFoundError:
        print(f"Error: Make sure '{RAW_DATA_PATH}' exists. You need to download and place it there.")
        return

    print(f"Loaded {len(df)} records")

    # --- Data Cleaning and Feature Engineering ---
    # we have sufficiently large dataset
    # drop the rows which dont have the price or name
    df.dropna(subset=['price', 'img', 'name'], inplace=True)

    #dropping the image URL (we have local copy) and product unique ID (useless) 
    df.drop(["p_id","img"], axis=1, inplace=True)

    #mapping the id to images
    df['image_path'] = df['Index'].apply(lambda row: os.path.join(IMAGE_DIR, f"{row}.jpg"))

    #converting product attributes string to a dict in a new column
    df['attributes_dict'] = df['p_attributes'].apply(parse_attributes)

    # Selecting visually indentifiable frequent attributes from attributes
    selected_attributes = [
        'Neck',
        'Sleeve Length',
        'Print or Pattern Type',
        'Hemline',
        'Pattern',
        'Sleeve Styling'
    ]


    # --- Create a new column for each selected attribute ---
    for attr in selected_attributes:
        # Using .get(attr, 'Unknown') to handle cases where a product doesn't have that specific key
        df[attr] = df['attributes_dict'].apply(lambda x: x.get(attr, 'Unknown'))

    # --- Cleaning up 'NA' or other junk values ---
    # Often, the data has 'NA' as a string. Let's standardize it.
    for attr in selected_attributes:
        df[attr] = df[attr].replace('NA', 'Unknown')
        df[attr] = df[attr].fillna('Unknown')

    #print value counts of each attributes chosen
    for attr in selected_attributes:
    print(f"\nValue counts for '{attr}':")
    print(df[attr].value_counts().nlargest(10)) # Show top 10 values

    #dropping of attributes in string and dictornary form from df
    df_final = df.drop(columns=['p_attributes', 'attributes_dict'],inplace=True)


    # Filter out rows where the image file doesn't actually exist
    df = df[df['image_path'].apply(os.path.exists)]
    print(f"Found {len(df)} records with corresponding images.")


    
    # Create the structured 'attributes_json'
    def create_attributes(row):
        attrs = {
            "Colour": row["colour"],
            "Neck": row["Neck"],
            "Sleeve Length	": row["Sleeve Length"],
            "Print or Pattern Type": row["Print or Pattern Type"],
            "Hemline": row["Hemline"],
            "Pattern": row["Pattern"],
            "Sleeve Styling": row["Sleeve Styling"],
        }
        return json.dumps(attrs)
    df['attributes_json'] = df.apply(create_attributes, axis=1)
   

    # Select and rename columns to match our project's expected format
    final_df = df[['name', 'description', 'image_path', 'price']]

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

# --- Define a safe parsing function ---
# This function takes in the attributes in string format and gives attributes dictonary
def parse_attributes(attr_string):
    # Check if the data is not a string (it might be a float NaN)
    if not isinstance(attr_string, str):
        return {} # Return an empty dict if the data is missing
    try:
        # ast.literal_eval is the safe way to evaluate a string containing a Python literal
        return ast.literal_eval(attr_string)
    except (ValueError, SyntaxError):
        # If it fails, return an empty dict to avoid crashing
        return {}




if __name__ == '__main__':
    process_dataset()