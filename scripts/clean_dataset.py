import pandas as pd
import re
import os
import argparse
from tqdm import tqdm

# Set pandas to display progress bar with apply
tqdm.pandas()

def clean_html(raw_html: str) -> str:
    """
    Removes all HTML tags from a string.
    
    Args:
        raw_html: The string containing HTML tags.
        
    Returns:
        The cleaned string without HTML tags.
    """
    text = str(raw_html)
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', text)
    cleantext = cleantext.replace('&nbsp;', ' ').strip()
    return cleantext


def remove_brand_from_name(name: str, brand: str) -> str:
    """
    Removes the brand from the product name by finding and replacing the
    exact string from the 'brand' column.
    
    Args:
        name: The full product name (from the 'name' column).
        brand: The brand name (from the 'brand' column).
        
    Returns:
        The product name with the brand removed and whitespace cleaned up.
    """
    # Ensure inputs are strings to prevent errors on NaN or other types
    name = str(name)
    brand = str(brand)

    # If the brand is empty or just 'nan', there's nothing to remove.
    if not brand or brand.lower() == 'nan':
        return name

    # Use regex to remove the brand only if it appears at the start of the name.
    # `re.escape` handles brand names that might have special regex characters.
    # `re.IGNORECASE` makes the match case-insensitive (e.g., "Nike" matches "nike").
    cleaned_name = re.sub(f"^{re.escape(brand)}", '', name, flags=re.IGNORECASE)
    
    # After removing, clean up any leading whitespace
    return cleaned_name.strip()

def main():
    """
    Main function to run the data cleaning process.
    """
    parser = argparse.ArgumentParser(description="Clean product data CSV files.")
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default='data/processed', 
        help='Directory containing the train.csv, val.csv, and test.csv files.'
    )
    args = parser.parse_args()

    files_to_clean = ['train.csv', 'val.csv', 'test.csv']

    print(f"Starting cleanup process in directory: {args.data_dir}")

    for filename in files_to_clean:
        file_path = os.path.join(args.data_dir, filename)

        if not os.path.exists(file_path):
            print(f"Warning: File not found at {file_path}. Skipping.")
            continue

        print(f"\n--- Processing {filename} ---")
        
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows.")

        # --- Clean 'description' column ---
        print("Cleaning 'description' column (removing HTML)...")
        if 'description' in df.columns:
            df['description'] = df['description'].progress_apply(clean_html)
        else:
            print("Warning: 'description' column not found. Skipping.")
        
        # --- Clean 'name' column ---
        print("Cleaning 'name' column (removing brand)...")
        #Check for required columns and use df.apply with axis=1
        if 'name' in df.columns and 'brand' in df.columns:
            # We use axis=1 to apply the function row-wise, so it can access
            # both the 'name' and 'brand' columns for each row.
            df['name'] = df.progress_apply(
                lambda row: remove_brand_from_name(row['name'], row['brand']),
                axis=1
            )
        else:
            print("Warning: 'name' and/or 'brand' column not found. Skipping 'name' cleaning.")

        # Overwrite the original file with the cleaned data
        print(f"Saving cleaned data back to {file_path}...")
        df.to_csv(file_path, index=False)
        
        print(f"Successfully cleaned and saved {filename}.")
        
        # Show a few examples of the cleaned data
        if 'name' in df.columns:
            print("\nExample of cleaned 'name':")
            print(df['name'].head(3).to_string(index=False))
        if 'description' in df.columns:
            print("\nExample of cleaned 'description':")
            print(df['description'].head(3).to_string(index=False))
        print("-" * (len(filename) + 20))

    print("\nData cleanup complete for all files.")


if __name__ == '__main__':
    main()