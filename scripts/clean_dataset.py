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
    # Ensure input is a string to prevent errors on NaN values
    text = str(raw_html)
    
    # Regex to find and remove all HTML tags
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', text)
    
    # Replace common HTML entities and clean up whitespace
    cleantext = cleantext.replace('&nbsp;', ' ').strip()
    return cleantext

def remove_brand_from_name(name: str) -> str:
    """
    Removes the brand from the product name, assuming the format
    "Brand Name ... Women ...". The goal is to start the name with "Women".
    
    Args:
        name: The full product name.
        
    Returns:
        The product name with the brand removed.
    """
    # Ensure input is a string
    name = str(name)
    
    # Find the position of "Women"
    if "Women" in name:
        # Split the string at the first occurrence of "Women"
        parts = name.split("Women", 1)
        # Reconstruct the string starting with "Women" and the rest of the name
        return "Women" + parts[1]
    
    # If "Women" is not in the name, return the original name
    return name

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
        
        # Load the dataset
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows.")

        # --- Clean 'description' column ---
        print("Cleaning 'description' column (removing HTML)...")
        # The `progress_apply` will show a nice progress bar
        df['description'] = df['description'].progress_apply(clean_html)
        
        # --- Clean 'name' column ---
        print("Cleaning 'name' column (removing brand)...")
        df['name'] = df['name'].progress_apply(remove_brand_from_name)

        # Overwrite the original file with the cleaned data
        print(f"Saving cleaned data back to {file_path}...")
        df.to_csv(file_path, index=False)
        
        print(f"Successfully cleaned and saved {filename}.")
        # Show a few examples of the cleaned data
        print("\nExample of cleaned 'name':")
        print(df['name'].head(3).to_string(index=False))
        print("\nExample of cleaned 'description':")
        print(df['description'].head(3).to_string(index=False))
        print("-" * (len(filename) + 20))

    print("\nData cleanup complete for all files.")


if __name__ == '__main__':
    main()