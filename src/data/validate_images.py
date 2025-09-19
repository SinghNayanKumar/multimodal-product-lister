import os
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import argparse

def validate_images(image_dir):
    """
    Scans a directory for image files and checks if they can be opened.
    Lists all corrupted or unreadable image files.
    """
    bad_files = []
    # Use os.walk to recursively go through all subdirectories
    for root, _, files in os.walk(image_dir):
        print(f"Scanning directory: {root}")
        for filename in tqdm(files, desc="Validating images"):
            # Check for common image extensions
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(root, filename)
                try:
                    # The main check: try to open and load the image
                    with Image.open(file_path) as img:
                        img.load() # This forces PIL to read the pixel data
                except (IOError, OSError, UnidentifiedImageError) as e:
                    # Catch a broad range of file-related and PIL-specific errors
                    print(f"\nFound corrupted file: {file_path}")
                    print(f"   - Error: {e}")
                    bad_files.append(file_path)
    
    return bad_files

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validate image files in a directory.")
    parser.add_argument('--image_dir', type=str, required=True, help="Path to the root directory of images.")
    args = parser.parse_args()

    corrupted_files = validate_images(args.image_dir)

    if not corrupted_files:
        print("\n✅ All images in the directory were validated successfully!")
    else:
        print(f"\nFound {len(corrupted_files)} corrupted image(s).")
        print("You should remove or replace these files from your dataset.")
        # Optional: Save the list of bad files to a text file
        with open("corrupted_files.txt", "w") as f:
            for file_path in corrupted_files:
                f.write(f"{file_path}\n")
        print("A list of corrupted files has been saved to 'corrupted_files.txt'.")