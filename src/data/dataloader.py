# Import necessary libraries
import torch
from torch.utils.data import Dataset, DataLoader  # Core PyTorch utilities for handling datasets
from PIL import Image  # Python Imaging Library for opening, manipulating, and saving many different image file formats
import pandas as pd  # Library for data manipulation and analysis, used here to read CSV files
import os  # Provides a way of using operating system dependent functionality, like path joining
import json  # Used for working with JSON data, in this case, loading attribute mappings

class ECommerceDataset(Dataset):
    """
    A custom PyTorch Dataset to load and preprocess image, text, price, and attribute data.
    Each item in the dataset will be a dictionary containing all the necessary inputs and targets
    for our multi-task model.
    """
    def __init__(self, df, image_dir, image_processor, tokenizer, attribute_mappers):
        """
        Initializes the dataset object.
        Args:
            df (pd.DataFrame): The dataframe containing the metadata (image paths, prices, attributes, text).
            image_dir (str): The directory where the product images are stored.
            image_processor: A Hugging Face processor (e.g., ViTImageProcessor) to transform images into tensors.
            tokenizer: A Hugging Face tokenizer (e.g., T5Tokenizer) to convert text into token IDs.
            attribute_mappers (dict): A nested dictionary mapping attribute names and their string values to integer IDs.
        """
        self.df = df
        self.image_dir = image_dir
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.attribute_mappers = attribute_mappers
        # Get a list of attribute names we need to process from the mappers dictionary.
        self.attributes_to_map = list(attribute_mappers.keys())

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieves and preprocesses a single data sample from the dataset at the given index.
        This method is called by the DataLoader to create a batch.
        Args:
            idx (int): The index of the data sample to retrieve.
        Returns:
            dict: A dictionary containing the processed data for a single product.
        """
        # Get the row of metadata for the given index
        row = self.df.iloc[idx]

        # --- 1. Load and Process Image ---
        # Construct the full path to the image file.
        image_path = os.path.join(self.image_dir, os.path.basename(row['image_path']))
        try:
            # Open the image file and convert it to RGB to ensure a consistent 3-channel format.
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            # Handle cases where an image is missing by creating a black placeholder image.
            # This prevents training from crashing due to missing data.
            print(f"Warning: Image not found at {image_path}. Using a placeholder black image.")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Use the pre-trained image processor to resize, normalize, and convert the image to a PyTorch tensor.
        # .squeeze() removes the unnecessary batch dimension (e.g., from [1, 3, 224, 224] to [3, 224, 224]).
        pixel_values = self.image_processor(image, return_tensors="pt")['pixel_values'].squeeze()

        # --- 2. Process Structured Attributes ---
        # This prepares the ground-truth labels for the attribute classification heads of our model.
        attribute_targets = {}
        for attr in self.attributes_to_map:
            # Get the attribute value from the dataframe row (e.g., 'V-Neck'). Default to 'Unknown' if missing.
            value = row.get(attr, 'Unknown')
            # Use the mapper to convert the string value into its corresponding integer ID (e.g., 'V-Neck' -> 2).
            target_id = self.attribute_mappers[attr].get(value, self.attribute_mappers[attr]['Unknown'])
            # Store the integer ID as a long tensor, which is required for CrossEntropyLoss.
            attribute_targets[f"{attr}_target"] = torch.tensor(target_id, dtype=torch.long)

        # --- 3. Process Price ---
        # This prepares the ground-truth label for the price regression head.
        price_target = torch.tensor(row['price'], dtype=torch.float32)

        # --- 4. Process Text (for Seq2Seq Generation) ---
        # This section sets up the input (prompt) and output (labels) for the text generation task.
        # The model will learn to generate the 'target_text' when given the image and the 'prompt'.
        
        # Fetch the best high-level, contextual attributes. Provide sensible defaults if they are missing.
        occasion = row.get('Occasion', 'everyday wear') # High frequency, great for tone
        item_type = row.get('Type', 'garment')         # High-level identity

        # Construct a prompt that gives context without revealing specific visual details.
        prompt = f"generate a product listing for an item of type '{item_type}', suitable for '{occasion}'"

        # Define the target text that the model should learn to generate.
        target_text = f"title: {row['name']} | description: {row['description']}"
        
        # Tokenize the input prompt. `max_length` ensures all sequences have the same length by padding or truncating.
        input_encoding = self.tokenizer(prompt, padding='max_length', max_length=64, truncation=True, return_tensors="pt")
        # Tokenize the target text. This will become the 'labels' for the language model.
        target_encoding = self.tokenizer(target_text, padding='max_length', max_length=128, truncation=True, return_tensors="pt")

        # --- 5. Collate and Return ---
        # Return a single dictionary containing all processed data.
        # The DataLoader will automatically stack these dictionaries into a batch.
        return {
            'pixel_values': pixel_values,
            'price_target': price_target,
            **attribute_targets,  # Unpacks the attribute_targets dictionary (e.g., {'neck_target': ..., 'sleeve_target': ...})
            'input_ids': input_encoding.input_ids.flatten(),
            'attention_mask': input_encoding.attention_mask.flatten(),
            'labels': target_encoding.input_ids.flatten()  # The language model expects the target token IDs under the key 'labels'
        }

def create_dataloaders(config):
    """
    A factory function to encapsulate the setup of training and validation dataloaders.
    It reads configuration, loads data, initializes processors, and creates DataLoader objects.
    Args:
        config (dict): A configuration dictionary containing paths, model names, and training parameters.
    Returns:
        tuple: A tuple containing the training DataLoader, validation DataLoader, and the attribute mappings dictionary.
    """
    # Load the training and validation metadata from CSV files specified in the config.
    train_df = pd.read_csv(os.path.join(config['data']['processed_dir'], config['data']['train_csv']))
    val_df = pd.read_csv(os.path.join(config['data']['processed_dir'], config['data']['val_csv']))

    # Load the attribute-to-integer mappings from a JSON file. This ensures consistency across runs.
    with open(os.path.join(config['data']['processed_dir'], config['data']['mappings'])) as f:
        mappings = json.load(f)

    # Initialize the tokenizer and image processor from pre-trained models on the Hugging Face Hub.
    # Using pre-trained components allows us to leverage knowledge from large-scale models (transfer learning).
    from transformers import T5Tokenizer, ViTImageProcessor
    tokenizer = T5Tokenizer.from_pretrained(config['model']['text_model_name'])
    image_processor = ViTImageProcessor.from_pretrained(config['model']['vision_model_name'])
    
    # Create instances of our custom ECommerceDataset for the training and validation sets.
    train_dataset = ECommerceDataset(train_df, config['data']['image_dir'], image_processor, tokenizer, mappings)
    val_dataset = ECommerceDataset(val_df, config['data']['image_dir'], image_processor, tokenizer, mappings)

    # Create the DataLoader objects, which will handle batching, shuffling, and multi-process data loading.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True,  # Shuffle the training data to prevent the model from learning the order of samples.
        num_workers=4  # Use multiple subprocesses to load data in parallel, speeding up training.
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, # No need to shuffle validation data.
        num_workers=4
    )
    
    # Return the dataloaders and mappings. The mappings will be needed by the main model to define its classification heads.
    return train_loader, val_loader, mappings