# Import necessary libraries
import torch
from torch.utils.data import Dataset, DataLoader  # Core PyTorch utilities for handling datasets
from PIL import Image  # Python Imaging Library for opening, manipulating, and saving many different image file formats
import pandas as pd  # Library for data manipulation and analysis, used here to read CSV files
import os  # Provides a way of using operating system dependent functionality, like path joining
import json  # Used for working with JSON data, in this case, loading attribute mappings
import numpy as np


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
        image_path = os.path.join(self.image_dir, os.path.basename(row['image_path']))
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Image not found at {image_path}. Using a placeholder black image.")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        pixel_values = self.image_processor(image, return_tensors="pt")['pixel_values'].squeeze()

        # --- 2. Process Structured Attributes ---
        attribute_targets = {}
        for attr in self.attributes_to_map:
            value = row.get(attr, 'Unknown')
            target_id = self.attribute_mappers[attr].get(value, self.attribute_mappers[attr]['Unknown'])
            attribute_targets[f"{attr}_target"] = torch.tensor(target_id, dtype=torch.long)

        # --- 3. Process Price ---
        price_target = torch.tensor(np.log1p(row['price']), dtype=torch.float32)

        # --- 4. Process Text (only if tokenizer is available) ---
        result = {
            'pixel_values': pixel_values,
            'image_tensor': pixel_values,  # For compatibility with other models
            'price_target': price_target,
            **attribute_targets,
        }
        
        # Only add text-related fields if tokenizer is available
        if self.tokenizer is not None:
            occasion = row.get('Occasion', 'everyday wear')
            item_type = row.get('Type', 'garment')
            prompt = f"generate a product listing for an item of type '{item_type}', suitable for '{occasion}'"
            target_text = f"title: {row['name']} | description: {row['description']}"
            
            input_encoding = self.tokenizer(prompt, padding='max_length', max_length=64, truncation=True, return_tensors="pt")
            target_encoding = self.tokenizer(target_text, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
            
            result.update({
                'input_ids': input_encoding.input_ids.flatten(),
                'attention_mask': input_encoding.attention_mask.flatten(),
                'labels': target_encoding.input_ids.flatten()
            })
        else:
            # Provide dummy values for text fields when no tokenizer (siloed models)
            result.update({
                'input_ids': torch.zeros(64, dtype=torch.long),
                'attention_mask': torch.zeros(64, dtype=torch.long),  
                'labels': torch.zeros(128, dtype=torch.long)
            })
            
        return result

def create_test_dataloaders(config):
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

    # Initialize the image processor (always needed)
    from transformers import ViTImageProcessor
    image_processor = ViTImageProcessor.from_pretrained(config['model']['vision_model_name'])
    
    # Initialize tokenizer only if text model is specified
    tokenizer = None
    if 'text_model_name' in config['model']:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config['model']['text_model_name'])
    
    # Add this fix for GPT-2 and other models without pad tokens:
    if tokenizer is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    
    # The mappings will be needed by the main model to define its classification heads.
    # Return the dataloaders, mappings, AND the tokenizer (which might be None).
    return train_loader, val_loader, mappings, tokenizer