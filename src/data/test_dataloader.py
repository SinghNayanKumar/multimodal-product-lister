import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import json
import numpy as np

class ECommerceDataset(Dataset):
    def __init__(self, df, image_dir, image_processor, tokenizer, attribute_mappers):
        self.df = df
        self.image_dir = image_dir
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.attribute_mappers = attribute_mappers
        self.attributes_to_map = list(attribute_mappers.keys())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = os.path.join(self.image_dir, os.path.basename(row['image_path']))
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        pixel_values = self.image_processor(image, return_tensors="pt")['pixel_values'].squeeze()

        attribute_targets = {}
        for attr in self.attributes_to_map:
            value = row.get(attr, 'Unknown')
            target_id = self.attribute_mappers[attr].get(value, self.attribute_mappers[attr]['Unknown'])
            attribute_targets[f"{attr}_target"] = torch.tensor(target_id, dtype=torch.long)

        price_target = torch.tensor(np.log1p(row['price']), dtype=torch.float32)

        result = {
            'pixel_values': pixel_values,
            'price_target': price_target,
            **attribute_targets,
        }
        
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
            
        return result

def create_test_dataloader(config):
    """
    Factory function to create a dataloader for the TEST set.
    """
    test_df = pd.read_csv(os.path.join(config['data']['processed_dir'], 'test.csv'))
    
    with open(os.path.join(config['data']['processed_dir'], config['data']['mappings'])) as f:
        mappings = json.load(f)

    from transformers import ViTImageProcessor, AutoTokenizer
    image_processor = ViTImageProcessor.from_pretrained(config['model']['vision_model_name'])
    
    tokenizer = None
    if 'text_model_name' in config['model']:
        tokenizer = AutoTokenizer.from_pretrained(config['model']['text_model_name'])
        
        # --- FIX: This check is now nested inside the block where we know ---
        # --- the tokenizer object has been created. This prevents the crash. ---
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    test_dataset = ECommerceDataset(test_df, config['data']['image_dir'], image_processor, tokenizer, mappings)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=False,  # Should not shuffle test data for reproducibility
        num_workers=4
    )
    
    # Return -1 as placeholder for the unused validation loader
    return test_loader, -1, mappings, tokenizer