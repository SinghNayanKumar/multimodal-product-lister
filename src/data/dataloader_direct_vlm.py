# FILE: src/data/dataloader_direct_vlm.py
# --- No changes to the code ---

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import json
from transformers import ViTImageProcessor, AutoTokenizer

class ECommerceDatasetDirectVLM(Dataset):
    """
    A custom PyTorch Dataset for the DirectVLM baseline.
    """
    def __init__(self, df, image_dir, image_processor, tokenizer, attribute_columns):
        self.df = df
        self.image_dir = image_dir
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.attribute_columns = attribute_columns

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

        target_data = {
            "attributes": {attr: row.get(attr, "Unknown") for attr in self.attribute_columns},
            "price": row['price'],
            "text": f"title: {row['name']} | description: {row['description']}"
        }
        target_json_string = json.dumps(target_data, separators=(',', ':'))

        target_encoding = self.tokenizer(
            target_json_string, padding='max_length', max_length=256, truncation=True, return_tensors="pt"
        )
        labels = target_encoding.input_ids.flatten()

        return {
            'pixel_values': pixel_values,
            'labels': labels,
            'ground_truth_json': target_json_string
        }

def create_dataloaders_for_direct_vlm(config):
    train_df = pd.read_csv(os.path.join(config['data']['processed_dir'], config['data']['train_csv']))
    val_df = pd.read_csv(os.path.join(config['data']['processed_dir'], config['data']['val_csv']))
    
    with open(os.path.join(config['data']['processed_dir'], config['data']['mappings'])) as f:
        mappings = json.load(f)
    attribute_columns = list(mappings.keys())

    image_processor = ViTImageProcessor.from_pretrained(config['model']['vision_model_name'])
    tokenizer = AutoTokenizer.from_pretrained(config['model']['text_model_name'])

    train_dataset = ECommerceDatasetDirectVLM(train_df, config['data']['image_dir'], image_processor, tokenizer, attribute_columns)
    val_dataset = ECommerceDatasetDirectVLM(val_df, config['data']['image_dir'], image_processor, tokenizer, attribute_columns)

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4)
    
    return train_loader, val_loader, tokenizer