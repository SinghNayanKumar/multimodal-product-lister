# src/data/dataloader_vlm.py

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import json
import numpy as np
from transformers import ViTImageProcessor, AutoTokenizer

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
            # This path should ideally not be taken if data is pre-cleaned
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        pixel_values = self.image_processor(image, return_tensors="pt")['pixel_values'].squeeze()

        target_text = f"title: {row['name']} | description: {row['description']}"
        
        labels = self.tokenizer(target_text, padding='max_length', max_length=128, truncation=True, return_tensors="pt").input_ids.flatten()
        labels[labels == self.tokenizer.pad_token_id] = -100
            
        return {
            'pixel_values': pixel_values,
            'labels': labels
        }

def create_datasets(config):
    train_df = pd.read_csv(os.path.join(config['data']['processed_dir'], config['data']['train_csv']))
    val_df = pd.read_csv(os.path.join(config['data']['processed_dir'], config['data']['val_csv']))

    # --- Data Cleaning: Prevent "nan" text ---
    train_df[['name', 'description']] = train_df[['name', 'description']].fillna('')
    val_df[['name', 'description']] = val_df[['name', 'description']].fillna('')

    with open(os.path.join(config['data']['processed_dir'], config['data']['mappings'])) as f:
        mappings = json.load(f)

    image_processor = ViTImageProcessor.from_pretrained(config['model']['vision_model_name'])
    tokenizer = AutoTokenizer.from_pretrained(config['model']['text_model_name'])
    
    # --- ARTICLE-COMPLIANT TOKENIZATION ---
    # 1. Add a new, distinct [PAD] token. This increases the tokenizer's vocabulary size.
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Set BOS and EOS tokens for consistency, although gpt2 already has them.
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({'bos_token': '[BOS]'})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '[EOS]'})

    train_dataset = ECommerceDataset(train_df, config['data']['image_dir'], image_processor, tokenizer, mappings)
    val_dataset = ECommerceDataset(val_df, config['data']['image_dir'], image_processor, tokenizer, mappings)

    return train_dataset, val_dataset, tokenizer