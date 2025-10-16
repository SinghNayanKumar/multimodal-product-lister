import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
from transformers import GitProcessor

class ECommerceCaptioningDataset(Dataset):
    def __init__(self, df, image_dir, processor):
        self.df = df
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = os.path.join(self.image_dir, os.path.basename(row['image_path']))
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        name = row['name']
        description = row['description']
        target_text = f"title: {name} | description: {description}"

        processed_data = self.processor(
            images=image,
            text=target_text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": processed_data['pixel_values'].squeeze(),
            "input_ids": processed_data['input_ids'].squeeze(),
            "attention_mask": processed_data['attention_mask'].squeeze(),
            "labels": processed_data['input_ids'].squeeze().clone()
        }

def create_baseline_dataloaders(config):
    train_df = pd.read_csv(os.path.join(config['data']['processed_dir'], config['data']['train_csv']))
    val_df = pd.read_csv(os.path.join(config['data']['processed_dir'], config['data']['val_csv']))

    processor = GitProcessor.from_pretrained(config['model']['model_name'], use_fast=True)

    train_dataset = ECommerceCaptioningDataset(train_df, config['data']['image_dir'], processor)
    val_dataset = ECommerceCaptioningDataset(val_df, config['data']['image_dir'], processor)

    # --- FIX: Set num_workers to 0 to prevent multiprocessing deadlocks ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0 # <-- CHANGED
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0 # <-- CHANGED
    )

    return train_loader, val_loader, processor