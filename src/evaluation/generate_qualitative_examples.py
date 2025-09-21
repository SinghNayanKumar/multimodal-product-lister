import torch
import yaml
import json
import pandas as pd
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

from src.data.dataloader import create_dataloaders
from src.models.multitask_model import MultitaskModel
from src.models.baselines.direct_vlm import DirectVLM

class QualitativeExampleGenerator:
    def __init__(self, base_config_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        _, self.val_loader, self.mappings, self.tokenizer = create_dataloaders(self.base_config)
    
    def generate_side_by_side_examples(self, mtl_model_path, vlm_config_path, vlm_model_path, 
                                     output_dir, num_examples=10):
        """Generate side-by-side comparison examples for the paper."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Load models
        mtl_model = MultitaskModel(self.base_config, self.mappings)
        mtl_model.load_state_dict(torch.load(mtl_model_path, map_location=self.device))
        mtl_model.eval().to(self.device)
        
        with open(vlm_config_path, 'r') as f:
            vlm_config = yaml.safe_load(f)
        vlm_model = DirectVLM(vlm_config)
        vlm_model.load_state_dict(torch.load(vlm_model_path, map_location=self.device))
        vlm_model.eval().to(self.device)
        
        from transformers import AutoTokenizer
        vlm_tokenizer = AutoTokenizer.from_pretrained(vlm_config['model']['text_model_name'])
        if vlm_tokenizer.pad_token is None:
            vlm_tokenizer.pad_token = vlm_tokenizer.eos_token
        
        examples = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Generating examples")):
                if len(examples) >= num_examples:
                    break
                
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
                
                pixel_values = batch['pixel_values'][:1]  # Take first image in batch
                
                # Get MTL predictions
                mtl_outputs = mtl_model.predict(pixel_values, self.tokenizer, self.mappings, use_hierarchical_prompt=True)
                if not isinstance(mtl_outputs, list):
                    mtl_outputs = [mtl_outputs]
                
                # Get VLM predictions
                vlm_outputs = vlm_model.predict(pixel_values, vlm_tokenizer)
                
                # Get ground truth
                true_text = self.tokenizer.batch_decode(batch['labels'][:1], skip_special_tokens=True)[0]
                
                # Save example
                example = {
                    'index': len(examples),
                    'image_tensor': pixel_values[0].cpu(),
                    'ground_truth': true_text,
                    'mtl_hierarchical': mtl_outputs[0]['generated_text'],
                    'mtl_attributes': mtl_outputs[0]['predicted_attributes'],
                    'mtl_price': mtl_outputs[0]['predicted_price'],
                    'direct_vlm': vlm_outputs[0] if isinstance(vlm_outputs, list) else vlm_outputs
                }
                
                examples.append(example)
        
        # Save examples as JSON (without image tensors)
        json_examples = []
        for ex in examples:
            json_ex = ex.copy()
            del json_ex['image_tensor']  # Remove tensor for JSON serialization
            json_examples.append(json_ex)
        
        with open(os.path.join(output_dir, 'qualitative_examples.json'), 'w') as f:
            json.dump(json_examples, f, indent=4)
        
        # Create visual comparison figures
        self.create_comparison_figures(examples, output_dir)
        
        return examples
    
    def create_comparison_figures(self, examples, output_dir):
        """Create matplotlib figures comparing model outputs."""
        for i, example in enumerate(examples[:5]):  # Create figures for first 5 examples
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            # Denormalize image
            img_tensor = example['image_tensor']
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = img_tensor * std + mean
            img = img_tensor.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            
            # Display image
            axes[0].imshow(img)
            axes[0].set_title("Input Image")
            axes[0].axis('off')
            
            # Display text comparisons
            texts = [
                ("Ground Truth", example['ground_truth']),
                ("MTL (Hierarchical)", example['mtl_hierarchical']),
                ("Direct VLM", example['direct_vlm'])
            ]
            
            for j, (title, text) in enumerate(texts):
                axes[j+1].text(0.05, 0.95, f"{title}:\n\n{text}", 
                             transform=axes[j+1].transAxes, 
                             verticalalignment='top',
                             wrap=True, fontsize=8)
                axes[j+1].set_xlim(0, 1)
                axes[j+1].set_ylim(0, 1)
                axes[j+1].axis('off')
            
            # Add attributes info
            attr_text = "Predicted Attributes:\n" + "\n".join([f"{k}: {v}" for k, v in example['mtl_attributes'].items()])
            axes[2].text(0.05, 0.4, attr_text, transform=axes[2].transAxes, fontsize=7, color='blue')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'comparison_example_{i+1}.png'), dpi=300, bbox_inches='tight')
            plt.close()
