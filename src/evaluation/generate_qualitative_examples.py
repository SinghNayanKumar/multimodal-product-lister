import torch
import yaml
import json
import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import textwrap

# --- FIX: Import all necessary models and data classes ---
from src.data.dataloader import create_dataloaders
from src.models.multitask_model import MultitaskModel
from src.models.baseline_git import GitBaselineModel
from src.models.baselines.direct_vlm_model import DirectVLM
from transformers import GitProcessor

class QualitativeExampleGenerator:
    def __init__(self, base_config_path, device='cuda'):
        print("Initializing QualitativeExampleGenerator...")
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        # This dataloader is for the MTL model and ground truth
        _, self.val_loader, self.mappings, self.tokenizer = create_dataloaders(self.base_config)
        print(f"Found {len(self.val_loader.dataset)} items in validation set.")
    
    def generate_side_by_side_examples(self, mtl_model_path, 
                                     git_vlm_config_path, git_vlm_model_path,
                                     t5_vlm_config_path, t5_vlm_model_path,
                                     output_dir, num_examples=10):
        """Generate side-by-side comparison examples for all models."""
        print("Starting to generate side-by-side examples...")
        os.makedirs(output_dir, exist_ok=True)
        
        # --- 1. Load All Models ---
        print("Loading MTL Model...")
        mtl_model = MultitaskModel(self.base_config, self.mappings)
        mtl_model.load_state_dict(torch.load(mtl_model_path, map_location=self.device))
        mtl_model.eval().to(self.device)
        
        print("Loading GIT-base VLM...")
        with open(git_vlm_config_path, 'r') as f:
            git_vlm_config = yaml.safe_load(f)
        git_vlm_model = GitBaselineModel(git_vlm_config)
        git_vlm_model.load_state_dict(torch.load(git_vlm_model_path, map_location=self.device))
        git_vlm_model.eval().to(self.device)
        git_processor = GitProcessor.from_pretrained(git_vlm_config['model']['model_name'], use_fast=True)

        print("Loading ViT-T5 VLM...")
        with open(t5_vlm_config_path, 'r') as f:
            t5_vlm_config = yaml.safe_load(f)
        t5_vlm_model = DirectVLM(
            vision_model_name=t5_vlm_config['model']['vision_model_name'],
            text_model_name=t5_vlm_config['model']['text_model_name']
        )
        t5_vlm_model.load_state_dict(torch.load(t5_vlm_model_path, map_location=self.device))
        t5_vlm_model.eval().to(self.device)
        
        examples = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Generating examples"):
                if len(examples) >= num_examples:
                    break
                
                pixel_values = batch['pixel_values'].to(self.device)
                
                # --- 2. Get Predictions from All Models ---
                # MTL predictions
                mtl_outputs = mtl_model.predict(pixel_values, self.tokenizer, self.mappings, use_hierarchical_prompt=True)[0]
                
                # GIT-base VLM predictions
                git_vlm_outputs = git_vlm_model.predict(pixel_values, git_processor)[0]
                
                # ViT-T5 VLM predictions (it uses the same tokenizer as MTL)
                # Note: The original DirectVLM was trained to generate JSON, we handle that here.
                t5_vlm_outputs = t5_vlm_model.generate(pixel_values, self.tokenizer)[0]

                # Get ground truth
                true_text = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)[0]
                
                # --- 3. Save Example Data ---
                example = {
                    'index': len(examples),
                    'image_tensor': pixel_values[0].cpu(),
                    'ground_truth': true_text,
                    'mtl_hierarchical': mtl_outputs['generated_text'],
                    'mtl_attributes': mtl_outputs['predicted_attributes'],
                    'mtl_price': float(mtl_outputs['predicted_price']),
                    'vlm_git_base': git_vlm_outputs,
                    'vlm_vit_t5': t5_vlm_outputs
                }
                examples.append(example)
        
        # Save examples as JSON
        json_examples = [ex.copy() for ex in examples]
        for ex in json_examples: del ex['image_tensor']
        with open(os.path.join(output_dir, 'qualitative_examples.json'), 'w') as f:
            json.dump(json_examples, f, indent=4)
        
        # --- 4. Create Visual Comparison Figures ---
        self.create_comparison_figures(examples, output_dir)
        return examples
    
    def create_comparison_figures(self, examples, output_dir):
        """Create 5-panel matplotlib figures comparing all model outputs."""
        for i, example in enumerate(tqdm(examples, desc="Creating figures")):
            fig, axes = plt.subplots(1, 5, figsize=(24, 6), gridspec_kw={'width_ratios': [1, 1, 1, 1, 1]})
            fig.suptitle(f"Qualitative Comparison: Example {i+1}", fontsize=16)

            # Denormalize and display image
            img_tensor = example['image_tensor']
            mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
            img_tensor = img_tensor * std[:, None, None] + mean[:, None, None]
            img = np.clip(img_tensor.permute(1, 2, 0).numpy(), 0, 1)
            
            axes[0].imshow(img)
            axes[0].set_title("Input Image")
            axes[0].axis('off')
            
            # Display all text comparisons
            texts = [
                ("Ground Truth", example['ground_truth']),
                ("Ours-MTL (Hierarchical)", example['mtl_hierarchical']),
                ("Baseline-VLM (GIT-base)", example['vlm_git_base']),
                ("Baseline-VLM (ViT-T5)", example['vlm_vit_t5'])
            ]
            
            for j, (title, text) in enumerate(texts):
                wrapped_text = textwrap.fill(text, width=45)
                axes[j+1].text(0.05, 0.95, f"{title}:\n\n{wrapped_text}", 
                             transform=axes[j+1].transAxes, 
                             verticalalignment='top', fontsize=9)
                axes[j+1].axis('off')
            
            # Add MTL attributes to its panel for context
            attr_text = "Predicted Attributes:\n" + "\n".join([f"- {k}: {v}" for k, v in example['mtl_attributes'].items()])
            axes[2].text(0.05, 0.1, attr_text, transform=axes[2].transAxes, fontsize=7, color='blue', verticalalignment='bottom')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(output_dir, f'comparison_example_{i+1}.png'), dpi=300)
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate qualitative comparison examples for all models.")
    # --- FIX: Update argument list for all models ---
    parser.add_argument('--base_config', type=str, required=True, help='Path to the base config for the MTL model.')
    parser.add_argument('--mtl_model', type=str, required=True, help='Path to the trained MTL model checkpoint.')
    
    parser.add_argument('--git_vlm_config', type=str, required=True, help='Path to the GIT-base VLM config file.')
    parser.add_argument('--git_vlm_model', type=str, required=True, help='Path to the trained GIT-base VLM checkpoint.')
    
    parser.add_argument('--t5_vlm_config', type=str, required=True, help='Path to the ViT-T5 VLM config file.')
    parser.add_argument('--t5_vlm_model', type=str, required=True, help='Path to the trained ViT-T5 VLM checkpoint.')
    
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output images and JSON.')
    parser.add_argument('--num_examples', type=int, default=10, help='Number of examples to generate.')
    args = parser.parse_args()

    generator = QualitativeExampleGenerator(base_config_path=args.base_config)

    generator.generate_side_by_side_examples(
        mtl_model_path=args.mtl_model,
        git_vlm_config_path=args.git_vlm_config,
        git_vlm_model_path=args.git_vlm_model,
        t5_vlm_config_path=args.t5_vlm_config,
        t5_vlm_model_path=args.t5_vlm_model,
        output_dir=args.output_dir,
        num_examples=args.num_examples
    )

    print(f"\nSuccessfully generated {args.num_examples} qualitative examples.")
    print(f"Results saved in: {args.output_dir}")

if __name__ == '__main__':
    main()