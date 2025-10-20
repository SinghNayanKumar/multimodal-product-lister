import torch
import yaml
import json
import pandas as pd
import os
import argparse
from tqdm import tqdm

from src.data.test_dataloader import create_test_dataloader
from src.models.multitask_model import MultitaskModel
from src.generation.suggestion_engine import SuggestionEngine

class UserStudyDataGenerator:
    def __init__(self, base_config_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)

        self.val_loader, _, self.mappings, self.tokenizer = create_test_dataloader(self.base_config)

        # Initialize suggestion engine
        self.suggestion_engine = SuggestionEngine(
            market_data_path=os.path.join(self.base_config['data']['processed_dir'], 
                                        self.base_config['data']['train_csv']),
            model_dir=self.base_config['suggestion_engine']['model_dir'],
            attribute_cols=self.base_config['suggestion_engine']['attribute_cols'],
            price_col=self.base_config['suggestion_engine']['price_col'],
            rating_col=self.base_config['suggestion_engine']['rating_col'],
            category_col=self.base_config['suggestion_engine']['category_col']
        )
    
    def generate_ab_comparison_data(self, model_path, output_dir, num_samples=30):
        """Generate A/B comparison data for user study."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Load MTL model
        model = MultitaskModel(self.base_config, self.mappings)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval().to(self.device)
        
        study_data = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Generating user study data")):
                if len(study_data) >= num_samples:
                    break
                
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
                
                pixel_values = batch['pixel_values'][:1]  # One item per batch
                
                # Generate predictions
                outputs = model.predict(pixel_values, self.tokenizer, self.mappings)
                if not isinstance(outputs, list):
                    outputs = [outputs]
                
                output = outputs[0]
                
                # Generate suggestions
                suggestions = self.suggestion_engine.generate_suggestions(output['predicted_attributes'])
                
                # Create Version A (without suggestions) and Version B (with suggestions)
                version_a = {
                    "predicted_attributes": output['predicted_attributes'],
                    "suggested_price": f"${output['predicted_price']:.2f}",
                    "generated_listing": output['generated_text']
                }
                
                version_b = {
                    "predicted_attributes": output['predicted_attributes'],
                    "suggested_price": f"${output['predicted_price']:.2f}",
                    "generated_listing": output['generated_text'],
                    "strategic_suggestions": suggestions
                }
                
                # Denormalize image for saving
                img_tensor = pixel_values[0].cpu()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_tensor = img_tensor * std + mean
                
                study_item = {
                    "item_id": len(study_data) + 1,
                    "version_a": version_a,
                    "version_b": version_b,
                    "image_tensor_shape": list(img_tensor.shape)  # For reference
                }
                
                study_data.append(study_item)
                
                # Save image separately
                from PIL import Image
                img_array = (img_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
                img = Image.fromarray(img_array)
                img.save(os.path.join(output_dir, f'item_{len(study_data)}_image.png'))
        
        # Save study data
        with open(os.path.join(output_dir, 'user_study_data.json'), 'w') as f:
            json.dump(study_data, f, indent=4)
        
        # Create evaluation template
        evaluation_template = {
            "instructions": "For each item, compare Version A and Version B. Which version would be more valuable for an e-commerce manager?",
            "items": [
                {
                    "item_id": item["item_id"],
                    "preference": "",  # To be filled by evaluator: "A" or "B"
                    "reason": ""       # To be filled by evaluator
                }
                for item in study_data
            ]
        }
        
        with open(os.path.join(output_dir, 'evaluation_template.json'), 'w') as f:
            json.dump(evaluation_template, f, indent=4)
        
        print(f"Generated {len(study_data)} items for user study")
        print(f"Files saved to: {output_dir}")
        
        return study_data

def main():
    parser = argparse.ArgumentParser(description="Generate data for user study")
    parser.add_argument('--base_config', type=str, default='configs/base_config.yaml')
    parser.add_argument('--model_path', type=str, default='results/exp001_base_multitask/model_best.pth')
    parser.add_argument('--output_dir', type=str, default='results/user_study')
    parser.add_argument('--num_samples', type=int, default=30)
    args = parser.parse_args()
    
    generator = UserStudyDataGenerator(args.base_config)
    study_data = generator.generate_ab_comparison_data(
        args.model_path, args.output_dir, args.num_samples
    )
    
    print(f"User study data generated successfully!")

if __name__ == '__main__':
    main()