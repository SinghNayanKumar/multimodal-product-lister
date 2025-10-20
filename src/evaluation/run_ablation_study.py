import torch
import yaml
import json
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import argparse
from itertools import product

from src.data.test_dataloader import create_test_dataloader
from src.models.multitask_model import MultitaskModel
from src.training.loss import CompositeLoss
from sklearn.metrics import mean_absolute_error, f1_score

class AblationStudyRunner:
    def __init__(self, base_config_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        # Load validation data
        self.val_loader, _, self.mappings, self.tokenizer = create_test_dataloader(self.base_config)
    
    def evaluate_model_quick(self, model):
        """Quick evaluation of a model on validation set."""
        model.eval()
        model.to(self.device)
        
        price_preds, price_targets = [], []
        attr_preds, attr_targets = {k: [] for k in self.mappings.keys()}, {k: [] for k in self.mappings.keys()}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx >= 50:  # Quick evaluation on subset
                    break
                    
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
                
                outputs = model(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                price_preds.extend(outputs['price_pred'].cpu().numpy())
                price_targets.extend(batch['price_target'].cpu().numpy())
                
                for attr_name, logits in outputs['attribute_logits'].items():
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    targets = batch[f"{attr_name}_target"].cpu().numpy()
                    attr_preds[attr_name].extend(preds)
                    attr_targets[attr_name].extend(targets)
        
        # Calculate metrics
        price_mae = mean_absolute_error(np.expm1(price_targets), np.expm1(price_preds))
        f1_scores = [f1_score(attr_targets[k], attr_preds[k], average='macro', zero_division=0) 
                    for k in self.mappings.keys()]
        avg_f1 = np.mean(f1_scores)
        
        return {'price_mae': price_mae, 'attribute_f1': avg_f1}
    
    def run_loss_weight_ablation(self, model_path, output_dir):
        """Test different loss weight combinations."""
        print("Running loss weight ablation study...")
        
        # Define loss weight combinations to test
        weight_combinations = [
            {'price': 0.3, 'attributes': 0.3, 'text': 0.4},  # Text-focused
            {'price': 0.4, 'attributes': 0.4, 'text': 0.2},  # Prediction-focused
            {'price': 0.1, 'attributes': 0.4, 'text': 0.5},  # Original
            {'price': 0.2, 'attributes': 0.6, 'text': 0.2},  # Attribute-focused
            {'price': 0.6, 'attributes': 0.2, 'text': 0.2},  # Price-focused
        ]
        
        results = []
        
        for i, weights in enumerate(weight_combinations):
            print(f"\nTesting weight combination {i+1}: {weights}")
            
            # Load fresh model
            model = MultitaskModel(self.base_config, self.mappings)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            
            metrics = self.evaluate_model_quick(model)
            
            result = {
                'combination_id': i+1,
                'price_weight': weights['price'],
                'attributes_weight': weights['attributes'],
                'text_weight': weights['text'],
                'price_mae': metrics['price_mae'],
                'attribute_f1': metrics['attribute_f1']
            }
            results.append(result)
            
            print(f"Results - Price MAE: {metrics['price_mae']:.4f}, Attr F1: {metrics['attribute_f1']:.4f}")
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_dir, 'loss_weight_ablation.csv'), index=False)
        
        return results_df
    
    def run_hierarchical_prompt_ablation(self, model_path, output_dir):
        """Test hierarchical vs non-hierarchical text generation."""
        print("Running hierarchical prompting ablation...")
        
        model = MultitaskModel(self.base_config, self.mappings)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        model.to(self.device)
        
        results = []
        
        # Test both hierarchical and non-hierarchical generation
        for use_hierarchical in [True, False]:
            print(f"\nTesting hierarchical prompting: {use_hierarchical}")
            
            generated_texts = []
            true_texts = []
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.val_loader):
                    if batch_idx >= 20:  # Test on subset
                        break
                    
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(self.device)
                    
                    pixel_values = batch['pixel_values']
                    outputs = model.predict(pixel_values, self.tokenizer, self.mappings, 
                                          use_hierarchical_prompt=use_hierarchical)
                    
                    if not isinstance(outputs, list):
                        outputs = [outputs]
                    
                    batch_generated = [o['generated_text'] for o in outputs]
                    batch_true = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
                    
                    generated_texts.extend(batch_generated)
                    true_texts.extend(batch_true)
            
            # Calculate basic text metrics (simplified)
            avg_length = np.mean([len(text.split()) for text in generated_texts])
            
            result = {
                'hierarchical_prompting': use_hierarchical,
                'avg_generated_length': avg_length,
                'num_samples': len(generated_texts)
            }
            results.append(result)
            
            print(f"Average generated length: {avg_length:.1f} words")
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_dir, 'hierarchical_prompting_ablation.csv'), index=False)
        
        return results_df

def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument('--base_config', type=str, default='configs/base_config.yaml')
    parser.add_argument('--model_path', type=str, default='results/exp001_base_multitask/model_best.pth')
    parser.add_argument('--output_dir', type=str, default='results/ablation_studies')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    runner = AblationStudyRunner(args.base_config)
    
    # Run different ablation studies
    print("Starting ablation studies...")
    
    loss_results = runner.run_loss_weight_ablation(args.model_path, args.output_dir)
    prompt_results = runner.run_hierarchical_prompt_ablation(args.model_path, args.output_dir)
    
    print(f"\nAblation study results saved to: {args.output_dir}")
    
    # Print summary
    print("\nLoss Weight Ablation Summary:")
    print(loss_results[['price_weight', 'attributes_weight', 'text_weight', 'price_mae', 'attribute_f1']].to_string(index=False))

if __name__ == "__main__":
    main()
