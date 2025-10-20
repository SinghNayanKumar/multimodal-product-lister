import torch
import yaml
import json
import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import argparse
from transformers import AutoTokenizer

from src.data.test_dataloader import create_test_dataloader
from src.models.multitask_model import MultitaskModel
from src.models.baseline_git import GitBaselineModel
from transformers import GitProcessor
from src.evaluation.evaluate_hallucinations import calculate_hallucination_rate
from src.models.baselines.direct_vlm_model import DirectVLM

def parse_json_output(json_string: str) -> str:
    """
    Safely parses a JSON string from the VLM and extracts title and description.
    """
    try:
        json_match = re.search(r'\{.*\}', json_string, re.DOTALL)
        if json_match:
            json_string = json_match.group(0)
        data = json.loads(json_string)
        title = data.get('text', {}).get('title', 'N/A')
        description = data.get('text', {}).get('description', 'N/A')
        return f"title: {title} | description: {description}"
    except (json.JSONDecodeError, TypeError, AttributeError):
        return f"[JSON PARSING FAILED] {json_string}"

class HierarchicalGenerationEvaluator:
    def __init__(self, base_config_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        # Load data and mappings
        self.val_loader, _, self.mappings, self.tokenizer = create_test_dataloader(self.base_config)
        
        # Initialize text quality metrics
        self.rouge = Rouge()
        self.smoothing = SmoothingFunction().method1
        
    def load_mtl_model(self, model_path):
        """Load the MTL model."""
        model = MultitaskModel(self.base_config, self.mappings)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        return model
    
    def load_direct_vlm_model(self, config_path, model_path):
        """Load the DirectVLM model."""
        with open(config_path, 'r') as f:
            vlm_config = yaml.safe_load(f)
        
        model = GitBaselineModel(vlm_config) # <-- Use our class
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        
        # Load the GitProcessor, which includes the tokenizer
        vlm_processor = GitProcessor.from_pretrained(vlm_config['model']['model_name'])
            
        return model, vlm_processor # <-- Return the processor
    
    def load_t5_vlm_model(self, config_path, model_path): # <-- ADDED FUNCTION
        """Load the ViT-T5 DirectVLM model."""
        with open(config_path, 'r') as f:
            vlm_config = yaml.safe_load(f)
        
        model = DirectVLM(
            vision_model_name=vlm_config['model']['vision_model_name'],
            text_model_name=vlm_config['model']['text_model_name']
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        
        # Load its corresponding tokenizer
        tokenizer = AutoTokenizer.from_pretrained(vlm_config['model']['text_model_name'])
            
        return model, tokenizer
    
    def generate_predictions(self, model, model_type, tokenizer_to_use=None, use_hierarchical=True):
        """Generate predictions from a model."""
        model.eval()
        predictions = []
        attribute_predictions = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc=f"Generating {model_type} predictions")):
                if batch_idx >= 100:  # Limit to 100 batches for evaluation
                    break
                    
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
                
                pixel_values = batch['pixel_values']
                batch_size = pixel_values.shape[0]
                
                if model_type == 'mtl_hierarchical':
                    outputs = model.predict(pixel_values, self.tokenizer, self.mappings, use_hierarchical_prompt=True)
                    if not isinstance(outputs, list):
                        outputs = [outputs]
                    
                    for output in outputs:
                        predictions.append({
                            'generated_text': output['generated_text'],
                            'predicted_attributes': output['predicted_attributes'],
                            'predicted_price': output['predicted_price']
                        })
                        attribute_predictions.append(output['predicted_attributes'])
                
                elif model_type == 'mtl_non_hierarchical':
                    outputs = model.predict(pixel_values, self.tokenizer, self.mappings, use_hierarchical_prompt=False)
                    if not isinstance(outputs, list):
                        outputs = [outputs]
                    
                    for output in outputs:
                        predictions.append({
                            'generated_text': output['generated_text'],
                            'predicted_attributes': output['predicted_attributes'],
                            'predicted_price': output['predicted_price']
                        })
                        attribute_predictions.append(output['predicted_attributes'])
                
                elif model_type == 'direct_vlm':
                    generated_texts = model.predict(pixel_values, tokenizer_to_use)
                    
                    for text in generated_texts:
                        predictions.append({
                            'generated_text': text,
                            'predicted_attributes': {},  # DirectVLM doesn't predict structured attributes
                            'predicted_price': None
                        })
                
                elif model_type == 'vit_t5_vlm': # <-- ADDED BLOCK
                    generated_json = model.generate(pixel_values, tokenizer_to_use)
                    
                    for json_str in generated_json:
                        # Parse the JSON output to get clean text for metric calculation
                        parsed_text = parse_json_output(json_str)
                        predictions.append({
                            'generated_text': parsed_text,
                            'predicted_attributes': {},
                            'predicted_price': None
                        })
                
                # Store ground truth for comparison
                true_texts = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
                for i, pred in enumerate(predictions[-batch_size:]):
                    pred['true_text'] = true_texts[i] if i < len(true_texts) else ""
        
        return predictions, attribute_predictions
    
    def calculate_text_quality_metrics(self, predictions):
        """Calculate BLEU, ROUGE, and other text metrics."""
        generated_texts = [p['generated_text'] for p in predictions]
        true_texts = [p['true_text'] for p in predictions if p['true_text']]
        
        if not generated_texts or not true_texts:
            return {}
        
        # Truncate to minimum length
        min_len = min(len(generated_texts), len(true_texts))
        generated_texts = generated_texts[:min_len]
        true_texts = true_texts[:min_len]
        
        # Calculate BLEU scores
        bleu_scores = []
        for gen, ref in zip(generated_texts, true_texts):
            ref_tokens = ref.split()
            gen_tokens = gen.split()
            if len(ref_tokens) > 0 and len(gen_tokens) > 0:
                bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=self.smoothing)
                bleu_scores.append(bleu)
        
        # Calculate ROUGE scores
        rouge_scores = []
        for gen, ref in zip(generated_texts, true_texts):
            if len(gen.strip()) > 0 and len(ref.strip()) > 0:
                try:
                    rouge_score = self.rouge.get_scores(gen, ref)[0]
                    rouge_scores.append(rouge_score)
                except:
                    continue
        
        return {
            'bleu': np.mean(bleu_scores) if bleu_scores else 0,
            'rouge_1': np.mean([s['rouge-1']['f'] for s in rouge_scores]) if rouge_scores else 0,
            'rouge_2': np.mean([s['rouge-2']['f'] for s in rouge_scores]) if rouge_scores else 0,
            'rouge_l': np.mean([s['rouge-l']['f'] for s in rouge_scores]) if rouge_scores else 0,
            'num_samples': min_len
        }
    
    def calculate_hallucination_rate(self, predictions):
        """Calculate attribute hallucination rate for MTL models."""
        # Filter out predictions without attributes (DirectVLM)
        mtl_predictions = [p for p in predictions if p['predicted_attributes']]
        
        if not mtl_predictions:
            return 0.0
        
        # Create DataFrame for hallucination calculation
        halluc_data = []
        for pred in mtl_predictions:
            row = {'generated_text': pred['generated_text']}
            row.update(pred['predicted_attributes'])
            halluc_data.append(row)
        
        halluc_df = pd.DataFrame(halluc_data)
        return calculate_hallucination_rate(halluc_df, list(self.mappings.keys()))
    
    def run_experiment_2_1(self, mtl_model_path, git_vlm_config_path, git_vlm_model_path, t5_vlm_config_path, t5_vlm_model_path, output_dir):
        """Run Experiment 2.1: Hierarchical vs Direct Generation."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("Loading models...")
        mtl_model = self.load_mtl_model(mtl_model_path)
        git_vlm_model, git_vlm_processor = self.load_direct_vlm_model(git_vlm_config_path, git_vlm_model_path)
        t5_vlm_model, t5_vlm_tokenizer = self.load_t5_vlm_model(t5_vlm_config_path, t5_vlm_model_path)
        
        results = {}
        
        # 1. MTL with Hierarchical Generation (Ours-MTL)
        print("\nEvaluating MTL with Hierarchical Generation...")
        mtl_hier_preds, mtl_hier_attrs = self.generate_predictions(
            mtl_model, 'mtl_hierarchical', use_hierarchical=True
        )
        
        results['mtl_hierarchical'] = {
            'text_metrics': self.calculate_text_quality_metrics(mtl_hier_preds),
            'hallucination_rate': self.calculate_hallucination_rate(mtl_hier_preds),
            'num_predictions': len(mtl_hier_preds)
        }
        
        # 2. MTL without Hierarchical Generation (Ours-Ablate-NoHier)
        print("\nEvaluating MTL without Hierarchical Generation...")
        mtl_no_hier_preds, _ = self.generate_predictions(
            mtl_model, 'mtl_non_hierarchical', use_hierarchical=False
        )
        
        results['mtl_non_hierarchical'] = {
            'text_metrics': self.calculate_text_quality_metrics(mtl_no_hier_preds),
            'hallucination_rate': self.calculate_hallucination_rate(mtl_no_hier_preds),
            'num_predictions': len(mtl_no_hier_preds)
        }
        
        # 3. Direct VLM (GIT-base) Baseline
        print("\nEvaluating Direct VLM (GIT-base)...")
        git_vlm_preds, _ = self.generate_predictions(git_vlm_model, 'direct_vlm', git_vlm_processor)
        
        results['direct_vlm_git'] = {
            'text_metrics': self.calculate_text_quality_metrics(git_vlm_preds),
            'hallucination_rate': 0.0,
            'num_predictions': len(git_vlm_preds)
        }
        
        # 4. Direct VLM (ViT-T5) Baseline <-- ADDED BLOCK
        print("\nEvaluating Direct VLM (ViT-T5)...")
        t5_vlm_preds, _ = self.generate_predictions(t5_vlm_model, 'vit_t5_vlm', t5_vlm_tokenizer)
        
        results['direct_vlm_t5'] = {
            'text_metrics': self.calculate_text_quality_metrics(t5_vlm_preds),
            'hallucination_rate': 0.0, 
            'num_predictions': len(t5_vlm_preds)
        }
        
        # Save detailed results
        with open(os.path.join(output_dir, 'hierarchical_generation_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        # Create summary table
        self.create_summary_table(results, output_dir)
        
        # Save example predictions
        self.save_prediction_examples(mtl_hier_preds, mtl_no_hier_preds, git_vlm_preds, t5_vlm_preds, output_dir)
        
        return results
    
    def create_summary_table(self, results, output_dir):
        """Create a summary comparison table."""
        summary = []
        
        for model_name, metrics in results.items():
            row = {
                'Model': model_name,
                'BLEU': f"{metrics['text_metrics']['bleu']:.4f}",
                'ROUGE-1': f"{metrics['text_metrics']['rouge_1']:.4f}",
                'ROUGE-2': f"{metrics['text_metrics']['rouge_2']:.4f}",
                'ROUGE-L': f"{metrics['text_metrics']['rouge_l']:.4f}",
                'Hallucination Rate': f"{metrics['hallucination_rate']:.4f}",
                'Samples': metrics['num_predictions']
            }
            summary.append(row)
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(output_dir, 'hierarchical_generation_summary.csv'), index=False)
        
        print("\n" + "="*80)
        print("EXPERIMENT 2.1: HIERARCHICAL vs DIRECT GENERATION RESULTS")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("="*80)
        
        return summary_df
    
    def save_prediction_examples(self, mtl_hier, mtl_no_hier, git_vlm_preds, t5_vlm_preds, output_dir, num_examples=10):
        """Save example predictions for qualitative analysis."""
        examples = []
        
        # --- FIX: Find the minimum length across all prediction lists ---
        min_len = min(len(mtl_hier), len(mtl_no_hier), len(git_vlm_preds), len(t5_vlm_preds))
        
        for i in range(min(num_examples, min_len)):
            example = {
                'index': i,
                'true_text': mtl_hier[i]['true_text'],
                'mtl_hierarchical': mtl_hier[i]['generated_text'],
                'mtl_non_hierarchical': mtl_no_hier[i]['generated_text'],
                # --- FIX: Add separate keys for each VLM baseline ---
                'direct_vlm_git': git_vlm_preds[i]['generated_text'],
                'direct_vlm_t5': t5_vlm_preds[i]['generated_text'],
                'predicted_attributes': mtl_hier[i]['predicted_attributes']
            }
            examples.append(example)
        
        with open(os.path.join(output_dir, 'prediction_examples.json'), 'w') as f:
            json.dump(examples, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Evaluate hierarchical generation vs direct VLM")
    parser.add_argument('--base_config', type=str, default='configs/base_config.yaml')
    parser.add_argument('--mtl_model', type=str, default='results/exp001_base_multitask/model_best.pth')

    # --- FIX: Use the correct default config file for the GIT model ---
    # This config MUST contain the 'model: model_name:' keys.
    parser.add_argument('--git_vlm_config', type=str, default='configs/exp2_1_direct_vlm.yaml', 
                        help="Config for the GIT-base VLM, must contain model_name.")
    parser.add_argument('--git_vlm_model', type=str, default='results/exp2_1_direct_vlm/model_best.pth')
    
    # --- FIX: Use the correct default config file for the ViT-T5 model ---
    # This config must contain 'vision_model_name' and 'text_model_name'.
    parser.add_argument('--t5_vlm_config', type=str, default='configs/config_direct_vlm.yaml',
                        help="Config for the ViT-T5 VLM.")
    parser.add_argument('--t5_vlm_model', type=str, default='results/direct_vlm_vit-t5/model_best.pth')
    
    parser.add_argument('--output_dir', type=str, default='results/experiment_2_1_hierarchical_generation')
    args = parser.parse_args()
    
    evaluator = HierarchicalGenerationEvaluator(args.base_config)
    results = evaluator.run_experiment_2_1(
        args.mtl_model, 
        args.git_vlm_config, 
        args.git_vlm_model,
        args.t5_vlm_config,
        args.t5_vlm_model,
        args.output_dir
    )
    
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == '__main__':
    main()