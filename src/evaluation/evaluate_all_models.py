import torch
import yaml
import json
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import argparse

from src.data.test_dataloader import create_test_dataloader
from src.models.multitask_model import MultitaskModel
from src.models.baselines.siloed_model import SiloedModel
from src.models.baselines.direct_vlm import DirectVLM
from src.evaluation.evaluate_hallucinations import calculate_hallucination_rate

def load_model(model_type, config_path, model_path, mappings):
    """Load a trained model based on its type."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if model_type == 'mtl':
        model = MultitaskModel(config, mappings)
    elif model_type == 'siloed_attributes':
        model = SiloedModel(config, mappings)
    elif model_type == 'siloed_price':
        model = SiloedModel(config)
    elif model_type == 'direct_vlm':
        model = DirectVLM(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model, config

def evaluate_model_predictions(model, model_type, dataloader, device, mappings, tokenizer):
    """Generate predictions and calculate metrics for a model."""
    model.eval()
    model.to(device)
    
    results = {
        'predictions': [],
        'price_metrics': {},
        'attribute_metrics': {},
        'text_metrics': {}
    }
    
    all_price_preds, all_price_targets = [], []
    attr_preds, attr_targets = {k: [] for k in mappings.keys()}, {k: [] for k in mappings.keys()}
    generated_texts, target_texts = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {model_type}"):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
            
            if model_type == 'mtl':
                # MTL model can do everything
                outputs = model(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                # Price predictions
                price_preds = outputs['price_pred'].cpu().numpy()
                all_price_preds.extend(price_preds)
                
                # Attribute predictions
                for attr_name, logits in outputs['attribute_logits'].items():
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    attr_preds[attr_name].extend(preds)
                
                # Text generation using predict method
                pixel_values = batch['pixel_values']
                text_outputs = model.predict(pixel_values, tokenizer, mappings)
                if not isinstance(text_outputs, list):
                    text_outputs = [text_outputs]
                
                for output in text_outputs:
                    generated_texts.append(output['generated_text'])
            
            elif model_type == 'siloed_attributes':
                outputs = model(pixel_values=batch['pixel_values'])
                for attr_name, logits in outputs['attribute_logits'].items():
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    attr_preds[attr_name].extend(preds)
            
            elif model_type == 'siloed_price':
                outputs = model(pixel_values=batch['pixel_values'])
                price_preds = outputs['price_pred'].cpu().numpy()
                all_price_preds.extend(price_preds)
            
            elif model_type == 'direct_vlm':
                generated = model.predict(batch['pixel_values'], tokenizer)
                generated_texts.extend(generated)
            
            # Collect targets
            all_price_targets.extend(batch['price_target'].cpu().numpy())
            for attr_name in mappings.keys():
                targets = batch[f"{attr_name}_target"].cpu().numpy()
                attr_targets[attr_name].extend(targets)
            
            # Decode target texts
            target_batch = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
            target_texts.extend(target_batch)
    
    # Calculate price metrics
    if all_price_preds:
        price_preds_orig = np.expm1(np.array(all_price_preds))
        price_targets_orig = np.expm1(np.array(all_price_targets))
        results['price_metrics'] = {
            'mae': mean_absolute_error(price_targets_orig, price_preds_orig),
            'rmse': np.sqrt(mean_squared_error(price_targets_orig, price_preds_orig)),
            'r2': r2_score(price_targets_orig, price_preds_orig)
        }
    
    # Calculate attribute metrics
    if any(attr_preds.values()):
        f1_scores = []
        for attr_name in mappings.keys():
            if attr_preds[attr_name]:
                f1 = f1_score(attr_targets[attr_name], attr_preds[attr_name], average='macro', zero_division=0)
                f1_scores.append(f1)
        results['attribute_metrics'] = {
            'f1_macro': np.mean(f1_scores) if f1_scores else 0,
            'individual_f1': {attr: f1_score(attr_targets[attr], attr_preds[attr], average='macro', zero_division=0) 
                            for attr in mappings.keys() if attr_preds[attr]}
        }
    
    # Calculate text metrics
    if generated_texts and target_texts:
        results['text_metrics'] = calculate_text_metrics(generated_texts[:len(target_texts)], target_texts)
        
        # Calculate hallucination rate if we have attributes
        if model_type == 'mtl':
            # Create DataFrame for hallucination calculation
            hallucination_data = {
                'generated_text': generated_texts[:len(target_texts)]
            }
            # Add predicted attributes (need to decode from indices)
            inverse_mappings = {attr: {i: label for label, i in mapping.items()} 
                              for attr, mapping in mappings.items()}
            
            for attr_name in mappings.keys():
                decoded_attrs = [inverse_mappings[attr_name].get(pred, 'Unknown') 
                               for pred in attr_preds[attr_name][:len(generated_texts)]]
                hallucination_data[attr_name] = decoded_attrs
            
            halluc_df = pd.DataFrame(hallucination_data)
            results['text_metrics']['hallucination_rate'] = calculate_hallucination_rate(
                halluc_df, list(mappings.keys())
            )
    
    return results

def calculate_text_metrics(generated_texts, reference_texts):
    """Calculate BLEU, ROUGE, and other text quality metrics."""
    bleu_scores = []
    rouge = Rouge()
    rouge_scores = []
    
    smoothing = SmoothingFunction().method1
    
    for gen, ref in zip(generated_texts, reference_texts):
        # BLEU score
        ref_tokens = ref.split()
        gen_tokens = gen.split()
        bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothing)
        bleu_scores.append(bleu)
        
        # ROUGE scores
        try:
            rouge_score = rouge.get_scores(gen, ref)[0]
            rouge_scores.append(rouge_score)
        except:
            continue
    
    # Average scores
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
    
    if rouge_scores:
        avg_rouge_1 = np.mean([score['rouge-1']['f'] for score in rouge_scores])
        avg_rouge_2 = np.mean([score['rouge-2']['f'] for score in rouge_scores])
        avg_rouge_l = np.mean([score['rouge-l']['f'] for score in rouge_scores])
    else:
        avg_rouge_1 = avg_rouge_2 = avg_rouge_l = 0
    
    return {
        'bleu': avg_bleu,
        'rouge_1': avg_rouge_1,
        'rouge_2': avg_rouge_2,
        'rouge_l': avg_rouge_l
    }

def main():
    parser = argparse.ArgumentParser(description="Comprehensive evaluation of all trained models")
    parser.add_argument('--base_config', type=str, default='configs/base_config.yaml', help='Base config for data loading')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Directory to save results')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load mappings and test data
    with open(args.base_config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Create test dataloader
    test_dataloader, _, mappings, tokenizer = create_test_dataloader(base_config)
    
    # Load test data (you might need to modify this based on your data split)
    test_df = pd.read_csv(os.path.join(base_config['data']['processed_dir'], 'test.csv'))
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define models to evaluate
    models_to_evaluate = [
        ('mtl', 'configs/base_config.yaml', 'results/exp001_base_multitask/model_best.pth'),
        ('siloed_attributes', 'configs/exp1_1_siloed_attributes.yaml', 'results/exp1_1_siloed_attributes/model_best.pth'),
        ('siloed_price', 'configs/exp1_1_siloed_price.yaml', 'results/exp1_1_siloed_price/model_best.pth'),
        ('direct_vlm', 'configs/exp2_1_direct_vlm.yaml', 'results/exp2_1_direct_vlm/model_best.pth')
    ]
    
    all_results = {}
    
    for model_type, config_path, model_path in models_to_evaluate:
        if os.path.exists(model_path):
            print(f"\nEvaluating {model_type} model...")
            model, config = load_model(model_type, config_path, model_path, mappings)
            results = evaluate_model_predictions(model, model_type, test_dataloader, device, mappings, tokenizer)
            all_results[model_type] = results
            
            # Save individual results
            with open(os.path.join(args.output_dir, f'{model_type}_results.json'), 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                json_results = json.loads(json.dumps(results, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
                json.dump(json_results, f, indent=4)
        else:
            print(f"Model not found: {model_path}")
    
    # Create summary comparison table
    create_results_summary(all_results, args.output_dir)
    print(f"\nEvaluation complete. Results saved to {args.output_dir}/")

def create_results_summary(all_results, output_dir):
    """Create a summary table comparing all models."""
    summary = []
    
    for model_type, results in all_results.items():
        row = {'Model': model_type}
        
        # Price metrics
        if results['price_metrics']:
            row.update({
                'Price MAE': results['price_metrics'].get('mae', 'N/A'),
                'Price RMSE': results['price_metrics'].get('rmse', 'N/A'),
                'Price R²': results['price_metrics'].get('r2', 'N/A')
            })
        
        # Attribute metrics
        if results['attribute_metrics']:
            row['Attribute F1'] = results['attribute_metrics'].get('f1_macro', 'N/A')
        
        # Text metrics
        if results['text_metrics']:
            row.update({
                'BLEU': results['text_metrics'].get('bleu', 'N/A'),
                'ROUGE-L': results['text_metrics'].get('rouge_l', 'N/A'),
                'Hallucination Rate': results['text_metrics'].get('hallucination_rate', 'N/A')
            })
        
        summary.append(row)
    
    # Save as CSV
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_dir, 'model_comparison_summary.csv'), index=False)
    print("\nModel Comparison Summary:")
    print(summary_df.to_string(index=False))

if __name__ == '__main__':
    main()