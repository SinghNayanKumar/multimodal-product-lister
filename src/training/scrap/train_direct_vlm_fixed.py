# src/training/train_direct_vlm.py

import torch
import yaml
import argparse
import os
from PIL import Image
import numpy as np
import json

from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    default_data_collator
)
import evaluate

from src.data.dataloader_vlm import create_datasets
from src.models.baselines.direct_vlm_fixed import DirectVLM

# --- THE CORRECT APPROACH: SUBCLASSING SEQ2SEQTRAINER ---
# This new class inherits from Seq2SeqTrainer and extends its functionality.
class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    # We override the 'evaluate' method.
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        # 1. Run the standard evaluation procedure to get metrics.
        # super().evaluate() calls the original method from the parent class.
        eval_output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # 2. After evaluation, run our custom prediction logic.
        # We now have access to 'self' which is the trainer instance.
        print("\n--- Generating predictions for JSON output ---")
        
        # Use the provided eval_dataset or the default one from the trainer.
        dataset_to_predict = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        raw_predictions = self.predict(dataset_to_predict, metric_key_prefix="predict")
        
        decoded_preds = self.tokenizer.batch_decode(raw_predictions.predictions, skip_special_tokens=True)
        
        output_data = []
        for i, pred_text in enumerate(decoded_preds):
            original_data_row = dataset_to_predict.df.iloc[i]
            output_data.append({
                'image_path': original_data_row['image_path'],
                'prediction': pred_text,
                'ground_truth': f"title: {original_data_row['name']} | description: {original_data_row['description']}"
            })
            
        # self.state.epoch is available to get the current epoch
        output_file = os.path.join(self.args.output_dir, f"predictions_epoch_{int(self.state.epoch)}.json")
        print(f"Saving predictions to {output_file}")
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=4)
            
        # 3. Return the original evaluation output.
        return eval_output

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Create Datasets and Tokenizer (This now uses the article's [PAD] token method)
    train_dataset, val_dataset, tokenizer = create_datasets(config)

    # 2. Load Model
    model = DirectVLM(config)
    
    # 2a. Resize embeddings to match the new tokenizer
    model.model.decoder.resize_token_embeddings(len(tokenizer))
    
    # 3. Configure Special Tokens
    model.model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.model.config.pad_token_id = tokenizer.pad_token_id
    model.model.config.eos_token_id = tokenizer.eos_token_id
    
    # 4. Define Metrics
    rouge = evaluate.load("rouge")
    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"]
        return {"rouge2_fmeasure": round(rouge_output, 4)}

    # 5. Define Training Arguments
    output_dir = os.path.join(config['output_dir'], config['experiment_name'])
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=float(config['training']['learning_rate']),
        num_train_epochs=config['training']['epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        weight_decay=float(config['training'].get('weight_decay', 0.01)),
        save_total_limit=2,
        logging_steps=50,
        push_to_hub=False,  
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_safetensors=False,
        report_to="wandb",
        run_name=config['experiment_name'],
        
        
    )

    # 6. Instantiate our NEW Custom Trainer
    trainer = CustomSeq2SeqTrainer( # <-- USE THE NEW CUSTOM CLASS
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )
    # No callback is needed anymore.

    # 7. Start Training
    print(f"Starting training for {config['training']['epochs']} epochs...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 8. Save and Evaluate
    print("Training complete. Saving the best performing model.")
    trainer.save_model(os.path.join(output_dir, 'model_best'))
    print("\n--- Final Evaluation on the Best Model ---")
    final_metrics = trainer.evaluate()
    print(final_metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DirectVLM baseline model using Seq2SeqTrainer.")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    parser.add_argument(
        '--resume_from_checkpoint', 
        type=str, 
        default=None, 
        help='Path to a checkpoint to resume training from.'
    ) 
    args = parser.parse_args()
    main(args.config)