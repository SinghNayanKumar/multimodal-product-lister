import torch
import yaml
import argparse
import os
import json
import evaluate

# --- Import the core model class ---
from transformers import Seq2SeqTrainingArguments, default_data_collator, VisionEncoderDecoderModel

from src.data.dataloader_vlm import create_datasets
# We no longer need the DirectVLM wrapper for this script
from src.training.train_direct_vlm_fixed import CustomSeq2SeqTrainer

def main(config_path, model_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load tokenizer and validation set as before
    _, val_dataset, tokenizer = create_datasets(config)

    # --- THIS IS THE DEFINITIVE FIX ---
    # Load the entire fine-tuned model from the saved directory.
    # This single command handles architecture, config, token resizing, and weights.
    # It replaces all the previous manual loading steps.
    print(f"Loading complete fine-tuned model from {model_path}...")
    model = VisionEncoderDecoderModel.from_pretrained(model_path)

    # The Trainer requires the model to be on the correct device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    rouge = evaluate.load("rouge")
    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"]
        return {"rouge2_fmeasure": round(rouge_output, 4)}

    training_args = Seq2SeqTrainingArguments(
        output_dir="results/evaluation_only/",
        predict_with_generate=True,
        per_device_eval_batch_size=config['training']['batch_size'],
        generation_max_length=128,
        eval_strategy="epoch", 
    )

    # The CustomSeq2SeqTrainer will work with the raw VisionEncoderDecoderModel
    # because it has the necessary .forward() and .generate() methods.
    trainer = CustomSeq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    print("Running final evaluation with the correctly loaded model...")
    # Assume the model you are evaluating is from the final epoch for naming purposes
    trainer.state.epoch = config['training']['epochs'] 
    final_metrics = trainer.evaluate()

    print("\n--- Evaluation Complete ---")
    print(final_metrics)
    print("\nCheck 'results/evaluation_only/' for the true, full-length predictions.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained DirectVLM model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the original config file.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model directory (e.g., results/exp2_1_direct_vlm_pad/model_best).')
    args = parser.parse_args()
    main(args.config, args.model_path)