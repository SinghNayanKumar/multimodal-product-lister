import argparse
import os
import torch
import yaml
from transformers import AutoTokenizer
from collections import OrderedDict

# Make sure the script can find your custom model class
from src.models.baselines.direct_vlm_model import DirectVLM

def main(args):
    """
    Loads a model's weights from a training checkpoint and saves it
    in a final, inference-ready format using .save_pretrained(),
    which correctly generates the config.json file.
    
    This version includes logic to strip the 'model.' prefix from checkpoint keys.
    """
    print("--- Starting Checkpoint to Final Model Conversion ---")

    # 1. Load the training configuration.
    print(f"Loading training config from: {args.config_path}")
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Create an empty "shell" of your DirectVLM model.
    print("Building model shell in memory...")
    model = DirectVLM(
        vision_model_name=config['model']['vision_model_name'],
        text_model_name=config['model']['text_model_name']
    )

    # 3. Locate and load the trained weights from the checkpoint file.
    weights_path = os.path.join(args.checkpoint_dir, 'pytorch_model.bin')
    if not os.path.exists(weights_path):
        print(f"FATAL: Could not find 'pytorch_model.bin' in {args.checkpoint_dir}")
        return
        
    print(f"Loading trained weights from: {weights_path}")
    state_dict = torch.load(weights_path, map_location='cpu')

    # --- START OF THE FIX ---
    # Create a new, ordered dictionary to hold the cleaned keys.
    new_state_dict = OrderedDict()
    prefix = "model."
    for k, v in state_dict.items():
        if k.startswith(prefix):
            # If the key has the prefix, remove it.
            name = k[len(prefix):]
            new_state_dict[name] = v
        else:
            # If for some reason a key doesn't have the prefix, keep it.
            new_state_dict[k] = v
    
    print("Cleaned state_dict by removing 'model.' prefix from keys.")
    
    # Load the new, cleaned-up state dictionary into our model shell.
    # This will now find a perfect match for every key.
    model.load_state_dict(new_state_dict)
    # --- END OF THE FIX ---


    # 4. Save the fully-loaded model to a new directory.
    print(f"Saving final, inference-ready model to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)

    # 5. Save the tokenizer to the final directory for completeness.
    print("Saving tokenizer to the final model directory...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n" + "="*50)
    print("✅ Conversion Complete!")
    print(f"A final model directory has been created at: {args.output_dir}")
    print("This directory now contains config.json and can be loaded with from_pretrained().")
    print("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a training checkpoint to a final, loadable model directory.")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the original .yaml config file used for training.")
    parser.add_argument('--checkpoint_dir', type=str, required=True, help="Path to the training checkpoint directory (e.g., '.../checkpoint-4272').")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the new, clean directory to save the final model.")
    args = parser.parse_args()
    main(args)