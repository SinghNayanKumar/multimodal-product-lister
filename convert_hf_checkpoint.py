import torch
import yaml
import os

# Import the model class you want to convert
from src.models.baselines.direct_vlm_model import DirectVLM

def convert_hf_directory_to_pth(config_path, input_dir, output_path):
    """
    Loads a model from a Hugging Face Trainer directory and saves its 
    state_dict to a single .pth file for compatibility with our evaluation scripts.
    """
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Initialize the model architecture. This loads all pre-trained weights.
    print("Initializing DirectVLM model structure with pre-trained weights...")
    model = DirectVLM(
        vision_model_name=config['model']['vision_model_name'],
        text_model_name=config['model']['text_model_name']
    )

    # 2. Construct the path to the actual weights file
    weights_path = os.path.join(input_dir, 'pytorch_model.bin')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Could not find 'pytorch_model.bin' in the directory: {input_dir}")
    
    print(f"Loading fine-tuned weights from: {weights_path}")
    
    # 3. Load the state dictionary from the Hugging Face file
    state_dict = torch.load(weights_path, map_location='cpu')
    
    # 4. Load these weights into your model instance
    # --- FIX: Use strict=False ---
    # This tells PyTorch to only load the keys that are present in the file
    # and not to error out if keys (like the frozen vision_encoder) are missing.
    model.load_state_dict(state_dict, strict=False)
    print("Successfully loaded fine-tuned weights into the model.")

    # 5. Save the complete model's state_dict to the desired .pth file
    torch.save(model.state_dict(), output_path)
    print(f"\nSuccessfully converted and saved compatible checkpoint to: {output_path}")


if __name__ == '__main__':
    # --- CONFIGURE YOUR PATHS HERE ---
    
    # Path to the config file used to train the ViT-T5 model
    VLM_CONFIG_PATH = 'configs/config_direct_vlm.yaml'
    
    # Path to the directory containing the 'pytorch_model.bin' file
    INPUT_HF_DIRECTORY = 'results/exp2_1_direct_vlm_pad/model_best'
    
    # Define where you want to save the new single .pth file
    OUTPUT_PTH_FILE = 'results/exp2_1_direct_vlm_pad/t5_vlm_model_best.pth'
    
    # --- RUN THE CONVERSION ---
    convert_hf_directory_to_pth(VLM_CONFIG_PATH, INPUT_HF_DIRECTORY, OUTPUT_PTH_FILE)