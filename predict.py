# --- 1. Import Necessary Libraries ---
# ANNOTATION: Standard libraries for file paths, argument parsing, and data handling.
import torch
import yaml
import json
import argparse
import os

# ANNOTATION: PIL (Pillow) is used for loading and handling images.
from PIL import Image

# ANNOTATION: We import the specific processors from Hugging Face. It's crucial that these
# match the models defined in the config file to ensure data is processed correctly.
from transformers import ViTImageProcessor, AutoTokenizer

# ANNOTATION: We import our custom-built modules. This modular structure is key to a clean codebase.
from src.models.multitask_model import MultitaskModel
from src.generation.suggestion_engine import SuggestionEngine

def main(args):
    """
    Main function to run the end-to-end prediction pipeline.
    It loads the model and necessary processors, processes the input image,
    generates the full e-commerce listing, and provides strategic suggestions.
    """
    
    # --- 2. Load Configuration and Mappings ---
    # ANNOTATION: Using a YAML config file is a best practice. It separates our code from
    # hyperparameters and paths, making the script reusable and experiments easy to manage.
    try:
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config_path}")
        return

    # ANNOTATION: The mappings file is essential. It contains the dictionaries that translate
    # human-readable attribute labels (e.g., 'V-Neck') to the integer IDs the model was trained on.
    # Our updated model's `predict` method will need this to decode its own output.
    mappings_path = os.path.join(config['data']['processed_dir'], config['data']['mappings'])
    try:
        with open(mappings_path, 'r') as f:
            mappings = json.load(f)
    except FileNotFoundError:
        print(f"Error: Mappings file not found at {mappings_path}")
        return

    # ANNOTATION: Set the computation device. We prioritize CUDA-enabled GPUs for performance
    # but fall back to the CPU if one isn't available.
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 3. Initialize Model and Processors ---
    # ANNOTATION: We first initialize the model architecture using the config and mappings.
    # The mappings are required so the model can build the correct number of output neurons
    # for each attribute classification head.
    model = MultitaskModel(config, mappings)
    
    # ANNOTATION: Next, we load the learned weights from our trained checkpoint file (.pth).
    # `map_location=device` ensures the model weights are loaded onto the correct device.
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    # ANNOTATION: This is a CRITICAL step for inference. `model.eval()` puts the model in
    # evaluation mode, which disables layers like Dropout and affects Batch Normalization.
    # Forgetting this can lead to inconsistent and incorrect predictions.
    model.eval()

    # ANNOTATION: Initialize the same image processor and tokenizer that were used during training.
    image_processor = ViTImageProcessor.from_pretrained(config['model']['vision_model_name'])
    tokenizer = AutoTokenizer.from_pretrained(config['model']['text_model_name'])

    # --- 4. Process the Input Image ---
    try:
        # ANNOTATION: Open the image and convert to 'RGB' to ensure a consistent 3-channel format,
        # which the ViT model expects.
        image = Image.open(args.image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Input image not found at {args.image_path}")
        return
        
    # ANNOTATION: Use the image processor to perform all necessary transformations: resizing,
    # normalization, and conversion to a PyTorch tensor. The result is a 'pixel_values' tensor
    # ready to be fed into the model.
    pixel_values = image_processor(images=image, return_tensors="pt")['pixel_values'].to(device)

    # --- 5. Generate Predictions from the Model ---
    # ANNOTATION: `torch.no_grad()` is a context manager that disables gradient calculations.
    # This is essential for inference as it significantly reduces memory usage and speeds up computation.
    with torch.no_grad():
        # ANNOTATION: This is the core inference call. We use our powerful, custom `predict` method
        # on the model. This single method encapsulates the entire hierarchical generation process:
        # 1. It predicts structured attributes and price from the image.
        # 2. It builds a factual prompt from those predictions.
        # 3. It generates text conditioned on BOTH the prompt AND the image's visual features.
        # This keeps our prediction script clean and puts the complex logic where it belongs: in the model class.
        outputs = model.predict(
            pixel_values=pixel_values, 
            tokenizer=tokenizer,
            attribute_mappers=mappings
        )

    # --- 6. Run the Prescriptive Analytics Pipeline ---
    # ANNOTATION: This step demonstrates the final, novel part of our system. We now initialize
    # the SuggestionEngine by pointing it to the pre-trained models. This is fast and correct for inference.
    print("Initializing suggestion engine and generating strategic advice...")
    suggestion_engine = SuggestionEngine(
        market_data_path=os.path.join(config['data']['processed_dir'], config['data']['train_csv']),
        model_dir=config['suggestion_engine']['model_dir'],
        attribute_cols=config['suggestion_engine']['attribute_cols'],
        price_col=config['suggestion_engine']['price_col'],
        rating_col=config['suggestion_engine']['rating_col'],
        category_col=config['suggestion_engine']['category_col']
    )
    # ANNOTATION: The engine runs "what-if" scenarios to find opportunities for improvement.
    suggestions = suggestion_engine.generate_suggestions(outputs['predicted_attributes'])

    # --- 7. Format and Display the Final Output ---
    # ANNOTATION: The `outputs` dictionary from the model already contains clean, decoded data.
    # We just need to assemble it into the final JSON structure for our application.
    result = {
        "predicted_attributes": outputs['predicted_attributes'],
        "suggested_price": f"{outputs['predicted_price']:.2f}",
        "generated_listing": outputs['generated_text'],
        "strategic_suggestions": suggestions
    }

    # ANNOTATION: Print the final, comprehensive result to the console in a readable format.
    print("\n--- Generated E-Commerce Listing ---")
    print(json.dumps(result, indent=4))
    
    # ANNOTATION: If an output path is provided, save the result to a file for later use.
    if args.output_path:
        with open(args.output_path, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"\n✅ Output saved to {args.output_path}")


if __name__ == '__main__':
    # ANNOTATION: `argparse` creates a user-friendly command-line interface (CLI) for our script.
    # This makes it easy to run predictions on different images and models without changing the code.
    parser = argparse.ArgumentParser(description="Generate a complete e-commerce listing from a single product image.")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input product image.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained MultitaskModel checkpoint (.pth).")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the model's training configuration file (.yaml).")
    parser.add_argument('--output_path', type=str, default=None, help="Optional path to save the output JSON file.")
    
    args = parser.parse_args()
    main(args)