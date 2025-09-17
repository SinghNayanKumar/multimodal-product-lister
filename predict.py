import torch
import yaml
import json
import argparse
from PIL import Image
from transformers import ViTImageProcessor, T5Tokenizer

from src.models.multitask_model import MultitaskModel
# from src.generation.suggestion_engine import SuggestionEngine # TODO: Implement this file

def main(args):
    # --- Load Config and Mappings ---
    # We need the config to know the model structure
    # For simplicity, we assume a config is available.
    # A more robust solution would save config with the model.
    with open('experiments/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    with open(config['data']['processed_dir'] + '/' + config['data']['mappings']) as f:
        mappings = json.load(f)
    
    # Create inverse mappings to convert IDs back to labels
    inverse_mappings = {attr: {i: label for label, i in mapping.items()} for attr, mapping in mappings.items()}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load Model ---
    model = MultitaskModel(config, mappings)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- Load Processors ---
    image_processor = ViTImageProcessor.from_pretrained(config['model']['vision_model_name'])
    tokenizer = T5Tokenizer.from_pretrained(config['model']['text_model_name'])

    # --- Process Input Image ---
    image = Image.open(args.image_path).convert("RGB")
    pixel_values = image_processor(image, return_tensors="pt")['pixel_values'].to(device)

    # --- Get Model Predictions ---
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    # --- Decode Predictions ---
    # Price
    predicted_price = outputs['price_pred'].item()

    # Attributes
    predicted_attributes = {}
    for attr, logits in outputs['attribute_logits'].items():
        pred_id = torch.argmax(logits, dim=1).item()
        predicted_attributes[attr] = inverse_mappings[attr][pred_id]
        
    # Text Generation
    prompt = f"generate listing for: type: {predicted_attributes['Type']} | pattern: {predicted_attributes['Pattern']}"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated_ids = model.text_generator.generate(input_ids, max_length=128)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # --- Get Strategic Suggestions (Placeholder) ---
    # suggestion_engine = SuggestionEngine(config['data']['processed_dir'] + '/' + config['data']['train_csv'])
    # suggestions = suggestion_engine.generate_suggestions(predicted_attributes)
    suggestions = ["TODO: Implement suggestion engine."]

    # --- Format Final Output ---
    result = {
        "predicted_price": f"{predicted_price:.2f}",
        "predicted_attributes": predicted_attributes,
        "generated_listing": generated_text,
        "strategic_suggestions": suggestions
    }

    print(json.dumps(result, indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate e-commerce listing from a single product image.")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model checkpoint (.pth).")
    args = parser.parse_args()
    main(args)