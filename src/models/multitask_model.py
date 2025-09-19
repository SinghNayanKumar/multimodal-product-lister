import torch
import torch.nn as nn
import numpy as np
# We import pre-trained models from the Hugging Face transformers library.
# ViTModel: The vision backbone for image feature extraction.
# T5ForConditionalGeneration: A powerful encoder-decoder model for text generation.
from transformers import ViTModel, T5ForConditionalGeneration

# --- Task-Specific Head for Price Prediction ---
# This is a small, dedicated sub-network for the regression task.
class PriceHead(nn.Module):
    def __init__(self, embedding_dim=768):
        """
        Initializes the regression head for price prediction.
        Args:
            embedding_dim (int): The size of the input feature vector from the vision backbone.
                                 For ViT-base, this is 768.
        """
        super().__init__()
        # We use a simple Multi-Layer Perceptron (MLP) for this task.
        self.layer = nn.Sequential(
            nn.Linear(embedding_dim, 128), # Reduce dimensionality
            nn.ReLU(),                     # Add non-linearity
            nn.Dropout(0.1),               # Regularization to prevent overfitting
            nn.Linear(128, 1)              # Output a single continuous value for the price
        )
    def forward(self, x):
        # x here will be the [batch_size, embedding_dim] image embedding.
        return self.layer(x)

# --- Task-Specific Head for Attribute Prediction ---
# This is a dedicated sub-network for the classification task.
class AttributeHead(nn.Module):
    def __init__(self, embedding_dim=768, num_classes=10):
        """
        Initializes a classification head for a single attribute type (e.g., color).
        Args:
            embedding_dim (int): The size of the input feature vector.
            num_classes (int): The number of possible values for this specific attribute.
        """
        super().__init__()
        self.layer = nn.Linear(embedding_dim, num_classes)
    def forward(self, x):
        # This outputs raw scores (logits). The loss function will handle the activation.
        return self.layer(x)

# --- The Main Multi-Task Learning Model ---
# This class orchestrates everything, tying the shared backbone to the task-specific heads.
class MultitaskModel(nn.Module):
    def __init__(self, config, attribute_mappers):
        """
        Initializes the complete multi-task model.
        Args:
            config (dict): A configuration dictionary with model names and other settings.
            attribute_mappers (dict): A dictionary where keys are attribute names (e.g., "neck")
                                      and values are mappings of their possible values.
        """
        super().__init__()
        
        # --- 1. SHARED VISION BACKBONE (Core of "Holistic Visual Representation") ---
        # We load a pre-trained Vision Transformer (ViT). Its job is to convert an image
        # into a rich numerical representation (embedding).
        self.vision_encoder = ViTModel.from_pretrained(config['model']['vision_model_name'])
        embedding_dim = self.vision_encoder.config.hidden_size

        # --- 2. TASK-SPECIFIC HEADS ---
        # The single image embedding will be fed into each of these heads.
        self.price_head = PriceHead(embedding_dim)
        
        # We use nn.ModuleDict to hold all the attribute heads. This is the correct
        # PyTorch way to store a variable number of sub-modules.
        self.attribute_heads = nn.ModuleDict()
        for attr_name, mapping in attribute_mappers.items():
            num_values = len(mapping)
            self.attribute_heads[attr_name] = AttributeHead(embedding_dim, num_classes=num_values)

        # --- 3. TEXT GENERATION MODEL ---
        # We load a pre-trained T5 model for our sequence-to-sequence text generation task.
        self.text_generator = T5ForConditionalGeneration.from_pretrained(config['model']['text_model_name'])

    def forward(self, pixel_values, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        Defines the forward pass of the model, used during TRAINING.
        Args:
            pixel_values: The processed image tensor from the dataloader.
            input_ids, attention_mask: Tokenized input text for the T5 model.
            labels: The target token IDs for the T5 model to learn from.
        """
        # Step 1: Get the holistic visual representation from the shared backbone.
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        image_embedding = vision_outputs.pooler_output # SHAPE: [batch_size, embedding_dim]

        # Step 2: Feed the shared embedding to the non-text heads.
        price_pred = self.price_head(image_embedding).squeeze(-1)
        
        attribute_logits = {}
        for attr_name, head in self.attribute_heads.items():
            attribute_logits[attr_name] = head(image_embedding)

        # Step 3: Handle the text generation task to calculate loss during training.
        text_loss = None
        if labels is not None:
            # ANNOTATION: The Hugging Face model conveniently calculates the cross-entropy loss
            # for text generation internally when 'labels' are provided.
            text_outputs = self.text_generator(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            text_loss = text_outputs.loss

        # ANNOTATION: We return a dictionary of all outputs. This is a clean way to pass multiple
        # distinct outputs to the composite loss function.
        return {
            'price_pred': price_pred,
            'attribute_logits': attribute_logits,
            'text_loss': text_loss
        }

    def predict(self, pixel_values, tokenizer, attribute_mappers, use_hierarchical_prompt=True):
        """
        Performs a full INFERENCE pass, implementing hierarchical generation and supporting ablation studies.

        Args:
            pixel_values (torch.Tensor): The preprocessed image tensor.
            tokenizer: The Hugging Face tokenizer for decoding.
            attribute_mappers (dict): The loaded attribute mappings for decoding IDs.
            use_hierarchical_prompt (bool): If True (default), uses the standard hierarchical prompt.
                                            If False, uses a generic prompt for the ablation study.
        """
        # Create inverse mappings for decoding integer IDs back to string labels.
        inverse_mappers = {
            attr: {i: label for label, i in mapping.items()} 
            for attr, mapping in attribute_mappers.items()
        }

        # ANNOTATION: Set model to evaluation mode and disable gradient calculations for efficiency.
        self.eval()
        with torch.no_grad():
            # --- Step 1: Get visual features and structured predictions ---
            vision_outputs = self.vision_encoder(pixel_values=pixel_values)
            image_embedding = vision_outputs.pooler_output

            # Get the raw log-price prediction from the model head.
            predicted_log_price = self.price_head(image_embedding).squeeze().item()

            # ANNOTATION: The dataloader uses `np.log1p()` to transform the price.
            # To convert the model's output back to the real dollar value, we must
            # apply the inverse function, `np.expm1()`. This is a critical step for
            # reporting correct MAE/RMSE and for the suggestion engine.
            predicted_price = np.expm1(predicted_log_price)

            predicted_attributes_decoded = {}
            for attr_name, head in self.attribute_heads.items():
                logits = head(image_embedding)
                pred_id = torch.argmax(logits, dim=-1).item()
                predicted_attributes_decoded[attr_name] = inverse_mappers[attr_name].get(pred_id, "Unknown")
            
            # --- Step 2: Build the prompt (with Ablation Logic) ---
            if use_hierarchical_prompt:
                # ANNOTATION: This is our main, proposed method. We construct a prompt from the model's
                # own structured predictions, making the text factually grounded.
                prompt_parts = [f"{attr}: {value}" for attr, value in predicted_attributes_decoded.items() if value != 'Unknown']
                prompt_text = "generate a title and description for a product with these features: " + ", ".join(prompt_parts)
            else:
                # ANNOTATION: This is for the 'Ours-Ablate-NoHier' experiment. The model gets a generic
                # prompt and must rely solely on the visual features passed to the decoder.
                prompt_text = "generate a title and description for the following product:"
            
            input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(pixel_values.device)

            # --- Step 3: Generate text conditioned on BOTH text prompt and image ---
            # ANNOTATION: This is the critical step for Hierarchical Generation. We pass the `encoder_outputs`
            # from the vision model directly into the text model's `generate` method. This allows the T5 decoder
            # to use cross-attention to "look at" the image while generating text.
            generated_ids = self.text_generator.generate(
                input_ids=input_ids,
                encoder_outputs=vision_outputs, # CRITICAL: This conditions the generation on the image!
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # --- Step 4: Return all predictions in a clean dictionary ---
        return {
            "predicted_price": predicted_price,
            "predicted_attributes": predicted_attributes_decoded,
            "generated_text": generated_text
        }