import torch
import torch.nn as nn
import numpy as np
from transformers import ViTModel, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

class PriceHead(nn.Module):
    """
    A simple MLP (Multi-Layer Perceptron) head for the price prediction task.
    It takes the shared image embedding and regresses a single continuous value (the price).
    """
    def __init__(self, embedding_dim=768):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.layer(x)

class AttributeHead(nn.Module):
    """
    A simple linear classification head for a single attribute (e.g., 'Color', 'Type').
    It takes the shared image embedding and predicts logits for each possible class of that attribute.
    """
    def __init__(self, embedding_dim=768, num_classes=10):
        super().__init__()
        self.layer = nn.Linear(embedding_dim, num_classes)
    def forward(self, x):
        return self.layer(x)

class MultitaskModel(nn.Module):
    """
    The main multitask model. It combines a vision encoder (ViT) with a text generator (T5)
    and includes separate heads for predicting price and structured attributes.
    """
    def __init__(self, config, attribute_mappers):
        super().__init__()
        # 1. Vision Backbone: A pre-trained Vision Transformer to extract features from images.
        self.vision_encoder = ViTModel.from_pretrained(config['model']['vision_model_name'])
        
        # 2. Text Generator: A pre-trained T5 model for conditional text generation.
        self.text_generator = T5ForConditionalGeneration.from_pretrained(config['model']['text_model_name'])
        
        # 3. Projection Layer: A linear layer to bridge the dimensional gap between the
        #    vision encoder's output and the text generator's expected input dimension.
        vision_embedding_dim = self.vision_encoder.config.hidden_size      # e.g., 768 for ViT-base
        text_embedding_dim = self.text_generator.config.d_model           # e.g., 512 for T5-small
        self.vit_to_t5 = nn.Linear(vision_embedding_dim, text_embedding_dim)
        
        # 4. Task-Specific Heads
        self.price_head = PriceHead(vision_embedding_dim)
        self.attribute_heads = nn.ModuleDict()
        for attr_name, mapping in attribute_mappers.items():
            num_values = len(mapping)
            self.attribute_heads[attr_name] = AttributeHead(vision_embedding_dim, num_classes=num_values)

    def forward(self, pixel_values, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        The forward pass for training the model.
        """
        # --- Step A: Process the image to get shared embeddings ---
        # This part remains the same. The image embedding is used for all tasks.
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        image_embedding = vision_outputs.pooler_output  # [Batch_Size, 768] - Used for price/attribute heads.
        
        # --- Step B: Calculate outputs for non-text tasks ---
        price_pred = self.price_head(image_embedding).squeeze(-1)
        attribute_logits = {attr_name: head(image_embedding) for attr_name, head in self.attribute_heads.items()}
        
        # --- Step C: Calculate loss for the text generation task (with the critical fix) ---
        text_loss = None
        if labels is not None:
            # --- START: MODIFIED LOGIC FOR MULTIMODAL ENCODING ---
            
            # 1. Get vision features from ViT's final hidden state (using the [CLS] token)
            #    and project them to match T5's embedding dimension.
            #    Shape: [Batch_Size, 1, 768]
            cls_token = vision_outputs.last_hidden_state[:, 0:1, :]
            #    Shape: [Batch_Size, 1, 512 (d_model)]
            projected_vision_features = self.vit_to_t5(cls_token)

            # 2. Get text prompt features by passing the `input_ids` through the T5 encoder.
            #    THIS IS THE CRITICAL STEP THAT WAS MISSING. We are now actively using the prompt.
            #    Shape: [Batch_Size, Prompt_Length, 512 (d_model)]
            prompt_encoder_outputs = self.text_generator.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            prompt_features = prompt_encoder_outputs.last_hidden_state
            
            # 3. Concatenate the vision and text features along the sequence dimension.
            #    This creates a single, unified "multimodal sentence" that the T5 decoder will attend to.
            #    Think of it as: [Image_Token, Prompt_Token_1, Prompt_Token_2, ...]
            #    Shape: [Batch_Size, 1 + Prompt_Length, 512 (d_model)]
            combined_features = torch.cat([projected_vision_features, prompt_features], dim=1)
            
            # 4. Create a corresponding attention mask for the combined features.
            #    The vision token is always present, so its mask is 1. We then append the prompt's mask.
            vision_attention_mask = torch.ones(projected_vision_features.shape[:2], device=projected_vision_features.device)
            combined_attention_mask = torch.cat([vision_attention_mask, attention_mask], dim=1)

            # 5. Wrap the combined features in the expected format for the T5 decoder.
            encoder_outputs_for_t5 = BaseModelOutput(last_hidden_state=combined_features)
            
            # 6. Pass the combined multimodal context to the T5 decoder to calculate the loss.
            #    The `attention_mask` here is for the decoder's cross-attention, telling it
            #    which parts of our combined encoder output to pay attention to.
            text_outputs = self.text_generator(
                labels=labels,
                encoder_outputs=encoder_outputs_for_t5,
                attention_mask=combined_attention_mask # Pass the combined mask here
            )
            text_loss = text_outputs.loss
            # --- END: MODIFIED LOGIC ---
            
        return {
            'price_pred': price_pred,
            'attribute_logits': attribute_logits,
            'text_loss': text_loss
        }

    @torch.no_grad()
    def predict(self, pixel_values, tokenizer, attribute_mappers, use_hierarchical_prompt=True):
        """
        The prediction/inference method. Generates text and predicts attributes/price for a batch of images.
        """
        self.eval()
        inverse_mappers = {
            attr: {i: label for label, i in mapping.items()} 
            for attr, mapping in attribute_mappers.items()
        }
        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        
        # --- Step A: Get image features and predict non-text tasks ---
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        image_embedding = vision_outputs.pooler_output
        predicted_log_price_tensor = self.price_head(image_embedding).squeeze(-1)
        predicted_prices = np.expm1(predicted_log_price_tensor.cpu().numpy())
        
        all_predicted_attributes = [{} for _ in range(batch_size)]
        for attr_name, head in self.attribute_heads.items():
            logits = head(image_embedding)
            pred_ids = torch.argmax(logits, dim=-1)
            for i in range(batch_size):
                all_predicted_attributes[i][attr_name] = inverse_mappers[attr_name].get(pred_ids[i].item(), "Unknown")
        
        # --- Step B: Create the appropriate text prompt ---
        prompts = []
        for i in range(batch_size):
            if use_hierarchical_prompt:
                prompt_parts = [f"{attr}: {value}" for attr, value in all_predicted_attributes[i].items() if value != 'Unknown']
                prompt_text = "generate a title and description for a product with these features: " + ", ".join(prompt_parts)
            else:
                prompt_text = "generate a title and description for the following product:"
            prompts.append(prompt_text)
        
        tokenized_prompts = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # --- Step C: Generate text using the fixed multimodal approach ---
        # The logic here mirrors the `forward` method to ensure consistency between training and inference.
        
        # --- START: MODIFIED LOGIC FOR MULTIMODAL GENERATION ---
        # 1. Get and project vision features.
        cls_token = vision_outputs.last_hidden_state[:, 0:1, :]
        projected_vision_features = self.vit_to_t5(cls_token)

        # 2. Encode the text prompt.
        prompt_encoder_outputs = self.text_generator.encoder(
            input_ids=tokenized_prompts.input_ids,
            attention_mask=tokenized_prompts.attention_mask
        )
        prompt_features = prompt_encoder_outputs.last_hidden_state
        
        # 3. Concatenate to create the multimodal context.
        combined_features = torch.cat([projected_vision_features, prompt_features], dim=1)
        
        # 4. Create the combined attention mask.
        vision_attention_mask = torch.ones(projected_vision_features.shape[:2], device=projected_vision_features.device)
        combined_attention_mask = torch.cat([vision_attention_mask, tokenized_prompts.attention_mask], dim=1)

        # 5. Wrap the combined features for the generator.
        encoder_outputs_for_t5 = BaseModelOutput(last_hidden_state=combined_features)
 