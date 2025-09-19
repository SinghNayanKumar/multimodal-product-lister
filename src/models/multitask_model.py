import torch
import torch.nn as nn
import numpy as np
from transformers import ViTModel, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

class PriceHead(nn.Module):
    """A simple MLP to predict a single continuous value (price) from the shared image embedding."""
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
    """A simple linear layer to predict class logits for a specific attribute from the shared embedding."""
    def __init__(self, embedding_dim=768, num_classes=10):
        super().__init__()
        self.layer = nn.Linear(embedding_dim, num_classes)
    def forward(self, x):
        return self.layer(x)

class MultitaskModel(nn.Module):
    def __init__(self, config, attribute_mappers):
        super().__init__()
        self.vision_encoder = ViTModel.from_pretrained(config['model']['vision_model_name'])
        self.text_generator = T5ForConditionalGeneration.from_pretrained(config['model']['text_model_name'])
        vision_embedding_dim = self.vision_encoder.config.hidden_size
        text_embedding_dim = self.text_generator.config.d_model
        self.vit_to_t5 = nn.Linear(vision_embedding_dim, text_embedding_dim)
        self.price_head = PriceHead(vision_embedding_dim)
        self.attribute_heads = nn.ModuleDict()
        for attr_name, mapping in attribute_mappers.items():
            num_values = len(mapping)
            self.attribute_heads[attr_name] = AttributeHead(vision_embedding_dim, num_classes=num_values)

    def forward(self, pixel_values, input_ids=None, attention_mask=None, labels=None, **kwargs):
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        image_embedding = vision_outputs.pooler_output
        price_pred = self.price_head(image_embedding).squeeze(-1)
        attribute_logits = {attr_name: head(image_embedding) for attr_name, head in self.attribute_heads.items()}
        text_loss = None
        if labels is not None:
            # Use only the CLS token for T5 encoder input
            cls_token = vision_outputs.last_hidden_state[:, 0:1, :]  # [B, 1, 768]
            projected_vision_features = self.vit_to_t5(cls_token)    # [B, 1, d_model]
            encoder_outputs_for_t5 = BaseModelOutput(last_hidden_state=projected_vision_features)
            text_outputs = self.text_generator(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                encoder_outputs=encoder_outputs_for_t5
            )
            text_loss = text_outputs.loss
        return {
            'price_pred': price_pred,
            'attribute_logits': attribute_logits,
            'text_loss': text_loss
        }

    @torch.no_grad()
    def predict(self, pixel_values, tokenizer, attribute_mappers, use_hierarchical_prompt=True):
        self.eval()
        inverse_mappers = {
            attr: {i: label for label, i in mapping.items()} 
            for attr, mapping in attribute_mappers.items()
        }
        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        
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
        
        prompts = []
        for i in range(batch_size):
            if use_hierarchical_prompt:
                prompt_parts = [f"{attr}: {value}" for attr, value in all_predicted_attributes[i].items() if value != 'Unknown']
                prompt_text = "generate a title and description for a product with these features: " + ", ".join(prompt_parts)
            else:
                prompt_text = "generate a title and description for the following product:"
            prompts.append(prompt_text)
        
        tokenized_prompts = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # Use only the CLS token for T5 encoder input
        cls_token = vision_outputs.last_hidden_state[:, 0:1, :]  # [B, 1, 768]
        projected_vision_features = self.vit_to_t5(cls_token)    # [B, 1, d_model]
        encoder_outputs_for_t5 = BaseModelOutput(last_hidden_state=projected_vision_features)
        
        generated_ids = self.text_generator.generate(
            input_ids=tokenized_prompts.input_ids,
            attention_mask=tokenized_prompts.attention_mask,
            encoder_outputs=encoder_outputs_for_t5,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
        all_generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        results = []
        for i in range(batch_size):
            results.append({
                "predicted_price": predicted_prices[i],
                "predicted_attributes": all_predicted_attributes[i],
                "generated_text": all_generated_text[i]
            })
        return results