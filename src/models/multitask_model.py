import torch
import torch.nn as nn
from transformers import ViTModel, T5ForConditionalGeneration

class PriceHead(nn.Module):
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
    def __init__(self, embedding_dim=768, num_classes=10):
        super().__init__()
        self.layer = nn.Linear(embedding_dim, num_classes)
    def forward(self, x):
        return self.layer(x)

class MultitaskModel(nn.Module):
    def __init__(self, config, attribute_mappers):
        super().__init__()
        self.vision_encoder = ViTModel.from_pretrained(config['model']['vision_model_name'])
        embedding_dim = self.vision_encoder.config.hidden_size

        self.price_head = PriceHead(embedding_dim)
        
        # Create a dictionary of attribute heads dynamically
        self.attribute_heads = nn.ModuleDict()
        for attr_name, mapping in attribute_mappers.items():
            self.attribute_heads[attr_name] = AttributeHead(embedding_dim, num_classes=len(mapping))

        self.text_generator = T5ForConditionalGeneration.from_pretrained(config['model']['text_model_name'])

    def forward(self, pixel_values, input_ids=None, attention_mask=None, labels=None):
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        image_embedding = vision_outputs.pooler_output

        price_pred = self.price_head(image_embedding).squeeze(-1)
        
        attribute_logits = {}
        for attr_name, head in self.attribute_heads.items():
            attribute_logits[attr_name] = head(image_embedding)

        # Calculate text loss only during training when labels are provided
        text_loss = None
        if labels is not None:
            text_outputs = self.text_generator(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            text_loss = text_outputs.loss

        return {
            'price_pred': price_pred,
            'attribute_logits': attribute_logits,
            'text_loss': text_loss
        }