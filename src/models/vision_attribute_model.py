import torch
import torch.nn as nn
from transformers import ViTModel

class VisionAttributeModel(nn.Module):
    """
    A vision model focused solely on predicting structured attributes from an image.
    This serves as Stage 1 of the two-stage hybrid baseline.
    """
    def __init__(self, config, mappings):
        super().__init__()
        self.config = config
        self.mappings = mappings

        # --- Vision Backbone ---
        vision_model_name = config['model']['vision_backbone']
        self.vision_backbone = ViTModel.from_pretrained(vision_model_name)
        
        # Freeze backbone if configured to do so
        if config['model'].get('freeze_vision_backbone', False):
            for param in self.vision_backbone.parameters():
                param.requires_grad = False

        # --- Attribute Prediction Heads ---
        # A dictionary to hold a separate classification head for each attribute
        self.attribute_heads = nn.ModuleDict()
        hidden_size = self.vision_backbone.config.hidden_size

        for attr_name, attr_map in mappings['attributes'].items():
            num_classes = len(attr_map)
            self.attribute_heads[attr_name] = nn.Linear(hidden_size, num_classes)

    def forward(self, image_tensor, **kwargs):
        """
        Forward pass for attribute prediction.
        Args:
            image_tensor (torch.Tensor): The input image tensor.
        
        Returns:
            dict: A dictionary containing the logits for each attribute.
                  e.g., {'attribute_logits': {'Type': tensor, 'colour': tensor}}
        """
        # Get image features from the vision backbone
        vision_outputs = self.vision_backbone(pixel_values=image_tensor)
        # Use the [CLS] token's representation for classification
        image_features = vision_outputs.pooler_output

        # Pass the features through each attribute head
        attribute_logits = {}
        for attr_name, head in self.attribute_heads.items():
            attribute_logits[attr_name] = head(image_features)

        return {'attribute_logits': attribute_logits}