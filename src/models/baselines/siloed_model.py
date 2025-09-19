import torch
import torch.nn as nn
from transformers import ViTModel

# ANNOTATION: These head modules are identical to the ones in your MultitaskModel.
# Reusing them ensures that any performance difference comes from the training strategy (MTL vs. Siloed),
# not from a different head architecture.
class PriceHead(nn.Module):
    def __init__(self, embedding_dim=768):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(embedding_dim, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.layer(x)

class AttributeHead(nn.Module):
    def __init__(self, embedding_dim=768, num_classes=10):
        super().__init__()
        self.layer = nn.Linear(embedding_dim, num_classes)
    def forward(self, x):
        return self.layer(x)

class SiloedModel(nn.Module):
    """
    A single-task vision model for either attribute prediction OR price prediction.
    This model is the core component for the 'Baseline-Siloed' and 'Baseline-Hybrid' experiments.
    Its key feature is the 'task' parameter, which configures it for one specific job,
    preventing any multi-task learning.
    """
    def __init__(self, config, attribute_mappers=None):
        super().__init__()
        
        # ANNOTATION: The task is specified in the config file (e.g., 'attributes' or 'price').
        # This allows us to use the same class to create different specialized models.
        self.task = config['model']['task']
        
        # ANNOTATION: A standard pre-trained vision backbone.
        self.vision_encoder = ViTModel.from_pretrained(config['model']['vision_model_name'])
        embedding_dim = self.vision_encoder.config.hidden_size

        # ANNOTATION: Based on the task, we attach the appropriate head.
        if self.task == 'price':
            self.head = PriceHead(embedding_dim)
            print("Initialized SiloedModel for PRICE prediction.")
        elif self.task == 'attributes':
            if attribute_mappers is None:
                raise ValueError("`attribute_mappers` must be provided for 'attributes' task.")
            # ANNOTATION: We use ModuleDict, just like in the MTL model, to handle all attribute heads.
            self.heads = nn.ModuleDict({
                attr_name: AttributeHead(embedding_dim, len(mapping))
                for attr_name, mapping in attribute_mappers.items()
            })
            print("Initialized SiloedModel for ATTRIBUTE prediction.")
        else:
            raise ValueError(f"Unknown task for SiloedModel: {self.task}")

    def forward(self, pixel_values, **kwargs):
        """
        A simplified forward pass that only computes outputs for its designated task.
        """
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        image_embedding = vision_outputs.pooler_output

        if self.task == 'price':
            price_pred = self.head(image_embedding).squeeze(-1)
            return {'price_pred': price_pred}
        
        elif self.task == 'attributes':
            attribute_logits = {
                attr_name: head(image_embedding)
                for attr_name, head in self.heads.items()
            }
            return {'attribute_logits': attribute_logits}

