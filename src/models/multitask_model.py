import torch
import torch.nn as nn
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
                               (e.g., if 'color' can be 'red', 'blue', 'green', num_classes=3)
        """
        super().__init__()
        # A single linear layer is often sufficient for a classification head on top of a powerful backbone.
        self.layer = nn.Linear(embedding_dim, num_classes)
    def forward(self, x):
        # This outputs raw scores (logits). The loss function (like BCEWithLogitsLoss)
        # will handle the activation (like sigmoid) internally for better numerical stability.
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
                                      and values are lists/mappings of their possible values.
                                      This allows for dynamic creation of attribute heads.
        """
        super().__init__()
        
        # 1. SHARED VISION BACKBONE (Core of "Holistic Visual Representation")
        # We load a pre-trained Vision Transformer (ViT). Its job is to convert an image
        # into a rich numerical representation (embedding). By training all tasks on this
        # single embedding, we force the model to learn features useful for everything.
        self.vision_encoder = ViTModel.from_pretrained(config['model']['vision_model_name'])
        embedding_dim = self.vision_encoder.config.hidden_size

        # 2. TASK-SPECIFIC HEADS
        # The single image embedding will be fed into each of these heads.
        self.price_head = PriceHead(embedding_dim)
        
        # We use nn.ModuleDict to hold all the attribute heads. This is the correct
        # way to store a variable number of sub-modules in PyTorch, ensuring they
        # are properly registered (e.g., for moving to GPU, collecting parameters).
        self.attribute_heads = nn.ModuleDict()
        for attr_name, mapping in attribute_mappers.items():
            num_values = len(mapping)
            self.attribute_heads[attr_name] = AttributeHead(embedding_dim, num_classes=num_values)

        # 3. TEXT GENERATION MODEL
        # We load a pre-trained T5 model. During training, we use it to calculate the
        # text generation loss. During inference, we will use its .generate() method
        # in a hierarchical way (by feeding it the predicted attributes as a prompt).
        self.text_generator = T5ForConditionalGeneration.from_pretrained(config['model']['text_model_name'])

    def forward(self, pixel_values, input_ids=None, attention_mask=None, labels=None):
        """
        Defines the forward pass of the model. This is what happens during one step of training.
        Args:
            pixel_values: The processed image tensor from the dataloader.
            input_ids, attention_mask: Tokenized input text for the T5 model.
            labels: The target token IDs for the T5 model to learn from.
        """
        # Step 1: Get the holistic visual representation from the shared backbone.
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        # We use the pooler_output, which is a summary of the image's content,
        # typically derived from the special [CLS] token's final hidden state.
        image_embedding = vision_outputs.pooler_output # SHAPE: [batch_size, embedding_dim]

        # Step 2: Feed the shared embedding to the non-text heads.
        # Price prediction returns a tensor of shape [batch_size, 1]. .squeeze(-1)
        # removes the last dimension to make it [batch_size], which is easier to work with.
        price_pred = self.price_head(image_embedding).squeeze(-1)
        
        # Get predictions for each attribute by iterating through our ModuleDict.
        attribute_logits = {}
        for attr_name, head in self.attribute_heads.items():
            attribute_logits[attr_name] = head(image_embedding)

        # Step 3: Handle the text generation task (for calculating loss during training).
        text_loss = None
        # The 'labels' argument is only provided during training. The Hugging Face model
        # is designed to automatically calculate the cross-entropy loss when labels are present.
        if labels is not None:
            # NOTE: In this basic training forward pass, the text generator does NOT
            # see the image embedding or the predicted attributes directly. It only sees
            # the text inputs/labels. The "Hierarchical Generation" happens during the
            # inference/generation step, where we will construct a prompt for the generator
            # using the predicted attributes. The model learns a shared visual representation
            # because the gradients from all task losses (including text) will flow back
            # through their respective heads and update the shared vision_encoder.
            text_outputs = self.text_generator(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            text_loss = text_outputs.loss

        # We return a dictionary of all outputs. This is a clean way to pass multiple
        # distinct outputs to the loss function and training script.
        return {
            'price_pred': price_pred,
            'attribute_logits': attribute_logits,
            'text_loss': text_loss
        }