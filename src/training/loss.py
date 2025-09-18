import torch
import torch.nn as nn

class CompositeLoss(nn.Module):
    def __init__(self, weights):
        """
        Initializes the composite loss module.
        Args:
            weights (dict): A dictionary containing the weights for each loss component, 
                            e.g., {'price': 0.1, 'attributes': 0.4, 'text': 0.5}. 
                            These are crucial hyperparameters for balancing the tasks.
        """
        super().__init__()
        self.weights = weights
        
        # ANNOTATION: Mean Squared Error (MSE) is a standard choice for regression tasks like price prediction.
        # It calculates the average squared difference between the predicted and actual values.
        self.price_loss_fn = nn.MSELoss()
        
        # ANNOTATION: CrossEntropyLoss is used for multi-class classification. This implies that for each attribute
        # (e.g., 'neck_style'), the model predicts one class from a set of mutually exclusive options 
        # (e.g., 'V-Neck', 'Crew-Neck', 'Turtleneck').
        # If our attributes were multi-label (e.g., 'material' could be 'Cotton' AND 'Polyester'),
        # we would have used nn.BCEWithLogitsLoss instead.
        self.attribute_loss_fn = nn.CrossEntropyLoss()

    def forward(self, model_outputs, batch):
        """
        Calculates the combined loss.
        Args:
            model_outputs (dict): A dictionary of tensors coming from the MultitaskModel.
                                  Expected keys: 'price_pred', 'attribute_logits', 'text_loss'.
            batch (dict): The dictionary of ground-truth data from the dataloader.
        """
        # --- Price Loss ---
        # Slicing the model's output and the batch to get the relevant tensors for price.
        price_pred = model_outputs['price_pred']
        price_target = batch['price_target']
        loss_price = self.price_loss_fn(price_pred, price_target)

        # --- Attribute Losses ---
        # The model outputs logits for each attribute in a dictionary. 
        attr_logits = model_outputs['attribute_logits']
        total_attribute_loss = 0
        num_attrs = 0
        
        # We iterate through each attribute's prediction, calculate its loss against
        # the corresponding target, and accumulate the result.
        for attr_name, logits in attr_logits.items():
            target = batch[f"{attr_name}_target"]
            total_attribute_loss += self.attribute_loss_fn(logits, target)
            num_attrs += 1
        
        # We average the loss across all attributes to prevent attributes with more classes
        # from dominating the loss landscape.
        loss_attributes = total_attribute_loss / num_attrs if num_attrs > 0 else 0

        # --- Text Loss ---
        # ANNOTATION: A key design choice. The text generation loss (Cross-Entropy on token predictions)
        # is calculated *inside* the model. This is a common and convenient pattern when using models
        # from libraries like Hugging Face's Transformers, as they can return the loss directly
        # when labels are provided. This script just retrieves the pre-calculated value.
        loss_text = model_outputs['text_loss']

        # --- Weighted Combination ---
        # ANNOTATION: This is the core of MTL. The individual losses are multiplied by their respective
        # weights and summed up. This allows you to control the influence of each task. For example,
        # if text generation is the most important task, you might give 'text' the highest weight.
        total_loss = (self.weights['price'] * loss_price +
                      self.weights['attributes'] * loss_attributes +
                      self.weights['text'] * loss_text)
        
        return {
            "total_loss": total_loss,
            "price_loss": loss_price.item(),
            "attribute_loss": loss_attributes.item() if isinstance(loss_attributes, torch.Tensor) else loss_attributes,
            "text_loss": loss_text.item()
        }