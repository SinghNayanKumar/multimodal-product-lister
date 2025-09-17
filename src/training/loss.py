import torch
import torch.nn as nn

class CompositeLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
        self.price_loss_fn = nn.MSELoss()
        self.attribute_loss_fn = nn.CrossEntropyLoss()

    def forward(self, model_outputs, batch):
        # --- Price Loss ---
        price_pred = model_outputs['price_pred']
        price_target = batch['price_target']
        loss_price = self.price_loss_fn(price_pred, price_target)

        # --- Attribute Losses ---
        attr_logits = model_outputs['attribute_logits']
        total_attribute_loss = 0
        num_attrs = 0
        for attr_name, logits in attr_logits.items():
            target = batch[f"{attr_name}_target"]
            total_attribute_loss += self.attribute_loss_fn(logits, target)
            num_attrs += 1
        
        loss_attributes = total_attribute_loss / num_attrs if num_attrs > 0 else 0

        # --- Text Loss ---
        loss_text = model_outputs['text_loss']

        # --- Weighted Combination ---
        total_loss = (self.weights['price'] * loss_price +
                      self.weights['attributes'] * loss_attributes +
                      self.weights['text'] * loss_text)
        
        return {
            "total_loss": total_loss,
            "price_loss": loss_price.item(),
            "attribute_loss": loss_attributes.item() if isinstance(loss_attributes, torch.Tensor) else loss_attributes,
            "text_loss": loss_text.item()
        }