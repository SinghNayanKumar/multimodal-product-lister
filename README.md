# multimodal-product-lister

This project implements a multimodal, multitask deep learning system that generates a complete e-commerce product listing from a single image.

## Features
- **Multimodal Input:** Takes a product image as primary input.
- **Multitask Output:**
  - Predicts structured attributes (e.g., Colour, Neck, Sleeve Length).
  - Suggests a listing price.
  - Generates a catchy title and marketing description.
  eg. 
attributes: { "category": "T-Shirts", "color": "blue", "style": "v-neck" }
price: $24.99
title/description
- **Strategic Suggestions:** Provides data-driven advice to improve the product's marketability based on a backend analysis of sales/rating data. 

eg. 
"strategic_suggestions": [
  {
    "attribute": "color",
    "current": "blue",
    "suggestion": "black",
    "justification": "Black is the top-selling color in the T-Shirts category, appearing 35% more frequently than the next most popular color."
  },
  {
    "attribute": "style",
    "current": "v-neck",
    "suggestion": "round-neck",
    "justification": "Round-neck styles are 50% more common in top-rated T-shirt listings."
  }
]

## Project Structure
The project is structured for scalability and reproducibility.
- `data/`: Holds raw and processed datasets.
- `scripts/`: Contains data processing scripts.
- `src/`: All source code for the model, training, and evaluation.
- `experiments/`: YAML configuration files for different training runs.
- `notebooks/`: Jupyter notebooks for exploration and analysis.
- `predict.py`: A script to run inference on a single image.

## Usage
1. **Setup:**
   ```bash
   pip install -r requirements.txt

## Model Architecture - Product Classifier
                                  +-----------------------+
Input Image --------------------->| Vision Transformer    |
                                  |      (e.g., ViT)      |
                                  +-----------+-----------+
                                              |
                                     (Visual Embeddings)
                                              |
                        +---------------------+---------------------+
                        |                                           |
            +-----------v-----------+                   +-----------v-----------+
            |    Attribute Decoder  |                   |  Price Prediction Head|
            | (Classification)      |                   |    (Regression)       |
            +-----------+-----------+                   +-----------+-----------+
                        |                                           |
           { "color": "blue", ... }                             $39.99
                        |                                           |
                        +---------------------+---------------------+
                                              |
                                (Embeddings + Predicted Attributes)
                                              |
                                  +-----------v-----------+
                                  |   Language Model      |
                                  |      (e.g., T5)       |
                                  +-----------+-----------+
                                              |
                          +-------------------+-------------------+
                          |                                       |
                  (Generated Title)                       (Generated Description)
                
