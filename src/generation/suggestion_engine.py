import pandas as pd

class SuggestionEngine:
    """
    Analyzes the entire dataset to find top-performing attributes
    and provides suggestions based on a new product's predicted attributes.
    """
    def __init__(self, full_dataset_path: str, rating_col='Average Rating of Product', count_col='Count of Ratings', min_ratings_threshold=30):
        """
        Initializes the engine by pre-computing market insights.
        
        Args:
            full_dataset_path (str): Path to the complete (train+val+test) processed CSV.
            min_ratings_threshold (int): Minimum number of ratings for a product to be considered in analysis.
        """
        print("Initializing Suggestion Engine...")
        df = pd.read_csv(full_dataset_path)
        
        # Filter for popular/well-reviewed products to get reliable insights
        popular_df = df[df[count_col] > min_ratings_threshold].copy()
        
        self.insights = {}
        
        # --- Pre-compute insights for each category ('Type') ---
        for product_type in popular_df['Type'].unique():
            type_df = popular_df[popular_df['Type'] == product_type]
            if len(type_df) < 10: # Skip very small categories
                continue
            
            # Find the highest-rated color and neck style for this type
            top_color = type_df.groupby('colour')[rating_col].mean().idxmax()
            top_neck = type_df.groupby('Neck')[rating_col].mean().idxmax()
            
            self.insights[product_type] = {
                'top_rated_color': top_color,
                'top_rated_neck': top_neck
            }
        print("Suggestion Engine initialized successfully.")

    def generate_suggestions(self, predicted_attributes: dict) -> list:
        """
        Generates suggestions for a single product based on its predicted attributes.
        """
        suggestions = []
        product_type = predicted_attributes.get('Type')
        
        if not product_type or product_type not in self.insights:
            return ["Not enough data to generate suggestions for this product type."]
            
        # --- Suggestion for Color ---
        current_color = predicted_attributes.get('colour')
        top_color = self.insights[product_type]['top_rated_color']
        if current_color and current_color != top_color:
            suggestions.append(
                f"Consider offering this in '{top_color}'. It is the highest-rated color for '{product_type}' products."
            )
            
        # --- Suggestion for Neck Style ---
        current_neck = predicted_attributes.get('Neck')
        top_neck = self.insights[product_type]['top_rated_neck']
        if current_neck and current_neck != top_neck:
            suggestions.append(
                f"Consider a '{top_neck}' style. It is the highest-rated neck design for '{product_type}' products."
            )
            
        if not suggestions:
            suggestions.append("This product's design aligns well with top-performing attributes.")
            
        return suggestions