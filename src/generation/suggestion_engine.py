import pandas as pd
import joblib # Import joblib for loading models
import os
import warnings

# Suppress pandas warnings for cleaner output
warnings.simplefilter(action='ignore', category=pd.core.common.is_categorical_dtype)
warnings.simplefilter(action='ignore', category=FutureWarning)

class SuggestionEngine:
    """
    Uses predictive models to run "what-if" scenarios on product attributes
    to generate quantitative, actionable business suggestions.

    This engine is initialized with pre-trained models for predicting rating and price,
    making it fast and efficient for inference.
    """
    def __init__(self,
                 market_data_path: str,
                 model_dir: str,
                 attribute_cols: list,
                 price_col: str,
                 rating_col: str,
                 category_col: str,
                 min_category_size: int = 50):
        """
        Initializes the engine by loading market data and pre-trained performance models.

        Args:
            market_data_path (str): Path to the CSV dataset for market analysis (e.g., train.csv).
            model_dir (str): Path to the directory containing pre-trained 'price_model.joblib' and 'rating_model.joblib'.
            attribute_cols (list): List of column names for product attributes.
            price_col (str): The name of the price column.
            rating_col (str): The name of the average rating column.
            category_col (str): The primary attribute for grouping (e.g., 'Type').
            min_category_size (int): Minimum number of items in a category to be considered.
        """
        print("Initializing Prescriptive Suggestion Engine...")
        self.attribute_cols = attribute_cols
        self.price_col = price_col
        self.rating_col = rating_col
        self.category_col = category_col
        
        if self.category_col not in self.attribute_cols:
            raise ValueError(f"'{self.category_col}' must be included in attribute_cols.")

        # --- 1. Load Data for Market Analysis ---
        df = self._load_and_clean_data(market_data_path)

        # --- 2. Pre-compute Market Leaders for Simulations ---
        self.market_leaders = self._compute_market_leaders(df, min_category_size)

        # --- 3. Load Pre-Trained Predictive Models ---
        # ANNOTATION: Instead of training, we now load the models from disk. This is essential for fast inference.
        price_model_path = os.path.join(model_dir, 'price_model.joblib')
        rating_model_path = os.path.join(model_dir, 'rating_model.joblib')
        
        try:
            self.price_model = joblib.load(price_model_path)
            self.rating_model = joblib.load(rating_model_path)
        except FileNotFoundError as e:
            # ANNOTATION: Provide a helpful error message if the models are missing.
            raise FileNotFoundError(
                f"A required model was not found. Please run `src/training/train_suggestion_models.py` first. "
                f"Missing file: {e.filename}"
            ) from e
        
        print("Suggestion Engine initialized with pre-trained models.")

    def _load_and_clean_data(self, path: str) -> pd.DataFrame:
        """Loads data and performs basic cleaning."""
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Market data not found at path: {path}")
        
        required_cols = self.attribute_cols + [self.price_col, self.rating_col]
        df = df[required_cols].dropna().copy()
        for col in self.attribute_cols:
            df[col] = df[col].astype('category')
        return df

    def _compute_market_leaders(self, df: pd.DataFrame, min_size: int) -> dict:
        """Finds the top 3 performing values for each attribute within each category."""
        leaders = {}
        for category_name, group in df.groupby(self.category_col):
            if len(group) < min_size:
                continue
            
            leaders[str(category_name)] = {}
            for attr in self.attribute_cols:
                if attr != self.category_col:
                    top_values = group.groupby(attr)[self.rating_col].mean().nlargest(3).index.tolist()
                    leaders[str(category_name)][attr] = top_values
        return leaders

    def generate_suggestions(self, predicted_attributes: dict) -> list:
        # ANNOTATION: This method requires NO changes. It's perfectly decoupled from how the
        # models were obtained, demonstrating good design.
        category_value = predicted_attributes.get(self.category_col)
        
        if not category_value or category_value not in self.market_leaders:
            return [f"Not enough market data to generate reliable suggestions for '{category_value}'."]

        suggestions = []
        current_product_df = pd.DataFrame([predicted_attributes]).astype({col: 'category' for col in self.attribute_cols})

        try:
            baseline_rating = self.rating_model.predict(current_product_df)[0]
            baseline_price = self.price_model.predict(current_product_df)[0]
        except Exception as e:
            return [f"Could not generate prediction for input attributes. Error: {e}"]

        for attr_to_change, top_values in self.market_leaders[category_value].items():
            current_value = predicted_attributes.get(attr_to_change)
            
            for top_value in top_values:
                if current_value != top_value:
                    hypothetical_attrs = predicted_attributes.copy()
                    hypothetical_attrs[attr_to_change] = top_value
                    hypothetical_df = pd.DataFrame([hypothetical_attrs]).astype({col: 'category' for col in self.attribute_cols})
                    
                    new_rating = self.rating_model.predict(hypothetical_df)[0]
                    new_price = self.price_model.predict(hypothetical_df)[0]
                    
                    rating_diff = new_rating - baseline_rating
                    price_diff = new_price - baseline_price
                    
                    if rating_diff > 0.1:
                        suggestions.append(
                            f"Consider a '{top_value}' {attr_to_change}. Our model estimates it could improve the rating by ~{rating_diff:.2f} points."
                        )
                    if price_diff > (baseline_price * 0.05):
                         suggestions.append(
                            f"The market may support a higher price for a '{top_value}' {attr_to_change}. Estimated price potential: ~${new_price:.2f} (an increase of ${price_diff:.2f})."
                        )

        if not suggestions:
            return ["This product's current attributes are already well-optimized against market leaders."]
        
        return sorted(list(set(suggestions)))