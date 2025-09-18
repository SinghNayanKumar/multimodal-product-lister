import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import warnings

# Suppress pandas warnings about chained assignment and future warnings for cleaner output
warnings.simplefilter(action='ignore', category=pd.core.common.is_categorical_dtype)
warnings.simplefilter(action='ignore', category=FutureWarning)


class SuggestionEngine:
    """
    Uses predictive models to run "what-if" scenarios on product attributes
    to generate quantitative, actionable business suggestions.

    This engine trains two Gradient Boosting models during initialization:
    1. A model to predict a product's average rating based on its attributes.
    2. A model to predict a product's price based on its attributes.

    It then uses these models to simulate the impact of changing an attribute
    (e.g., color, neck style) and generates concrete suggestions if the
    change is predicted to be beneficial.
    """
    def __init__(self,
                 full_dataset_path: str,
                 attribute_cols: list,
                 price_col: str,
                 rating_col: str,
                 category_col: str,
                 min_category_size: int = 50):
        """
        Initializes the engine by loading data and training performance models.

        Args:
            full_dataset_path (str): Path to the complete CSV dataset.
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

        # --- 1. Load and Prepare Data ---
        df = self._load_and_clean_data(full_dataset_path)

        # --- 2. Pre-compute Market Leaders for Simulations ---
        self.market_leaders = self._compute_market_leaders(df, min_category_size)

        # --- 3. Train Predictive Models ---
        self.price_model = self._train_model(df, self.price_col)
        self.rating_model = self._train_model(df, self.rating_col)
        
        print("Suggestion Engine initialized and models are trained.")

    def _load_and_clean_data(self, path: str) -> pd.DataFrame:
        """Loads data and performs basic cleaning."""
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at path: {path}")
        
        required_cols = self.attribute_cols + [self.price_col, self.rating_col]
        df = df[required_cols].dropna().copy()
        # Convert all attribute columns to 'category' dtype for efficiency and correctness
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
        
    def _train_model(self, df: pd.DataFrame, target_col: str) -> Pipeline:
        """Defines and trains a regression model pipeline."""
        X = df[self.attribute_cols]
        y = df[target_col]

        preprocessor = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.attribute_cols)])

        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42, enable_categorical=True))
        ])

        print(f"Training model for target: '{target_col}'...")
        model_pipeline.fit(X, y)
        return model_pipeline

    def generate_suggestions(self, predicted_attributes: dict) -> list:
        """
        Generates quantitative suggestions for a product by running simulations.
        """
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
        
        return sorted(list(set(suggestions))) # Return unique, sorted suggestions

### Example Usage ###
if __name__ == '__main__':
    # -------------------------------------------------------
    DATASET_PATH = 'data/processed/train.csv'
    ATTRIBUTE_COLS = ['Type', 'colour', 'Neck', 'Sleeve Length', 'Fit', 'Hemline']
    PRICE_COL = 'price'
    RATING_COL = 'avg_rating'
    CATEGORY_COL = 'Type'
    # -------------------------------------------------------

    try:
        # 1. Initialize the engine (this will train the models)
        engine = SuggestionEngine(
            full_dataset_path=DATASET_PATH,
            attribute_cols=ATTRIBUTE_COLS,
            price_col=PRICE_COL,
            rating_col=RATING_COL,
            category_col=CATEGORY_COL
        )

        # 2. Define a new product (this would come from your main vision model's output)
        new_product_attributes = {
            'Type': 'Basic Jumpsuit', 
            'colour': 'Blue',
            'Neck': 'Round Neck',
            'Sleeve Length': 'Short Sleeve',
            'Fit': 'Regular',
            'Hemline': 'Straight'
        }

        # 3. Generate suggestions
        print(f"\n--- Generating suggestions for: {new_product_attributes} ---")
        product_suggestions = engine.generate_suggestions(new_product_attributes)
        
        for suggestion in product_suggestions:
            print(f"-> {suggestion}")

    except FileNotFoundError as e:
        print(e)
        print("\nPlease ensure the dataset path is correct and the file exists.")
    except KeyError as e:
        print(f"Caught a KeyError: {e}. This likely means a column name is incorrect or missing from the CSV.")
        print("Please double-check your column names provided.")