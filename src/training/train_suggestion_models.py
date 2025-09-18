import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib  # Library for saving and loading Python objects
import os
import argparse
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def train_suggestion_model(df: pd.DataFrame, attribute_cols: list, target_col: str) -> Pipeline:
    """
    Defines and trains a regression model pipeline for a given target.

    Args:
        df (pd.DataFrame): The training dataframe.
        attribute_cols (list): List of feature column names.
        target_col (str): The name of the target column to predict.

    Returns:
        Pipeline: The trained scikit-learn pipeline object.
    """
    X = df[attribute_cols]
    y = df[target_col]

    # Define the preprocessing step for categorical features
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), attribute_cols)])

    # Create the full pipeline with preprocessing and the XGBoost regressor
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(
            objective='reg:squarederror', 
            n_estimators=100, 
            learning_rate=0.1, 
            random_state=42, 
            enable_categorical=True
        ))
    ])

    print(f"Training model for target: '{target_col}'...")
    model_pipeline.fit(X, y)
    print(f"Training for '{target_col}' complete.")
    return model_pipeline

def main(args):
    """
    Main function to load data, train the price and rating models, 
    and save them to the specified output directory.
    """
    print("Starting the training process for suggestion engine models...")
    
    # --- Configuration ---
    # These should match the columns used by the SuggestionEngine during inference.
    ATTRIBUTE_COLS = ['Type', 'colour', 'Neck', 'Sleeve Length', 'Fit', 'Hemline']
    PRICE_COL = 'price'
    RATING_COL = 'avg_rating'
    
    # --- Load and Clean Data ---
    try:
        df = pd.read_csv(args.data_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at path: {args.data_path}")
        return
        
    required_cols = ATTRIBUTE_COLS + [PRICE_COL, RATING_COL]
    df = df[required_cols].dropna().copy()
    for col in ATTRIBUTE_COLS:
        df[col] = df[col].astype('category')
    print(f"Loaded and cleaned {len(df)} data rows.")

    # --- Train Models ---
    price_model = train_suggestion_model(df, ATTRIBUTE_COLS, PRICE_COL)
    rating_model = train_suggestion_model(df, ATTRIBUTE_COLS, RATING_COL)

    # --- Save Models ---
    # ANNOTATION: We create the output directory if it doesn't exist.
    os.makedirs(args.output_dir, exist_ok=True)
    
    price_model_path = os.path.join(args.output_dir, 'price_model.joblib')
    rating_model_path = os.path.join(args.output_dir, 'rating_model.joblib')
    
    # ANNOTATION: `joblib.dump` is the standard and recommended way to save scikit-learn
    # models and pipelines. It is more efficient for objects containing large NumPy arrays.
    joblib.dump(price_model, price_model_path)
    print(f"✅ Price prediction model saved to: {price_model_path}")
    
    joblib.dump(rating_model, rating_model_path)
    print(f"✅ Rating prediction model saved to: {rating_model_path}")
    print("\nAll models for the suggestion engine have been trained and saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and save the predictive models for the Suggestion Engine.")
    parser.add_argument('--data_path', type=str, default='data/processed/train.csv', help="Path to the training CSV dataset.")
    parser.add_argument('--output_dir', type=str, default='models/suggestion_engine', help="Directory to save the trained model files.")
    args = parser.parse_args()
    main(args)