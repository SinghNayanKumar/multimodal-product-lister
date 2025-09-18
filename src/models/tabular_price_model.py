import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import joblib

class TabularPriceModel:
    """
    A wrapper for an XGBoost regression model that predicts price from
    structured product attributes. This is Stage 2 of the hybrid baseline.
    """
    def __init__(self, categorical_features: list):
        self.categorical_features = categorical_features
        self._pipeline = self._build_pipeline()

    def _build_pipeline(self):
        """Builds the scikit-learn pipeline with a preprocessor and regressor."""
        preprocessor = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)])

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.1, random_state=42))
        ])
        return pipeline

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Trains the model on the provided data."""
        print("Training tabular price model...")
        self._pipeline.fit(X_train, y_train)
        print("Training complete.")

    def predict(self, X: pd.DataFrame) -> list:
        """Makes predictions on new data."""
        return self._pipeline.predict(X)

    def save(self, filepath: str):
        """Saves the trained pipeline to a file."""
        joblib.dump(self._pipeline, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Loads a trained pipeline from a file."""
        self._pipeline = joblib.load(filepath)
        print(f"Model loaded from {filepath}")