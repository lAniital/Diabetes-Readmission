"""Model training utilities for the diabetes readmission project."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


DEFAULT_MODEL_PATH = Path("models/random_forest_readmission.joblib")


def train_random_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train a Random Forest model with class weights for readmission imbalance."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return model


def save_model(
    model: RandomForestClassifier,
    output_path: str | Path = DEFAULT_MODEL_PATH,
) -> Path:
    """Persist a trained model to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path