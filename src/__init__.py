"""Data preprocessing for the diabetes readmission dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


TRACKING_COLUMNS = [
    "encounter_id",
    "patient_nbr",
]

HIGH_MISSING_COLUMNS = [
    "weight",
    "payer_code",
    "medical_specialty",
]

DIAGNOSIS_COLUMNS = [
    "diag_1",
    "diag_2",
    "diag_3",
]


def load_and_preprocess_data(
    csv_path: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load, clean, encode, and split the diabetes readmission dataset.

    The target is binary:
    - 1 means readmitted in fewer than 30 days.
    - 0 means not readmitted within 30 days.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Place diabetic_data.csv in the data folder."
        )

    data = pd.read_csv(csv_path)
    data = clean_data(data)

    target = (data["readmitted"] == "<30").astype(int)
    features = data.drop(columns=["readmitted"])
    features = encode_features(features)

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Apply dataset-specific cleaning rules."""
    cleaned = data.copy()
    cleaned = cleaned.replace("?", pd.NA)

    columns_to_drop = [
        column
        for column in TRACKING_COLUMNS + HIGH_MISSING_COLUMNS
        if column in cleaned.columns
    ]
    cleaned = cleaned.drop(columns=columns_to_drop)

    cleaned = drop_single_value_columns(cleaned)
    cleaned = normalize_diagnosis_columns(cleaned)

    return cleaned


def drop_single_value_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that do not add predictive information."""
    single_value_columns = [
        column for column in data.columns if data[column].nunique(dropna=False) <= 1
    ]
    return data.drop(columns=single_value_columns)


def normalize_diagnosis_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Convert diagnosis code columns to numeric values where possible."""
    normalized = data.copy()
    for column in DIAGNOSIS_COLUMNS:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized


def encode_features(features: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features and impute remaining missing values."""
    encoded = features.copy()

    numeric_columns = encoded.select_dtypes(include=["number"]).columns
    categorical_columns = encoded.columns.difference(numeric_columns)

    for column in numeric_columns:
        encoded[column] = encoded[column].fillna(encoded[column].median())

    for column in categorical_columns:
        encoded[column] = encoded[column].fillna("Missing").astype(str)

    encoded = pd.get_dummies(encoded, columns=categorical_columns, drop_first=True)
    return encoded