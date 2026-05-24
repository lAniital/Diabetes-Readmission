"""Pipeline entry point for the diabetes readmission project."""

from pathlib import Path

from src.data_preprocessing import load_and_preprocess_data


DATA_PATH = Path("data/diabetic_data.csv")


def main() -> None:
    """Run the project pipeline."""
    try:
        x_train, x_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)
    except FileNotFoundError as error:
        print(error)
        return

    print("Preprocessing complete.")
    print(f"Training rows: {len(x_train):,}")
    print(f"Test rows: {len(x_test):,}")
    print(f"Training features: {x_train.shape[1]:,}")
    print(f"Positive class rate in train: {y_train.mean():.3f}")


if __name__ == "__main__":
    main()