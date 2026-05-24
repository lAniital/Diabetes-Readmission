"""Pipeline entry point for the diabetes readmission project."""

from pathlib import Path

from src.data_preprocessing import load_and_preprocess_data
from src.evaluate_model import evaluate_model
from src.explain_model import explain_model
from src.train_model import save_model, train_random_forest


DATA_PATH = Path("data/diabetic_data.csv")
MODEL_PATH = Path("models/random_forest_readmission.joblib")


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

    model = train_random_forest(x_train, y_train)
    saved_path = save_model(model, MODEL_PATH)

    print("Training complete.")
    print(f"Model saved to: {saved_path}")

    evaluate_model(model, x_test, y_test)
    print("Evaluation complete.")

    explain_model(model, x_test, max_samples=50)
    print("Explainability complete.")


if __name__ == "__main__":
    main()

