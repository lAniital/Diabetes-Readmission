# Diabetes Readmission Random Forest

Explainable machine learning pipeline for predicting whether a diabetic patient is at risk of being readmitted within 30 days of hospital discharge.

The project uses the UCI/Kaggle diabetes readmission dataset, a Random Forest classifier with class weighting, and SHAP explanations for model interpretability.

## Project Structure

```text
Diabet-readmission/
├── data/
│   └── diabetic_data.csv
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── explain_model.py
├── main.py
├── requirements.txt
└── README.md
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place the dataset at:

```text
data/diabetic_data.csv
```

## Run

```powershell
python main.py
```

## Roadmap

1. Data preprocessing and train/test split
2. Random Forest training with class imbalance handling
3. Clinical evaluation using recall, confusion matrix, and ROC-AUC
4. SHAP explainability plots
5. One-command orchestration from `main.py`