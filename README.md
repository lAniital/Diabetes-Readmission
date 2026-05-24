# Clinical Risk Prediction: Diabetes Hospital Readmission Pipeline Using Random Forest

## Project Overview

Predicting hospital readmissions within 30 days of discharge is an important problem in healthcare analytics and clinical informatics. Early readmissions can increase hospital costs, indicate gaps in transitional care, and negatively affect patient outcomes.

This project builds an explainable machine learning pipeline to predict whether a diabetic patient is at high risk of readmission within 30 days. The model uses clinical, demographic, medication, and hospital encounter data from the diabetes readmission dataset.

The pipeline uses a Random Forest classifier with class weighting to handle class imbalance, then applies SHAP explainability to help interpret which features influence model predictions.

> This project is for educational and portfolio purposes only. It is not intended for clinical deployment or medical decision-making.

## Key Objectives

- Clean real-world hospital data with missing values and categorical features.
- Convert the readmission target into a binary clinical risk label.
- Train a Random Forest model using class imbalance handling.
- Evaluate the model with clinically relevant metrics such as recall, precision, F1-score, confusion matrix, and ROC-AUC.
- Generate SHAP plots to explain model behavior.

## Project Structure

```text
Diabetes-Readmission/
|-- data/
|   |-- diabetic_data.csv
|-- models/
|   |-- random_forest_readmission.joblib
|-- outputs/
|   |-- plots/
|       |-- shap_summary.png
|       |-- shap_bar.png
|-- src/
|   |-- __init__.py
|   |-- data_preprocessing.py
|   |-- train_model.py
|   |-- evaluate_model.py
|   |-- explain_model.py
|-- main.py
|-- requirements.txt
|-- .gitignore
|-- README.md
```

## Pipeline Phases

### Phase 1: Data Preprocessing

The preprocessing script loads the raw dataset, replaces missing values marked as `?`, removes tracking identifiers, drops highly missing columns, converts the target variable into a binary label, encodes categorical variables, and creates a train/test split.

Target definition:

```text
1 = readmitted within 30 days
0 = not readmitted within 30 days
```

### Phase 2: Model Training

The training script fits a `RandomForestClassifier` using:

```python
class_weight="balanced"
```

This helps the model pay more attention to the minority class: patients readmitted within 30 days.

### Phase 3: Model Evaluation

The evaluation script reports:

- Confusion matrix
- Classification report
- Recall
- Precision
- F1-score
- ROC-AUC

In this healthcare-style problem, recall is especially important because missing a high-risk patient can be more costly than flagging a low-risk patient for extra follow-up.

### Phase 4: Explainability With SHAP

The explainability script uses SHAP `TreeExplainer` to interpret the Random Forest model and saves plots to:

```text
outputs/plots/shap_summary.png
outputs/plots/shap_bar.png
```

These plots help show which features are most influential in the model's readmission risk predictions.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Place the dataset here:

```text
data/diabetic_data.csv
```

## Run the Pipeline

```powershell
python main.py
```

The pipeline will:

1. Preprocess the dataset.
2. Train the Random Forest model.
3. Save the trained model.
4. Evaluate clinical metrics.
5. Generate SHAP explainability plots.

## Current Model Results

Example evaluation output from the current Random Forest run:

```text
Recall: 0.0048
Precision: 0.5500
F1-score: 0.0096
ROC-AUC: 0.6565
```

The very low recall shows that the baseline Random Forest is not yet strong at detecting the minority readmission class. This is an important finding and a natural next step for future improvement.

## Future Improvements

- Tune Random Forest hyperparameters.
- Adjust the prediction threshold to improve recall.
- Compare with Logistic Regression, XGBoost, or LightGBM.
- Add cross-validation.
- Improve feature engineering for diagnosis codes and medication history.
- Save evaluation reports automatically.

## Tech Stack

- Python
- pandas
- scikit-learn
- matplotlib
- SHAP
- joblib
