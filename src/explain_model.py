"""SHAP explainability utilities for the diabetes readmission project."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap


PLOTS_DIR = Path("outputs/plots")
SUMMARY_PLOT_PATH = PLOTS_DIR / "shap_summary.png"
BAR_PLOT_PATH = PLOTS_DIR / "shap_bar.png"


def explain_model(model, x_test: pd.DataFrame, max_samples: int = 300) -> dict[str, Path]:
    """Create and save SHAP plots for the trained model."""
    sample = x_test.sample(
        n=min(max_samples, len(x_test)),
        random_state=42,
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    shap_values = get_positive_class_shap_values(shap_values)

    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    plt.savefig(SUMMARY_PLOT_PATH, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"SHAP summary plot saved to: {SUMMARY_PLOT_PATH}")

    plt.figure()
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    plt.savefig(BAR_PLOT_PATH, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"SHAP bar plot saved to: {BAR_PLOT_PATH}")

    return {
        "summary_plot": SUMMARY_PLOT_PATH,
        "bar_plot": BAR_PLOT_PATH,
    }


def get_positive_class_shap_values(shap_values):
    """Return SHAP values for the positive readmission class when needed."""
    if isinstance(shap_values, list):
        return shap_values[1]

    if getattr(shap_values, "ndim", None) == 3:
        return shap_values[:, :, 1]

    return shap_values

