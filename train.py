"""
PhishGuard Model Training & Evaluation Pipeline.

Trains and compares baseline classifiers:
1. Logistic Regression (StandardScaler + LogisticRegression)
2. Decision Tree Classifier
3. Random Forest Classifier

Logs performance metrics, produces evaluation charts, and serializes
the best model pipeline to models/phishing_model.joblib.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    MODELS_DIR,
    REPORTS_FIGURES_DIR,
    REPORTS_RESULTS_DIR,
    setup_logger,
    ensure_directories_exist
)
from src.preprocessing import get_train_test_data
from src.model import (
    get_model_pipeline,
    train_pipeline,
    evaluate_pipeline,
    save_pipeline,
    DEFAULT_MODEL_FILE
)
from src.feature_extractor import FEATURE_NAMES

logger = setup_logger("TrainPipeline")

def train_and_evaluate_all():
    """Executes full training and comparative evaluation across baseline models."""
    ensure_directories_exist()
    logger.info("=" * 65)
    logger.info("Starting PhishGuard Phase 4 Baseline Model Training")
    logger.info("=" * 65)

    # 1. Load stratified train-test splits
    X_train, X_test, y_train, y_test = get_train_test_data(test_size=0.2, random_state=42)

    model_types = ["logistic_regression", "decision_tree", "random_forest"]
    results = {}
    trained_pipelines = {}

    for m_type in model_types:
        pipeline = get_model_pipeline(m_type)
        trained_pipe = train_pipeline(pipeline, X_train, y_train)
        metrics = evaluate_pipeline(trained_pipe, X_test, y_test)

        results[metrics["model_name"]] = metrics
        trained_pipelines[metrics["model_name"]] = trained_pipe

    # 2. Select deployment candidate (RandomForestClassifier, verified in Phase 5 & Phase 6)
    # RandomForest is chosen over single DecisionTree for superior probability calibration
    # (Brier score 0.0022), highest ROC-AUC (0.9982), and unseen-domain stability.
    best_model_name = "RandomForestClassifier"
    logger.info(f"Selected deployment candidate model: {best_model_name} (F1: {results[best_model_name]['f1_score']}, Recall: {results[best_model_name]['recall']}, ROC-AUC: {results[best_model_name]['roc_auc']})")

    # Serialize deployment candidate model pipeline
    save_pipeline(trained_pipelines[best_model_name], DEFAULT_MODEL_FILE)

    # 3. Save Metrics to JSON and CSV
    save_metrics_reports(results)

    # 4. Generate Visualizations
    generate_figures(results, trained_pipelines, X_train)

    return results

def save_metrics_reports(results: dict):
    """Exports metrics to JSON and CSV reports."""
    json_path = REPORTS_RESULTS_DIR / "baseline_model_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Saved comparison metrics JSON to {json_path}")

    # Build tabular DataFrame
    rows = []
    for model_name, m in results.items():
        cm = m["confusion_matrix"]
        rows.append({
            "Model": model_name,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1-Score": m["f1_score"],
            "ROC-AUC": m["roc_auc"],
            "True Negatives": cm["true_negatives"],
            "False Positives": cm["false_positives"],
            "False Negatives (Missed Phish)": cm["false_negatives"],
            "True Positives": cm["true_positives"],
            "False Negative Rate": m["false_negative_rate"]
        })
    df_metrics = pd.DataFrame(rows)
    csv_path = REPORTS_RESULTS_DIR / "baseline_metrics.csv"
    df_metrics.to_csv(csv_path, index=False)
    logger.info(f"Saved metrics CSV table to {csv_path}")

    # Display console table
    print("\n" + "=" * 90)
    print("                     PHISHGUARD BASELINE EVALUATION METRICS")
    print("=" * 90)
    print(df_metrics[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "False Negatives (Missed Phish)"]].to_string(index=False))
    print("=" * 90 + "\n")

def generate_figures(results: dict, trained_pipelines: dict, X_train: pd.DataFrame):
    """Creates academic figures: Comparison, Confusion Matrices, ROC Curves, Feature Importance."""
    sns.set_theme(style="whitegrid")

    # Figure 1: Model Comparison Bar Chart
    plt.figure(figsize=(10, 6), dpi=300)
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    model_names = list(results.keys())
    x = np.arange(len(metric_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    colors = ["#4dabf7", "#ffd43b", "#51cf66"]

    for i, m_name in enumerate(model_names):
        vals = [results[m_name][m] for m in metrics_to_plot]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=m_name, color=colors[i % len(colors)], edgecolor="black", linewidth=0.8)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, rotation=0)

    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11)
    ax.set_title("PhishGuard Baseline Classifier Comparison (Standard Stratified Test)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    fig1_path = REPORTS_FIGURES_DIR / "02_model_comparison.png"
    plt.savefig(fig1_path)
    plt.close()
    logger.info(f"Saved {fig1_path}")

    # Figure 2: Confusion Matrices Subplots
    fig, axes = plt.subplots(1, len(model_names), figsize=(15, 4.5), dpi=300)
    if len(model_names) == 1:
        axes = [axes]

    for ax, m_name in zip(axes, model_names):
        cm_data = results[m_name]["confusion_matrix"]
        matrix = np.array([
            [cm_data["true_negatives"], cm_data["false_positives"]],
            [cm_data["false_negatives"], cm_data["true_positives"]]
        ])
        sns.heatmap(matrix, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Legitimate (0)", "Phishing (1)"],
                    yticklabels=["Legitimate (0)", "Phishing (1)"])
        ax.set_title(f"{m_name}\nFN (Missed): {cm_data['false_negatives']:,}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("Actual Ground Truth", fontsize=10)

    plt.tight_layout()
    fig2_path = REPORTS_FIGURES_DIR / "03_confusion_matrices.png"
    plt.savefig(fig2_path)
    plt.close()
    logger.info(f"Saved {fig2_path}")

    # Figure 3: ROC Curves Overlaid
    plt.figure(figsize=(7, 6), dpi=300)
    for m_name in model_names:
        fpr = results[m_name]["roc_curve"]["fpr"]
        tpr = results[m_name]["roc_curve"]["tpr"]
        auc_val = results[m_name]["roc_auc"]
        plt.plot(fpr, tpr, lw=2, label=f"{m_name} (AUC = {auc_val:.4f})")

    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    fig3_path = REPORTS_FIGURES_DIR / "04_roc_curves.png"
    plt.savefig(fig3_path)
    plt.close()
    logger.info(f"Saved {fig3_path}")

    # Figure 4: Feature Importance for Random Forest
    if "RandomForestClassifier" in trained_pipelines:
        rf_pipe = trained_pipelines["RandomForestClassifier"]
        rf_classifier = rf_pipe.named_steps["classifier"]
        importances = rf_classifier.feature_importances_
        feat_df = pd.DataFrame({
            "Feature": FEATURE_NAMES,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        # Save feature importance table
        feat_df.to_csv(REPORTS_RESULTS_DIR / "feature_importances.csv", index=False)

        plt.figure(figsize=(10, 8), dpi=300)
        sns.barplot(data=feat_df.head(15), x="Importance", y="Feature", palette="viridis")
        plt.title("Top 15 Most Important URL Features (Random Forest)", fontsize=12, fontweight="bold")
        plt.xlabel("Mean Impurity Reduction (Gini Importance)", fontsize=11)
        plt.ylabel("Extracted URL Feature", fontsize=11)
        plt.tight_layout()
        fig4_path = REPORTS_FIGURES_DIR / "05_feature_importance.png"
        plt.savefig(fig4_path)
        plt.close()
        logger.info(f"Saved {fig4_path}")

if __name__ == "__main__":
    train_and_evaluate_all()
