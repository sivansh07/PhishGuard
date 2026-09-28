"""
PhishGuard Standalone Model Evaluation Script.

Evaluates trained model pipeline artifacts on the test split, prints
classification reports, confusion matrices, and detailed false negative breakdown.
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logger
from src.preprocessing import get_train_test_data
from src.model import load_pipeline, evaluate_pipeline, DEFAULT_MODEL_FILE

logger = setup_logger("Evaluation")

def run_evaluation(model_path: Path = DEFAULT_MODEL_FILE):
    """Loads trained model pipeline and runs thorough evaluation on test split."""
    logger.info(f"Loading model pipeline from {model_path}...")
    pipeline = load_pipeline(model_path)
    model_name = pipeline.named_steps["classifier"].__class__.__name__

    _, X_test, _, y_test = get_train_test_data(test_size=0.2, random_state=42)

    logger.info(f"Evaluating {model_name} on {len(X_test):,} held-out test samples...")
    metrics = evaluate_pipeline(pipeline, X_test, y_test)
    y_pred = pipeline.predict(X_test)

    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm["true_negatives"], cm["false_positives"], cm["false_negatives"], cm["true_positives"]

    print("\n" + "=" * 70)
    print(f"        PHISHGUARD MODEL EVALUATION REPORT: {model_name}")
    print("=" * 70)
    print(f"Test Samples:              {metrics['test_samples']:,}")
    print(f"Accuracy:                  {metrics['accuracy']:.4f}")
    print(f"Precision (Phishing):      {metrics['precision']:.4f}")
    print(f"Recall (Phishing):         {metrics['recall']:.4f}")
    print(f"F1-Score (Phishing):       {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:                   {metrics['roc_auc']:.4f}")
    print("-" * 70)
    print("CONFUSION MATRIX BREAKDOWN:")
    print(f"  • True Negatives  (Legitimate correctly identified):  {tn:,}")
    print(f"  • False Positives (Legitimate incorrectly blocked):   {fp:,}  (FPR: {metrics['false_positive_rate']:.4f})")
    print(f"  • False Negatives (Phishing MISSED as safe):          {fn:,}  (FNR: {metrics['false_negative_rate']:.4f}) [CRITICAL]")
    print(f"  • True Positives  (Phishing correctly caught):        {tp:,}")
    print("-" * 70)
    print("DETAILED SCIKIT-LEARN CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate (0)", "Phishing (1)"], digits=4))
    print("=" * 70 + "\n")

    return metrics

if __name__ == "__main__":
    run_evaluation()
