"""
PhishGuard Model Definition, Training, and Evaluation Pipelines.

Implements baseline classifiers:
1. Logistic Regression (with StandardScaler in Pipeline to avoid data leakage)
2. Decision Tree Classifier
3. Random Forest Classifier

Calculates standard classification metrics with emphasis on phishing recall
and false negative rate.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import MODELS_DIR, setup_logger
from src.feature_extractor import FEATURE_NAMES

logger = setup_logger("ModelEngine")

DEFAULT_MODEL_FILE = MODELS_DIR / "phishing_model.joblib"

def get_model_pipeline(model_type: str = "random_forest") -> Pipeline:
    """Builds a scikit-learn Pipeline for the specified model architecture.

    Args:
        model_type: One of 'logistic_regression', 'decision_tree', 'random_forest'.

    Returns:
        Configured scikit-learn Pipeline instance.
    """
    model_type = model_type.lower()
    if model_type in ("logistic_regression", "lr"):
        # Logistic Regression requires scaled inputs.
        # StandardScaler is placed inside the pipeline so scaling parameters
        # are fit strictly on training splits, eliminating data leakage.
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight="balanced"
            ))
        ])

    elif model_type in ("decision_tree", "dt"):
        return Pipeline([
            ("classifier", DecisionTreeClassifier(
                max_depth=15,
                min_samples_split=10,
                random_state=42,
                class_weight="balanced"
            ))
        ])

    elif model_type in ("random_forest", "rf"):
        return Pipeline([
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            ))
        ])

    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'. Choose from 'logistic_regression', 'decision_tree', 'random_forest'.")

def train_pipeline(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fits the model pipeline on training data.

    Args:
        pipeline: Unfitted scikit-learn Pipeline.
        X_train: Feature DataFrame matching FEATURE_NAMES.
        y_train: Target Series (0 = Legitimate, 1 = Phishing).

    Returns:
        Fitted Pipeline.
    """
    classifier_name = pipeline.named_steps["classifier"].__class__.__name__
    logger.info(f"Training {classifier_name} on {X_train.shape[0]:,} samples with {X_train.shape[1]} features...")
    pipeline.fit(X_train, y_train)
    logger.info(f"Training completed for {classifier_name}.")
    return pipeline

def evaluate_pipeline(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Computes comprehensive performance metrics for a trained pipeline.

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        X_test: Test feature DataFrame.
        y_test: Test target Series (0 = Legitimate, 1 = Phishing).

    Returns:
        Dictionary containing all evaluation metrics and confusion matrix elements.
    """
    classifier_name = pipeline.named_steps["classifier"].__class__.__name__
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
    auc = float(roc_auc_score(y_test, y_prob))

    # Security-centric calculations
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0  # False Negative Rate
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0  # False Positive Rate

    fpr_curve, tpr_curve, _ = roc_curve(y_test, y_prob)

    metrics = {
        "model_name": classifier_name,
        "test_samples": int(len(y_test)),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "false_negative_rate": round(fnr, 4),
        "false_positive_rate": round(fpr, 4),
        "roc_curve": {
            "fpr": fpr_curve.tolist(),
            "tpr": tpr_curve.tolist()
        }
    }

    return metrics

def save_pipeline(pipeline: Pipeline, filepath: Path = DEFAULT_MODEL_FILE) -> None:
    """Serializes the complete preprocessing + classifier pipeline."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, filepath)
    logger.info(f"Saved model pipeline artifact to {filepath}")

def load_pipeline(filepath: Path = DEFAULT_MODEL_FILE) -> Pipeline:
    """Loads a serialized model pipeline artifact."""
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found at {filepath}. Run train.py first.")
    pipeline = joblib.load(filepath)
    return pipeline
