"""
Integration and Unit Tests for Model Pipeline and Inference.
"""

import pytest
import sys
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extractor import FEATURE_NAMES, extract_url_features
from src.model import (
    get_model_pipeline,
    train_pipeline,
    evaluate_pipeline,
    save_pipeline,
    load_pipeline
)
from src.predictor import PhishGuardPredictor

@pytest.fixture
def dummy_dataset():
    """Generates synthetic data matching the exact 27-feature schema for fast testing."""
    np.random.seed(42)
    n_samples = 50
    data = {f: np.random.randint(0, 10, size=n_samples) for f in FEATURE_NAMES}
    X = pd.DataFrame(data)
    y = pd.Series(np.random.choice([0, 1], size=n_samples))
    return X, y

def test_pipeline_creation():
    """Ensures supported model types create valid pipelines with correct step names."""
    for m_type in ["logistic_regression", "decision_tree", "random_forest"]:
        pipe = get_model_pipeline(m_type)
        assert "classifier" in pipe.named_steps
        if m_type == "logistic_regression":
            assert "scaler" in pipe.named_steps

def test_pipeline_train_evaluate(dummy_dataset):
    """Tests training and evaluation workflow without errors."""
    X, y = dummy_dataset
    pipe = get_model_pipeline("decision_tree")
    trained = train_pipeline(pipe, X, y)
    metrics = evaluate_pipeline(trained, X, y)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert "confusion_matrix" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0

def test_pipeline_save_and_load(tmp_path, dummy_dataset):
    """Tests joblib serialization and reloading of a trained pipeline."""
    X, y = dummy_dataset
    pipe = get_model_pipeline("decision_tree")
    trained = train_pipeline(pipe, X, y)

    saved_file = tmp_path / "test_model.joblib"
    save_pipeline(trained, saved_file)
    assert saved_file.exists()

    loaded = load_pipeline(saved_file)
    preds = loaded.predict(X)
    assert len(preds) == len(y)

def test_predictor_e2e(tmp_path, dummy_dataset):
    """Tests PhishGuardPredictor end-to-end inference on a raw URL."""
    X, y = dummy_dataset
    pipe = get_model_pipeline("decision_tree")
    trained = train_pipeline(pipe, X, y)

    model_file = tmp_path / "phishing_model.joblib"
    save_pipeline(trained, model_file)

    predictor = PhishGuardPredictor(model_path=model_file)
    result = predictor.predict("https://www.example.com/test-url")

    assert "url" in result
    assert "prediction" in result
    assert "is_phishing" in result
    assert "phishing_probability" in result
    assert "features" in result
    assert len(result["features"]) == len(FEATURE_NAMES)
    assert 0.0 <= result["phishing_probability"] <= 1.0
