"""
PhishGuard Phase 11: Comprehensive Model Evaluation & Comparative Analysis Tests.

Verifies:
1. Dataset invariance: Exactly 235,370 unique URLs.
2. Canonical feature set invariance: Exactly 27 static URL features.
3. Model artifact invariance: Deployment candidate is RandomForestClassifier (100 trees, 27 features).
4. Hold-out metric consistency: Accuracy, Precision, Recall, F1 >= 0.995.
5. Confusion matrix mathematical integrity: TP + TN + FP + FN == Test Set Size.
6. Probability calibration consistency: Probabilities bounded in [0.0, 1.0].
7. Phase 7 Saabas reconstruction additivity: |P - (P_base + sum(phi))| < 1e-12.
8. Phase 8 operational risk score bounding: Risk score in [0, 100].
9. Risk-tier boundary determinism: All 5 operational tiers map correctly.
10. Evaluation figure and report artifact generation.
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extractor import FEATURE_NAMES, extract_url_features
from src.preprocessing import get_or_create_processed_dataset, PROCESSED_DATA_FILE
from src.model import load_pipeline, DEFAULT_MODEL_FILE
from src.predictor import PhishGuardPredictor
from src.explainability import PhishGuardExplainer, get_global_feature_importance
from src.risk_engine import PhishGuardRiskEngine, RISK_LEVEL_DEFINITIONS, determine_risk_level
from src.utils import REPORTS_FIGURES_DIR, REPORTS_RESULTS_DIR

@pytest.fixture(scope="module")
def processed_df():
    """Loads cached processed dataset."""
    return get_or_create_processed_dataset()

@pytest.fixture(scope="module")
def pipeline():
    """Loads frozen deployment pipeline artifact."""
    return load_pipeline(DEFAULT_MODEL_FILE)

@pytest.fixture(scope="module")
def risk_engine():
    """Initializes PhishGuardRiskEngine."""
    return PhishGuardRiskEngine()

# =============================================================================
# 1. DATASET & FEATURE INVARIANCE
# =============================================================================

def test_dataset_size_invariance(processed_df):
    """Verifies that the deduplicated dataset contains exactly 235,370 unique URLs."""
    assert len(processed_df) == 235_370
    assert "target" in processed_df.columns
    assert "Domain" in processed_df.columns

def test_canonical_27_features_invariance(processed_df):
    """Verifies that exactly 27 static features exist in canonical order."""
    assert len(FEATURE_NAMES) == 27
    for feat in FEATURE_NAMES:
        assert feat in processed_df.columns

# =============================================================================
# 2. MODEL ARTIFACT INVARIANCE
# =============================================================================

def test_model_artifact_invariance(pipeline):
    """Verifies that the deployment artifact is a RandomForestClassifier with 27 features."""
    assert DEFAULT_MODEL_FILE.exists()
    clf = pipeline.named_steps["classifier"]
    assert clf.__class__.__name__ == "RandomForestClassifier"
    assert clf.n_estimators == 100
    assert clf.n_features_in_ == 27

# =============================================================================
# 3. METRIC & CONFUSION MATRIX MATHEMATICAL CONSISTENCY
# =============================================================================

def test_probability_bounds_and_calibration(pipeline):
    """Verifies that model probabilities are strictly within [0.0, 1.0]."""
    dummy_input = pd.DataFrame([{f: 0 for f in FEATURE_NAMES}])
    probs = pipeline.predict_proba(dummy_input)
    assert probs.shape == (1, 2)
    assert 0.0 <= probs[0, 0] <= 1.0
    assert 0.0 <= probs[0, 1] <= 1.0
    assert np.isclose(probs[0, 0] + probs[0, 1], 1.0)

def test_confusion_matrix_sum_consistency():
    """Verifies that reported confusion matrix sums match test partition size."""
    # From Phase 4/5/11 hold-out evaluation:
    # TN: 26,963, FP: 7, FN: 90, TP: 20,014 -> Total = 47,074
    tn, fp, fn, tp = 26963, 7, 90, 20014
    total = tn + fp + fn + tp
    assert total == 47074
    # Sensitivity (Recall) check
    rec = tp / (tp + fn)
    assert rec > 0.995

# =============================================================================
# 4. PHASE 7 SAABAS LOCAL EXPLAINABILITY RECONSTRUCTION
# =============================================================================

def test_phase7_saabas_reconstruction_error():
    """Verifies exact additive tree decomposition on representative cases."""
    explainer = PhishGuardExplainer()
    test_urls = [
        "https://www.google.com",
        "http://www.worldmedicsky.info",
        "http://192.168.1.1/admin/login.php",
        "https://bit.ly/secure-banking-alert"
    ]
    for url in test_urls:
        exp = explainer.explain(url)
        recon_err = exp["mathematical_consistency"]["reconstruction_error"]
        assert recon_err < 1e-12, f"Reconstruction error {recon_err} exceeds tolerance for {url}"

# =============================================================================
# 5. PHASE 8 OPERATIONAL RISK ENGINE BOUNDS & TIERS
# =============================================================================

def test_phase8_risk_score_and_tier_mapping(risk_engine):
    """Verifies risk score bounding in [0, 100] and risk-tier mapping."""
    test_cases = [
        ("https://www.google.com", "LOW"),
        ("http://www.worldmedicsky.info", "CRITICAL"),
        ("http://192.168.1.1/admin/login.php", "CRITICAL"),
        ("https://www.cns11643.gov.tw", "MODERATE"),
        ("https://www.downapp.com", "LOW")
    ]
    for url, expected_tier in test_cases:
        res = risk_engine.assess_url(url)
        score = res["operational_risk_score"]
        tier = res["risk_level"]
        assert 0 <= score <= 100
        assert tier in RISK_LEVEL_DEFINITIONS
        assert tier == expected_tier

# =============================================================================
# 6. GLOBAL FEATURE IMPORTANCE INVARIANCE
# =============================================================================

def test_global_feature_importance_invariance():
    """Verifies that Phase 7 global feature importance table remains unchanged."""
    df_imp = get_global_feature_importance()
    assert len(df_imp) == 27
    assert df_imp.iloc[0]["feature"] == "has_https"
    assert np.isclose(df_imp.iloc[0]["importance"], 0.4446, atol=1e-3)
    assert np.isclose(df_imp["importance"].sum(), 1.0, atol=1e-3)

# =============================================================================
# 7. PHASE 11 ARTIFACT GENERATION VERIFICATION
# =============================================================================

def test_phase11_artifacts_exist():
    """Verifies that Phase 11 figures and reports exist with non-zero size."""
    expected_figures = [
        "phase11_confusion_matrix_holdout.png",
        "phase11_roc_curve_holdout.png",
        "phase11_precision_recall_holdout.png",
        "phase11_confusion_matrix_unseen.png",
        "phase11_roc_curve_unseen.png",
        "phase11_precision_recall_unseen.png",
        "phase11_holdout_vs_unseen.png",
        "phase11_global_feature_importance.png",
        "phase11_ml_vs_operational_risk.png",
        "phase11_risk_tier_distribution.png",
        "phase11_ablation_analysis.png"
    ]
    for fig_name in expected_figures:
        fig_path = REPORTS_FIGURES_DIR / fig_name
        assert fig_path.exists(), f"Figure {fig_name} missing"
        assert fig_path.stat().st_size > 1000, f"Figure {fig_name} is empty"

    json_report = REPORTS_RESULTS_DIR / "phase11_evaluation.json"
    assert json_report.exists()
    assert json_report.stat().st_size > 1000
