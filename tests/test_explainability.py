"""
Unit and Integration Tests for PhishGuard Phase 7 Explainability Engine.

Verifies:
1. Exactly 27 features are used in global importance and local attribution.
2. Feature names match FEATURE_NAMES in src/feature_extractor.py.
3. explain_url accepts a raw URL string without network calls.
4. Prediction from explanation matches PhishGuardPredictor.
5. Probability from explanation matches PhishGuardPredictor within numerical tolerance.
6. Global feature importance contains exactly 27 features.
7. Feature importance values are finite (no NaN, Inf).
8. Feature importance values sum to approximately 1.0.
9. Local explanation contains valid feature references and descriptions.
10. Local feature decomposition is strictly offline (no socket/HTTP requests).
11. Explanation functions correctly for both phishing and legitimate examples.
12. Mathematical reconstruction error of Saabas decomposition is practically zero (< 1e-5).
"""

import pytest
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extractor import FEATURE_NAMES, extract_url_features
from src.predictor import PhishGuardPredictor
from src.explainability import (
    PhishGuardExplainer,
    explain_url,
    get_global_feature_importance,
    compute_tree_local_contributions,
    FEATURE_METADATA
)

@pytest.fixture(scope="module")
def explainer():
    """Initializes a shared PhishGuardExplainer instance."""
    return PhishGuardExplainer()

@pytest.fixture(scope="module")
def predictor():
    """Initializes a shared PhishGuardPredictor instance."""
    return PhishGuardPredictor()

def test_feature_metadata_count():
    """1. Verifies that exactly 27 features are documented in FEATURE_METADATA."""
    assert len(FEATURE_METADATA) == 27
    assert set(FEATURE_METADATA.keys()) == set(FEATURE_NAMES)

def test_feature_names_match():
    """2. Verifies that all metadata keys match FEATURE_NAMES in src/feature_extractor.py."""
    for feat in FEATURE_NAMES:
        assert feat in FEATURE_METADATA
        meta = FEATURE_METADATA[feat]
        assert "name" in meta
        assert "category" in meta
        assert "description" in meta
        assert len(meta["name"]) > 0

def test_explain_url_accepts_raw_url():
    """3. Verifies that explain_url accepts a raw URL string and returns a complete dictionary."""
    url = "https://example.com/login"
    res = explain_url(url)
    assert isinstance(res, dict)
    assert res["url"] == url
    assert "prediction" in res
    assert "phishing_probability" in res
    assert "local_contributions" in res
    assert "explanation" in res

def test_prediction_matches_predictor(explainer, predictor):
    """4. Verifies that the prediction label from explanation matches PhishGuardPredictor."""
    test_urls = [
        "https://www.google.com",
        "http://paypal-security-update.com/login/index.php",
        "https://github.com/login"
    ]
    for url in test_urls:
        pred_res = predictor.predict(url)
        exp_res = explainer.explain(url)
        assert pred_res["prediction"] == exp_res["prediction"]
        assert pred_res["is_phishing"] == exp_res["is_phishing"]

def test_probability_matches_predictor(explainer, predictor):
    """5. Verifies that phishing probability matches PhishGuardPredictor within numerical tolerance."""
    test_urls = [
        "https://www.wikipedia.org",
        "http://192.168.1.1/login.php",
        "https://bank-verify-account.tk/signin"
    ]
    for url in test_urls:
        pred_res = predictor.predict(url)
        exp_res = explainer.explain(url)
        assert np.isclose(pred_res["phishing_probability"], exp_res["phishing_probability"], atol=1e-4)

def test_global_importance_feature_count(explainer):
    """6. Verifies that global feature importance contains exactly 27 features."""
    df_imp = explainer.get_global_importance()
    assert len(df_imp) == 27
    assert set(df_imp["feature"]) == set(FEATURE_NAMES)

def test_global_importance_values_finite(explainer):
    """7. Verifies that all feature importance values are non-negative and finite."""
    df_imp = explainer.get_global_importance()
    assert not df_imp["importance"].isnull().any()
    assert not np.isinf(df_imp["importance"]).any()
    assert (df_imp["importance"] >= 0.0).all()

def test_global_importance_sum(explainer):
    """8. Verifies that feature importance values sum to approximately 1.0 for Random Forest."""
    df_imp = explainer.get_global_importance()
    total_importance = df_imp["importance"].sum()
    assert np.isclose(total_importance, 1.0, atol=1e-3)

def test_local_explanation_feature_references(explainer):
    """9. Verifies that local explanation contains valid feature references and breakdown keys."""
    url = "http://account-verification.suspicious.com/auth"
    res = explainer.explain(url)

    assert len(res["local_contributions"]) == 27
    for feat_name, contrib in res["local_contributions"].items():
        assert feat_name in FEATURE_NAMES
        assert isinstance(contrib, (float, np.floating))
        assert np.isfinite(contrib)

    # Check top contributing features list
    for top_f in res["top_contributing_features"]:
        assert top_f["feature"] in FEATURE_NAMES
        assert "contribution" in top_f
        assert "direction" in top_f
        assert top_f["direction"] in ["increases_phishing", "decreases_phishing", "neutral"]

def test_no_network_requests_required(monkeypatch):
    """10. Verifies that feature extraction and explanation require no socket/network calls."""
    import socket
    def fail_socket(*args, **kwargs):
        raise RuntimeError("Network socket call attempted during offline explainability!")

    monkeypatch.setattr(socket, "socket", fail_socket)

    url = "https://safe-domain.org/about/us?lang=en#header"
    res = explain_url(url)
    assert res["prediction"] is not None
    assert "explanation" in res

def test_explanation_phishing_and_legitimate(explainer):
    """11. Verifies that explanation produces sound results for both phishing and legitimate examples."""
    phish_url = "http://paypal-verification-alert.com/login"
    legit_url = "https://www.nationalgeographic.com"

    exp_phish = explainer.explain(phish_url)
    exp_legit = explainer.explain(legit_url)

    # Phishing check
    assert exp_phish["phishing_probability"] > 0.50
    assert len(exp_phish["top_positive_contributors"]) > 0
    assert "http" in exp_phish["explanation"].lower() or "phishing" in exp_phish["explanation"].lower()

    # Legitimate check
    assert exp_legit["phishing_probability"] < 0.50
    assert len(exp_legit["top_negative_contributors"]) > 0
    assert "legitimate" in exp_legit["explanation"].lower()

def test_saabas_reconstruction_precision(explainer):
    """12. Verifies that Bias + Sum(Contributions) exactly equals predicted probability."""
    test_urls = [
        "https://www.google.com",
        "http://phish-login-portal.com/auth.php?token=xyz",
        "http://192.168.0.1/admin",
        "https://sub1.sub2.example.org/test"
    ]
    for url in test_urls:
        res = explainer.explain(url)
        recon_err = res["mathematical_consistency"]["reconstruction_error"]
        assert recon_err < 1e-5, f"Reconstruction error {recon_err} too high for {url}"
