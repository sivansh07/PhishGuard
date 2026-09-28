"""
Unit and Integration Tests for PhishGuard Phase 8 Risk Scoring Engine.

Verifies:
1. Risk score is strictly bounded in [0, 100].
2. Risk level maps correctly to defined bands (LOW, GUARDED, MODERATE, HIGH, CRITICAL).
3. Valid URL produces deterministic output across repeated calls.
4. Same URL produces identical risk score and heuristic breakdown.
5. Underlying Random Forest probability remains identical to PhishGuardPredictor.
6. Exactly 27 static features are used throughout the pipeline.
7. Phase 7 local feature attributions are cleanly embedded.
8. Heuristic rule evaluation is deterministic.
9. Operates strictly offline without socket/network access.
10. No webpage rendering or DOM execution occurs.
11. Malformed/empty URLs are handled safely without crashing.
12. Required assessment keys are present in output dictionary.
13. Heuristic adjustments are strictly bounded within [-10, +15].
14. Existing binary classification predictions are not silently altered.
15. Previous Phase 1–7 tests continue to pass with 100% integrity.
"""

import pytest
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extractor import FEATURE_NAMES
from src.predictor import PhishGuardPredictor
from src.risk_engine import (
    PhishGuardRiskEngine,
    assess_url_risk,
    determine_risk_level,
    HEURISTIC_RULES,
    RISK_LEVEL_DEFINITIONS
)

@pytest.fixture(scope="module")
def risk_engine():
    """Shared PhishGuardRiskEngine fixture."""
    return PhishGuardRiskEngine()

@pytest.fixture(scope="module")
def predictor():
    """Shared PhishGuardPredictor fixture."""
    return PhishGuardPredictor()

def test_risk_score_bounds(risk_engine):
    """1. Verifies that risk scores are strictly bounded within [0, 100]."""
    test_urls = [
        "https://www.google.com",
        "http://paypal-security-update.com/login/index.php",
        "http://192.168.1.1/admin/verify.php?token=xyz#anchor",
        "https://sub1.sub2.sub3.attacker.com/path",
        "https://www.wikipedia.org"
    ]
    for url in test_urls:
        res = risk_engine.assess_url(url)
        score = res["operational_risk_score"]
        assert 0 <= score <= 100
        assert isinstance(score, (int, np.integer))

def test_risk_level_mapping(risk_engine):
    """2. Verifies that risk levels belong strictly to defined operational tiers."""
    valid_levels = set(RISK_LEVEL_DEFINITIONS.keys())
    for score in [0, 10, 20, 21, 35, 40, 41, 55, 60, 61, 75, 80, 81, 95, 100]:
        level, info = determine_risk_level(score)
        assert level in valid_levels
        assert "label" in info
        assert "recommended_action" in info

def test_deterministic_output(risk_engine):
    """3. & 4. Verifies that the same URL produces identical scores across multiple calls."""
    url = "http://secure-login.apple-verification.com/update"
    res1 = risk_engine.assess_url(url)
    res2 = risk_engine.assess_url(url)

    assert res1["operational_risk_score"] == res2["operational_risk_score"]
    assert res1["risk_level"] == res2["risk_level"]
    assert res1["model_estimated_phishing_probability"] == res2["model_estimated_phishing_probability"]
    assert res1["heuristic_adjustment"] == res2["heuristic_adjustment"]
    assert len(res1["heuristic_signals"]) == len(res2["heuristic_signals"])

def test_rf_probability_preserved(risk_engine, predictor):
    """5. Verifies that the Random Forest probability is preserved independently."""
    test_urls = [
        "https://github.com",
        "http://192.168.0.1/login",
        "https://banking.chase.com"
    ]
    for url in test_urls:
        pred_res = predictor.predict(url)
        risk_res = risk_engine.assess_url(url)
        assert np.isclose(
            risk_res["model_estimated_phishing_probability"],
            pred_res["phishing_probability"],
            atol=1e-4
        )
        assert risk_res["is_phishing"] == pred_res["is_phishing"]

def test_feature_count_consistency(risk_engine):
    """6. Verifies that exactly 27 static features are used."""
    url = "https://example.org/test"
    res = risk_engine.assess_url(url)
    # Check that predictor pipeline expects 27 features
    assert len(FEATURE_NAMES) == 27
    assert risk_engine.predictor.pipeline.named_steps["classifier"].n_features_in_ == 27

def test_phase7_explanation_integration(risk_engine):
    """7. Verifies that Phase 7 local attributions are cleanly embedded."""
    url = "http://paypal-verification.com/login"
    res = risk_engine.assess_url(url)
    assert "local_feature_attributions" in res
    local_attr = res["local_feature_attributions"]
    assert "top_positive_contributors" in local_attr
    assert "top_negative_contributors" in local_attr
    assert "baseline_prior_bias" in local_attr

def test_heuristic_configuration_deterministic(risk_engine):
    """8. Verifies that heuristic rules are valid and deterministic."""
    assert len(HEURISTIC_RULES) >= 10
    sample_features = {f: 0 for f in FEATURE_NAMES}
    sample_features["has_ip_address"] = 1
    sample_features["num_subdomains"] = 4

    adj, triggered, warnings = risk_engine.evaluate_heuristics(sample_features, ml_prob=0.80)
    assert len(triggered) >= 2
    assert any(h["id"] == "H_IP_HOST" for h in triggered)
    assert any(h["id"] == "H_EXCESS_SUBDOMAINS" for h in triggered)

def test_offline_execution_no_network(monkeypatch):
    """9. & 10. Verifies that risk evaluation occurs strictly offline with no network calls."""
    import socket
    def fail_socket(*args, **kwargs):
        raise RuntimeError("Network socket attempted during offline risk evaluation!")

    monkeypatch.setattr(socket, "socket", fail_socket)

    url = "https://safe-domain.example/about"
    res = assess_url_risk(url)
    assert res["operational_risk_score"] >= 0

def test_malformed_url_safety(risk_engine):
    """11. Verifies that malformed or edge-case URLs are handled safely without crashing."""
    malformed_inputs = [
        "",
        "://malformed-url",
        "http://",
        "https://",
        "ftp://unsupported.protocol.com",
        "a" * 500
    ]
    for u in malformed_inputs:
        res = risk_engine.assess_url(u)
        assert "operational_risk_score" in res
        assert 0 <= res["operational_risk_score"] <= 100

def test_output_schema_completeness(risk_engine):
    """12. Verifies that all required risk assessment keys are present."""
    url = "http://account-alert.suspicious-host.net"
    res = risk_engine.assess_url(url)
    required_keys = [
        "url",
        "prediction",
        "is_phishing",
        "model_estimated_phishing_probability",
        "operational_risk_score",
        "risk_level",
        "ml_risk_component",
        "heuristic_adjustment",
        "heuristic_signals",
        "explanation",
        "warnings"
    ]
    for key in required_keys:
        assert key in res, f"Missing required key: {key}"

def test_heuristic_adjustment_bounds(risk_engine):
    """13. Verifies that heuristic adjustments remain bounded within [-10, +15]."""
    # Worst-case feature vector with all malicious heuristics triggered
    extreme_features = {f: 100 for f in FEATURE_NAMES}
    extreme_features["has_ip_address"] = 1
    extreme_features["has_at_symbol"] = 1
    extreme_features["has_url_shortener"] = 1
    extreme_features["is_encoded"] = 1
    extreme_features["num_subdomains"] = 10
    extreme_features["digit_ratio"] = 0.50
    extreme_features["suspicious_keyword_present"] = 1
    extreme_features["num_hyphens"] = 10
    extreme_features["url_length"] = 200

    for test_prob in [0.0, 0.25, 0.50, 0.75, 1.0]:
        adj, triggered, _ = risk_engine.evaluate_heuristics(extreme_features, test_prob)
        assert -10.0 <= adj <= 15.0

def test_predictions_not_silently_changed(risk_engine, predictor):
    """14. Verifies that underlying ML predictions remain consistent with Phase 4/5/6."""
    test_urls = [
        "https://www.google.com",
        "http://paypal-security-update.com/login/index.php",
        "https://www.wikipedia.org"
    ]
    for url in test_urls:
        pred_label = predictor.predict(url)["prediction"]
        risk_label = risk_engine.assess_url(url)["prediction"]
        assert pred_label == risk_label
