"""
Unit and Integration Tests for PhishGuard Streamlit Application (Phase 9).

Verifies:
1. Application module imports cleanly without missing dependencies.
2. Risk tier color mapping returns valid CSS hex colors for all 5 tiers.
3. Prediction and risk tier badge HTML generation is correct.
4. Engine caching and model artifact loading operates properly.
5. Global feature importance loader returns exactly 27 ranked features.
6. All pre-configured benchmark demo URLs execute successfully through the engine.
7. plot_local_attributions produces a valid matplotlib Figure.
8. Application helpers execute in strictly offline mode with zero network requests.
9. Malformed input strings are handled gracefully without raising unhandled exceptions.
"""

import pytest
import sys
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import (
    get_risk_tier_color,
    get_prediction_badge_html,
    get_risk_tier_badge_html,
    plot_local_attributions,
    load_risk_engine,
    load_global_importance_df,
    DEMO_URLS
)
from src.feature_extractor import FEATURE_NAMES

def test_risk_tier_color_mapping():
    """1. Verifies that valid hex colors are returned for all 5 operational risk tiers."""
    for tier in ["LOW", "GUARDED", "MODERATE", "HIGH", "CRITICAL"]:
        color = get_risk_tier_color(tier)
        assert color.startswith("#")
        assert len(color) == 7

def test_prediction_badge_html():
    """2. Verifies HTML badge output for phishing and legitimate predictions."""
    phish_badge = get_prediction_badge_html("POTENTIAL PHISHING")
    legit_badge = get_prediction_badge_html("LEGITIMATE (BENIGN)")

    assert "POTENTIAL PHISHING" in phish_badge
    assert "LEGITIMATE (BENIGN)" in legit_badge
    assert "<span" in phish_badge and "</span>" in phish_badge
    assert "<span" in legit_badge and "</span>" in legit_badge

def test_risk_tier_badge_html():
    """3. Verifies risk tier badge formatting."""
    for tier in ["LOW", "GUARDED", "MODERATE", "HIGH", "CRITICAL"]:
        badge = get_risk_tier_badge_html(tier)
        assert f"{tier} RISK" in badge
        assert "<span" in badge

def test_load_risk_engine():
    """4. Verifies cached loader returns a functional PhishGuardRiskEngine instance."""
    engine = load_risk_engine()
    assert hasattr(engine, "assess_url")
    assert hasattr(engine, "predictor")
    assert hasattr(engine, "explainer")

def test_load_global_importance_df():
    """5. Verifies global importance loader returns 27 features."""
    df_imp = load_global_importance_df()
    assert len(df_imp) == 27
    assert "feature" in df_imp.columns
    assert "importance" in df_imp.columns
    assert set(df_imp["feature"]) == set(FEATURE_NAMES)

def test_demo_benchmark_urls(tmp_path):
    """6. Verifies that all pre-configured demo benchmark URLs evaluate cleanly."""
    engine = load_risk_engine()
    for label, url in DEMO_URLS.items():
        if not url:
            continue
        assessment = engine.assess_url(url)
        assert "operational_risk_score" in assessment
        assert 0 <= assessment["operational_risk_score"] <= 100
        assert assessment["risk_level"] in ["LOW", "GUARDED", "MODERATE", "HIGH", "CRITICAL"]
        assert "features" in assessment
        assert len(assessment["features"]) == 27

def test_plot_local_attributions():
    """7. Verifies that local attribution plotting produces a valid matplotlib Figure."""
    engine = load_risk_engine()
    assessment = engine.assess_url("http://paypal-security-update.com/login/index.php")
    all_contribs = assessment["local_feature_attributions"]["all_contributions"]
    bias = assessment["local_feature_attributions"]["baseline_prior_bias"]

    fig = plot_local_attributions(all_contribs, bias)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_app_offline_execution(monkeypatch):
    """8. Verifies that application engine functions offline without network requests."""
    import socket
    def fail_socket(*args, **kwargs):
        raise RuntimeError("Network socket call attempted in Streamlit offline engine!")

    monkeypatch.setattr(socket, "socket", fail_socket)

    engine = load_risk_engine()
    res = engine.assess_url("https://www.example.org/path/to/test")
    assert res["operational_risk_score"] >= 0

def test_app_error_handling_graceful():
    """9. Verifies that malformed input strings do not raise uncaught exceptions."""
    engine = load_risk_engine()
    edge_cases = ["", "   ", "http://", "not_a_valid_url", "https://%ZZ%invalid"]
    for u in edge_cases:
        res = engine.assess_url(u)
        assert "operational_risk_score" in res
        assert 0 <= res["operational_risk_score"] <= 100
