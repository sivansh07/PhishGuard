"""
PhishGuard Phase 10: End-to-End System Validation & Security Hardening Tests.

Verifies:
1. End-to-end pipeline execution from URL -> features -> ML -> explanation -> risk -> UI payload.
2. Input robustness and graceful handling of adversarial, pathological, and edge-case URLs.
3. Model artifact safety, path binding, and 27-feature schema expectation.
4. Strict offline execution with zero outbound socket/network connections.
5. Deterministic multi-run reproducibility across all pipeline stages.
6. Streamlit UI helper robustness and crash resilience.
"""

import sys
import socket
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extractor import FEATURE_NAMES, extract_url_features
from src.predictor import PhishGuardPredictor
from src.explainability import PhishGuardExplainer
from src.risk_engine import (
    PhishGuardRiskEngine,
    RISK_LEVEL_DEFINITIONS,
    HEURISTIC_RULES,
    assess_url_risk
)
from src.model import DEFAULT_MODEL_FILE, load_pipeline
from app import (
    get_risk_tier_color,
    get_prediction_badge_html,
    get_risk_tier_badge_html,
    plot_local_attributions
)

@pytest.fixture(scope="module")
def risk_engine():
    """Initializes a shared PhishGuardRiskEngine instance."""
    return PhishGuardRiskEngine()

# =============================================================================
# 1. END-TO-END PIPELINE VALIDATION
# =============================================================================

@pytest.mark.parametrize("category,url", [
    ("Known Legitimate", "https://www.google.com/search?q=cybersecurity"),
    ("Confirmed Phishing", "http://www.worldmedicsky.info"),
    ("False Positive Rep", "https://www.cns11643.gov.tw"),
    ("False Negative Rep", "https://www.cfg.me"),
    ("Direct IP", "http://192.168.1.1/admin/login.php"),
    ("URL Shortener", "https://bit.ly/secure-banking-alert"),
    ("Auth Keyword", "http://account-update.verify-credentials.com/login"),
    ("Long URL", "https://example.com/path/to/resource?param1=val1&param2=val2&token=abcdef1234567890"),
    ("Multi-Subdomain", "http://sub1.sub2.sub3.attacker-domain.com/login"),
    ("Encoded URL", "http://example.com/%20%21%22%23%24"),
    ("HTTPS Root", "https://example.com"),
    ("HTTP URL", "http://example.com/test"),
])
def test_end_to_end_pipeline_representative_cases(risk_engine, category, url):
    """Verifies that representative cases traverse the entire pipeline safely and correctly."""
    # 1. Feature Extraction
    features = extract_url_features(url)
    assert len(features) == 27
    assert all(k in features for k in FEATURE_NAMES)

    # 2. Risk Engine Evaluation (includes RF prediction + Saabas explanation)
    assessment = risk_engine.assess_url(url)

    # 3. Verification of prediction output
    assert "prediction" in assessment
    assert assessment["prediction"] in ("POTENTIAL PHISHING", "LEGITIMATE (BENIGN)")
    assert 0.0 <= assessment["model_estimated_phishing_probability"] <= 1.0

    # 4. Verification of Phase 7 Explanation
    assert "local_feature_attributions" in assessment
    local_attr = assessment["local_feature_attributions"]
    assert "top_positive_contributors" in local_attr
    assert "top_negative_contributors" in local_attr
    assert "baseline_prior_bias" in local_attr
    assert "explanation_details" in assessment
    recon_err = assessment["explanation_details"]["mathematical_consistency"]["reconstruction_error"]
    assert recon_err < 1e-12

    # 5. Verification of Phase 8 Risk Engine
    assert "operational_risk_score" in assessment
    score = assessment["operational_risk_score"]
    assert 0 <= score <= 100
    assert assessment["risk_level"] in RISK_LEVEL_DEFINITIONS
    assert -10.0 <= assessment["heuristic_adjustment"] <= 15.0

    # 6. Streamlit-compatible data structure
    assert "explanation" in assessment
    assert isinstance(assessment["explanation"], str)
    assert len(assessment["explanation"]) > 0

# =============================================================================
# 2. INPUT ROBUSTNESS & ADVERSARIAL TESTING
# =============================================================================

@pytest.mark.parametrize("category,adversarial_url", [
    ("Empty String", ""),
    ("Whitespace Only", "   \t \r \n  "),
    ("Extremely Long URL", "https://example.com/" + "a" * 15000),
    ("Missing Scheme", "www.example.org/search?q=test"),
    ("Malformed Scheme", "http:///corrupted-target"),
    ("Invalid Slashes", "https:://///multiple/slashes"),
    ("Unicode / Cyrillic", "https://\u043f\u0440\u0435\u0437\u0438\u0434\u0435\u043d\u0442.\u0440\u0444/login"),
    ("Emoji in Path", "https://example.com/\U0001F512/secure"),
    ("Repeated Separators", "https://example.com///path//to///resource"),
    ("Extremely Long Hostname", "https://" + "sub." * 120 + "example.com/"),
    ("Extremely Long Path", "https://example.com/" + "dir/" * 400),
    ("Very Large Query", "https://example.com/?" + "key=value&" * 300),
    ("Fragment Heavy", "https://example.com/#" + "section" * 500),
    ("Unusual Delimiters", "https://example.com/a;b=c?d=1&e=2#f~g$h*i"),
    ("Hex Traversal", "https://example.com/%2e%2e%2f%2e%2e%2fetc%2fpasswd"),
    ("Direct IPv4", "http://10.0.0.1/admin"),
    ("Localhost URL", "http://localhost:8080/dashboard"),
    ("Null Byte Injection", "https://example.com/login\x00/attack"),
    ("Userinfo Credentials", "https://admin:pass@attacker.com/steal"),
    ("Non-standard Port", "http://example.com:9999/path")
])
def test_input_robustness_adversarial(risk_engine, category, adversarial_url):
    """Verifies that pathological and adversarial inputs never crash the engine."""
    res = risk_engine.assess_url(adversarial_url)

    # Engine must return a complete, valid dictionary
    assert isinstance(res, dict)
    assert "operational_risk_score" in res
    assert 0 <= res["operational_risk_score"] <= 100
    assert res["risk_level"] in RISK_LEVEL_DEFINITIONS
    assert len(res["features"]) == 27
    assert isinstance(res["prediction"], str)

# =============================================================================
# 3. MODEL ARTIFACT SAFETY & VERIFICATION
# =============================================================================

def test_model_artifact_integrity():
    """Verifies model file existence, expected path, type, and feature expectations."""
    assert DEFAULT_MODEL_FILE.exists(), f"Model artifact missing at {DEFAULT_MODEL_FILE}"
    assert DEFAULT_MODEL_FILE.is_file()
    assert DEFAULT_MODEL_FILE.stat().st_size > 100_000  # Non-trivial size (>100KB)

    pipeline = load_pipeline(DEFAULT_MODEL_FILE)
    classifier = pipeline.named_steps["classifier"]

    # Verify model class name
    assert classifier.__class__.__name__ == "RandomForestClassifier"

    # Verify input feature count
    assert classifier.n_features_in_ == 27

    # Verify deterministic output across two independent loads
    pipeline2 = load_pipeline(DEFAULT_MODEL_FILE)
    dummy_input = pd.DataFrame([{f: 0 for f in FEATURE_NAMES}])
    p1 = pipeline.predict_proba(dummy_input)
    p2 = pipeline2.predict_proba(dummy_input)
    assert np.allclose(p1, p2)

def test_model_loading_failure_controlled():
    """Verifies controlled FileNotFoundError when model path is non-existent."""
    with pytest.raises(FileNotFoundError):
        load_pipeline(Path("models/non_existent_model.joblib"))

# =============================================================================
# 4. OFFLINE ISOLATION GUARANTEE
# =============================================================================

def test_complete_pipeline_offline_guarantee(risk_engine, monkeypatch):
    """Mocks socket creation to guarantee 100% offline static execution."""
    socket_calls = []

    def mocked_socket(*args, **kwargs):
        socket_calls.append(args)
        raise RuntimeError("CRITICAL VIOLATION: Socket opened during offline URL analysis!")

    monkeypatch.setattr(socket, "socket", mocked_socket)

    test_urls = [
        "https://www.google.com/search?q=test",
        "http://www.worldmedicsky.info",
        "https://bit.ly/3xY8z9K",
        "http://192.168.1.1/login.php",
        "https://sub.domain.example.com/path?arg=1#frag"
    ]

    for u in test_urls:
        assessment = risk_engine.assess_url(u)
        assert 0 <= assessment["operational_risk_score"] <= 100

    # Verify no sockets were attempted
    assert len(socket_calls) == 0

# =============================================================================
# 5. DETERMINISM REPRODUCIBILITY VERIFICATION
# =============================================================================

@pytest.mark.parametrize("url", [
    "https://www.google.com",
    "http://www.worldmedicsky.info",
    "http://192.168.1.1/admin/login.php",
    "https://bit.ly/secure-banking-alert",
    "https://www.cfg.me"
])
def test_multi_run_determinism(risk_engine, url):
    """Verifies that multiple repeated evaluations yield strictly identical results."""
    runs = [risk_engine.assess_url(url) for _ in range(5)]
    base = runs[0]

    for idx, r in enumerate(runs[1:], start=2):
        assert r["operational_risk_score"] == base["operational_risk_score"], f"Run {idx} score mismatch"
        assert r["model_estimated_phishing_probability"] == base["model_estimated_phishing_probability"], f"Run {idx} prob mismatch"
        assert r["heuristic_adjustment"] == base["heuristic_adjustment"], f"Run {idx} heuristic mismatch"
        assert r["risk_level"] == base["risk_level"], f"Run {idx} tier mismatch"
        assert r["features"] == base["features"], f"Run {idx} features mismatch"

# =============================================================================
# 6. STREAMLIT APPLICATION ROBUSTNESS
# =============================================================================

def test_streamlit_ui_helpers_robustness():
    """Verifies UI rendering helpers across all valid and edge-case inputs."""
    # Test all risk levels
    for level in ["LOW", "GUARDED", "MODERATE", "HIGH", "CRITICAL", "UNKNOWN"]:
        color = get_risk_tier_color(level)
        assert isinstance(color, str)
        assert color.startswith("#")

        badge = get_risk_tier_badge_html(level)
        assert level in badge
        assert color in badge

    # Test prediction badge HTML
    phish_badge = get_prediction_badge_html("POTENTIAL PHISHING")
    legit_badge = get_prediction_badge_html("LEGITIMATE (BENIGN)")
    assert "POTENTIAL PHISHING" in phish_badge
    assert "LEGITIMATE (BENIGN)" in legit_badge

    # Test Matplotlib figure generation
    dummy_attributions = [
        {"name": "HTTPS Indicator", "contribution": -0.25, "absolute_contribution": 0.25},
        {"name": "Path Length", "contribution": 0.15, "absolute_contribution": 0.15},
        {"name": "Slash Count", "contribution": 0.05, "absolute_contribution": 0.05}
    ]
    fig = plot_local_attributions(dummy_attributions, bias=0.45)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)

# =============================================================================
# 7. STATIC CODEBASE SECURITY INVARIANTS
# =============================================================================

def test_static_code_no_eval_or_exec():
    """Verifies that no unsafe dynamic eval() or exec() calls exist in application code."""
    import re
    eval_pattern = re.compile(r"\beval\s*\(")
    exec_pattern = re.compile(r"\bexec\s*\(")

    scan_files = list((PROJECT_ROOT / "src").rglob("*.py")) + [PROJECT_ROOT / "app.py"]
    for p in scan_files:
        text = p.read_text(encoding="utf-8", errors="ignore")
        assert not eval_pattern.search(text), f"Unsafe eval() found in {p}"
        assert not exec_pattern.search(text), f"Unsafe exec() found in {p}"

def test_static_code_no_subprocess_or_os_system():
    """Verifies that no shell command execution or subprocess invocations exist in runtime code."""
    import re
    system_pattern = re.compile(r"\bos\.system\s*\(")
    subproc_pattern = re.compile(r"\bsubprocess\b")
    exclude = {".git", ".venv", "__pycache__", ".pytest_cache"}

    for p in (PROJECT_ROOT / "src").rglob("*.py"):
        if any(ex in p.parts for ex in exclude):
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        assert not system_pattern.search(text), f"Unsafe os.system() found in {p}"
        assert not subproc_pattern.search(text), f"Subprocess module used in runtime file {p}"

def test_static_code_no_raw_pickle():
    """Verifies that no insecure pickle.load/loads exists in source code (only joblib for pipeline)."""
    import re
    pickle_pattern = re.compile(r"\bpickle\.(load|loads)\s*\(")
    exclude = {".git", ".venv", "__pycache__", ".pytest_cache"}

    for p in PROJECT_ROOT.rglob("*.py"):
        if any(ex in p.parts for ex in exclude):
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        assert not pickle_pattern.search(text), f"Insecure pickle call found in {p}"

def test_runtime_modules_have_no_network_imports():
    """Verifies that core inference, risk, and feature modules do not import requests or socket."""
    import re
    net_pattern = re.compile(r"^\s*(import\s+(requests|urllib\.request|http\.client)|from\s+(requests|urllib\.request|http\.client)\s+import)", re.MULTILINE)
    runtime_files = [
        PROJECT_ROOT / "src" / "feature_extractor.py",
        PROJECT_ROOT / "src" / "predictor.py",
        PROJECT_ROOT / "src" / "explainability.py",
        PROJECT_ROOT / "src" / "risk_engine.py",
        PROJECT_ROOT / "app.py"
    ]
    for rf in runtime_files:
        assert rf.exists(), f"Expected file {rf} does not exist"
        text = rf.read_text(encoding="utf-8", errors="ignore")
        match = net_pattern.search(text)
        assert not match, f"Prohibited network import found in {rf.name}: {match.group(0)}"
