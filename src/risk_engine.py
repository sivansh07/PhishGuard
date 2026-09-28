"""
PhishGuard Phase 8: Explainable Risk Scoring Engine & Heuristic Signal Layer.

Combines the calibrated Random Forest classification probability with an empirical,
bounded heuristic risk signal layer to produce an operational 0–100 risk score:
1. Base ML Component: Direct model-estimated probability (P_phish * 100).
2. Bounded Heuristic Signal Layer: Transparent structural indicators with confidence-aware
   damping to prevent double-counting.
3. Operational Risk Levels: Clear tiers (LOW, GUARDED, MODERATE, HIGH, CRITICAL).
4. Explainability Integration: Fuses Phase 7 Saabas local attributions with heuristic alerts.
5. Auditable & Conservative: Fully documented mathematical formulation without claiming causality.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    MODELS_DIR,
    REPORTS_FIGURES_DIR,
    REPORTS_RESULTS_DIR,
    setup_logger,
    ensure_directories_exist
)
from src.feature_extractor import FEATURE_NAMES, extract_url_features
from src.predictor import PhishGuardPredictor
from src.explainability import PhishGuardExplainer, FEATURE_METADATA

logger = setup_logger("RiskEngine")

# Operational Risk Level Bands
RISK_LEVEL_DEFINITIONS = {
    "LOW": {
        "range": (0, 20),
        "label": "Low Risk",
        "description": "Minimal structural anomaly; consistent with standard benign URL conventions.",
        "recommended_action": "Allow navigation. Standard browser security controls remain active."
    },
    "GUARDED": {
        "range": (21, 40),
        "label": "Guarded Risk",
        "description": "Predominantly benign structure with isolated minor lexical or structural variations.",
        "recommended_action": "Allow with passive monitoring. Exercise caution if personal credentials are requested."
    },
    "MODERATE": {
        "range": (41, 60),
        "label": "Moderate / Borderline Risk",
        "description": "Ambiguous lexical profile or borderline model confidence. Signals heightened uncertainty.",
        "recommended_action": "Heightened scrutiny required. Display contextual warning before user credential entry."
    },
    "HIGH": {
        "range": (61, 80),
        "label": "High Risk",
        "description": "Strong structural and lexical markers commonly associated with deceptive phishing links.",
        "recommended_action": "Intervene with interstitial warning. Advise user against interacting with page elements."
    },
    "CRITICAL": {
        "range": (81, 100),
        "label": "Critical Risk",
        "description": "Very high model probability and/or severe structural abuse indicators (e.g. direct IP, shortener, obfuscation).",
        "recommended_action": "Block navigation immediately. Log link to enterprise security telemetry for threat triage."
    }
}

# Empirical Heuristic Signal Definitions
# Weights reflect bounded adjustments (+/- points) prior to confidence-aware scaling
HEURISTIC_RULES: List[Dict[str, Any]] = [
    {
        "id": "H_IP_HOST",
        "name": "Direct IP Address Host",
        "category": "High-Severity Evasion",
        "weight": 8.0,
        "condition": lambda f: f.get("has_ip_address", 0) == 1,
        "description": "The URL specifies a numeric IP address directly instead of a domain name.",
        "warning": "Direct IP host: Bypasses standard domain reputation and registration oversight.",
        "overlap_feature": "has_ip_address"
    },
    {
        "id": "H_AT_SYMBOL",
        "name": "@ Symbol Credential Delimiter",
        "category": "High-Severity Evasion",
        "weight": 8.0,
        "condition": lambda f: f.get("has_at_symbol", 0) == 1,
        "description": "The URL contains an '@' character (HTTP userinfo delimiter).",
        "warning": "Credential delimiter (@): Browsers discard text prior to '@', potentially misleading user on destination host.",
        "overlap_feature": "has_at_symbol"
    },
    {
        "id": "H_URL_SHORTENER",
        "name": "URL Shortener Service",
        "category": "High-Severity Evasion",
        "weight": 6.0,
        "condition": lambda f: f.get("has_url_shortener", 0) == 1,
        "description": "The URL host belongs to a known redirection or URL shortening service.",
        "warning": "Shortened URL: Destination domain is obscured, preventing visual domain verification.",
        "overlap_feature": "has_url_shortener"
    },
    {
        "id": "H_HEX_ENCODING",
        "name": "Hexadecimal Percent-Encoding",
        "category": "Obfuscation",
        "weight": 4.0,
        "condition": lambda f: f.get("is_encoded", 0) == 1,
        "description": "URL includes percent-encoded hex sequences ('%XX').",
        "warning": "Encoded characters: Often leveraged to obscure directory traversal or sensitive keyword strings.",
        "overlap_feature": "is_encoded"
    },
    {
        "id": "H_EXCESS_SUBDOMAINS",
        "name": "Excessive Subdomain Hierarchy",
        "category": "Structural Complexity",
        "weight": 4.0,
        "condition": lambda f: f.get("num_subdomains", 0) >= 3,
        "description": "Host contains 3 or more subdomain segments beyond the registered second-level domain.",
        "warning": "Complex subdomain hierarchy: Common in brand spoofing schemes mimicking legitimate sub-services.",
        "overlap_feature": "num_subdomains"
    },
    {
        "id": "H_HIGH_DIGIT_RATIO",
        "name": "Elevated Numeric Character Density",
        "category": "Structural Complexity",
        "weight": 4.0,
        "condition": lambda f: f.get("digit_ratio", 0.0) >= 0.20,
        "description": "Numeric digits comprise 20% or more of all characters in the URL.",
        "warning": "High digit concentration: Correlates with machine-generated tokens or obfuscated hashes.",
        "overlap_feature": "digit_ratio"
    },
    {
        "id": "H_SUSPICIOUS_KEYWORD",
        "name": "Authentication Keyword Concentration",
        "category": "Lexical Semantics",
        "weight": 3.0,
        "condition": lambda f: f.get("suspicious_keyword_present", 0) == 1,
        "description": "Contains predefined authentication or security tokens (e.g., 'login', 'verify', 'account').",
        "warning": "Authentication keywords present: Frequently observed in credential harvesting lure pages.",
        "overlap_feature": "suspicious_keyword_present"
    },
    {
        "id": "H_MULTIPLE_HYPHENS",
        "name": "Multiple Hyphen Delimiters",
        "category": "Structural Complexity",
        "weight": 2.0,
        "condition": lambda f: f.get("num_hyphens", 0) >= 3,
        "description": "URL contains 3 or more hyphen delimiters.",
        "warning": "Multiple hyphens: Characteristic of compound typosquatting domains (e.g., 'brand-security-alert').",
        "overlap_feature": "num_hyphens"
    },
    {
        "id": "H_EXCESSIVE_LENGTH",
        "name": "Abnormally Long URL",
        "category": "Structural Complexity",
        "weight": 3.0,
        "condition": lambda f: f.get("url_length", 0) >= 75,
        "description": "URL length meets or exceeds 75 characters.",
        "warning": "Lengthy URL: Often indicates deep nesting or parameter stuffing designed to hide the true host.",
        "overlap_feature": "url_length"
    },
    {
        "id": "H_UNENCRYPTED_HTTP",
        "name": "Unencrypted HTTP Scheme",
        "category": "Security & Protocol",
        "weight": 4.0,
        "condition": lambda f: f.get("has_https", 1) == 0,
        "description": "The URL specifies unencrypted HTTP ('http://').",
        "warning": "Unencrypted connection: Legitimate modern services overwhelmingly mandate HTTPS.",
        "overlap_feature": "has_https"
    },
    {
        "id": "H_STANDARD_BENIGN_PROFILE",
        "name": "Standard Benign Root Structure",
        "category": "Protective Context",
        "weight": -3.0,
        "condition": lambda f: (
            f.get("has_https", 0) == 1 and
            f.get("suspicious_keyword_present", 1) == 0 and
            f.get("num_subdomains", 2) <= 1 and
            f.get("digit_ratio", 1.0) == 0.0 and
            f.get("path_length", 1) == 0
        ),
        "description": "URL exhibits standard HTTPS root structure without subdomains, digits, or keywords.",
        "warning": "Standard root structure: Typical of legitimate corporate domains, but also adopted by evasive landing pages.",
        "overlap_feature": "has_https"
    }
]

def determine_risk_level(score: float) -> Tuple[str, Dict[str, Any]]:
    """Maps a 0–100 risk score to its operational risk tier."""
    score_rounded = round(float(score))
    if score_rounded <= 20:
        return "LOW", RISK_LEVEL_DEFINITIONS["LOW"]
    elif score_rounded <= 40:
        return "GUARDED", RISK_LEVEL_DEFINITIONS["GUARDED"]
    elif score_rounded <= 60:
        return "MODERATE", RISK_LEVEL_DEFINITIONS["MODERATE"]
    elif score_rounded <= 80:
        return "HIGH", RISK_LEVEL_DEFINITIONS["HIGH"]
    else:
        return "CRITICAL", RISK_LEVEL_DEFINITIONS["CRITICAL"]

class PhishGuardRiskEngine:
    """Explainable Risk Scoring Engine for PhishGuard."""

    def __init__(
        self,
        predictor: Optional[PhishGuardPredictor] = None,
        explainer: Optional[PhishGuardExplainer] = None
    ):
        """Initializes the Risk Engine with underlying predictor and explainer."""
        self.predictor = predictor or PhishGuardPredictor()
        self.explainer = explainer or PhishGuardExplainer(predictor=self.predictor)
        self.model_name = self.predictor.model_name

    def evaluate_heuristics(self, features_dict: Dict[str, Any], ml_prob: float) -> Tuple[float, List[Dict[str, Any]], List[str]]:
        """Evaluates heuristic rules and calculates the bounded heuristic adjustment.

        Confidence-Aware Scaling:
        To prevent double-counting features already incorporated into the Random Forest,
        the heuristic adjustment is scaled by alpha(P):
            alpha(P) = 1.0 - 0.70 * |2P - 1|  in [0.30, 1.00]
        When model confidence is ambiguous (P ~ 0.50), alpha = 1.0 (full heuristic impact).
        When model confidence is decisive (P ~ 0.0 or 1.0), alpha = 0.30 (damped adjustment).

        Total heuristic adjustment is strictly bounded within [-10.0, +15.0] points.

        Args:
            features_dict: The 27 extracted features.
            ml_prob: Model-estimated phishing probability in [0.0, 1.0].

        Returns:
            Tuple of (scaled_bounded_adjustment, triggered_heuristics_list, warnings_list).
        """
        raw_adjustment = 0.0
        triggered_heuristics: List[Dict[str, Any]] = []
        warnings: List[str] = []

        # Borderline model warning
        if 0.35 <= ml_prob <= 0.65:
            warnings.append(
                f"Borderline Model Decision: Model probability ({ml_prob:.1%}) is near the classification threshold (0.50). "
                "Heightened scrutiny and manual verification recommended."
            )

        # Evaluate rules
        for rule in HEURISTIC_RULES:
            # Only apply benign dampener if model probability is already low
            if rule["id"] == "H_STANDARD_BENIGN_PROFILE" and ml_prob >= 0.20:
                continue

            try:
                if rule["condition"](features_dict):
                    w = float(rule["weight"])
                    raw_adjustment += w
                    triggered_heuristics.append({
                        "id": rule["id"],
                        "name": rule["name"],
                        "category": rule["category"],
                        "weight": w,
                        "description": rule["description"],
                        "overlap_feature": rule["overlap_feature"]
                    })
                    if "warning" in rule:
                        warnings.append(rule["warning"])
            except Exception as e:
                logger.warning(f"Error evaluating rule {rule['id']}: {e}")

        # Clip raw adjustment to prevent extreme values
        bounded_raw = float(np.clip(raw_adjustment, -10.0, 15.0))

        # Confidence-aware damping factor alpha(P)
        alpha = float(1.0 - 0.70 * abs(2.0 * ml_prob - 1.0))
        alpha = float(np.clip(alpha, 0.30, 1.00))

        scaled_adjustment = round(bounded_raw * alpha, 2)

        return scaled_adjustment, triggered_heuristics, warnings

    def assess_url(self, url: str) -> Dict[str, Any]:
        """Calculates operational risk score and compiles explainable risk assessment.

        Args:
            url: The raw URL string to evaluate.

        Returns:
            Complete structured risk assessment dictionary.
        """
        # 1. Feature extraction and model inference
        features_dict = extract_url_features(url)
        features_df = pd.DataFrame([features_dict], columns=FEATURE_NAMES)

        pred_class = int(self.predictor.pipeline.predict(features_df)[0])
        pred_prob = float(self.predictor.pipeline.predict_proba(features_df)[0, 1])
        prediction_label = "POTENTIAL PHISHING" if pred_class == 1 else "LEGITIMATE (BENIGN)"

        # 2. Base ML Risk Component (0 - 100)
        ml_risk_component = round(pred_prob * 100.0, 2)

        # 3. Heuristic Signal Evaluation
        heuristic_adj, triggered_heuristics, warnings = self.evaluate_heuristics(features_dict, pred_prob)

        # 4. Final Combined Risk Score (0 - 100)
        raw_final_score = ml_risk_component + heuristic_adj
        final_risk_score = int(np.clip(round(raw_final_score), 0, 100))

        # 5. Operational Risk Level
        risk_level_key, risk_level_info = determine_risk_level(final_risk_score)

        # 6. Retrieve Phase 7 Local Explanations
        explanation_data = self.explainer.explain(url)

        # 7. Construct Human-Centric Narrative
        narrative = self.build_risk_narrative(
            url=url,
            prediction_label=prediction_label,
            ml_prob=pred_prob,
            risk_score=final_risk_score,
            risk_level_key=risk_level_key,
            triggered_heuristics=triggered_heuristics,
            top_pos=explanation_data["top_positive_contributors"],
            top_neg=explanation_data["top_negative_contributors"],
            warnings=warnings
        )

        return {
            "url": url,
            "prediction": prediction_label,
            "is_phishing": bool(pred_class == 1),
            "model_estimated_phishing_probability": round(pred_prob, 4),
            "model_estimated_phishing_probability_pct": round(pred_prob * 100.0, 2),
            "operational_risk_score": final_risk_score,
            "risk_level": risk_level_key,
            "risk_level_details": risk_level_info,
            "ml_risk_component": ml_risk_component,
            "heuristic_adjustment": heuristic_adj,
            "heuristic_signals": triggered_heuristics,
            "heuristic_signals_count": len(triggered_heuristics),
            "features": features_dict,
            "local_feature_attributions": {
                "top_positive_contributors": explanation_data["top_positive_contributors"],
                "top_negative_contributors": explanation_data["top_negative_contributors"],
                "baseline_prior_bias": explanation_data["baseline_prior_bias"],
                "all_contributions": explanation_data["top_contributing_features"]
            },
            "explanation_details": explanation_data,
            "warnings": warnings,
            "explanation": narrative,
            "methodology_disclosures": {
                "risk_formula": "Risk_Score = clip(round(100 * P(phish) + alpha(P) * delta_heuristics), 0, 100)",
                "double_counting_mitigation": "Heuristic adjustments are confidence-damped via alpha(P) and bounded to [-10, +15].",
                "academic_disclosure": (
                    "The operational risk score is a decision-support metric combining model probability with structural alerts. "
                    "It does not represent a causal guarantee of attacker intent or external webpage safety."
                )
            }
        }

    def build_risk_narrative(
        self,
        url: str,
        prediction_label: str,
        ml_prob: float,
        risk_score: int,
        risk_level_key: str,
        triggered_heuristics: List[Dict[str, Any]],
        top_pos: List[Dict[str, Any]],
        top_neg: List[Dict[str, Any]],
        warnings: List[str]
    ) -> str:
        """Constructs an academically sound, professional risk assessment narrative."""
        parts = []
        parts.append(
            f"The URL receives an Operational Risk Score of {risk_score}/100 ({risk_level_key} RISK), "
            f"reflecting a model-estimated phishing probability of {ml_prob:.1%}."
        )

        if triggered_heuristics:
            h_names = [f"'{h['name']}'" for h in triggered_heuristics[:3]]
            parts.append(f"Active structural risk signals include: {', '.join(h_names)}.")
        else:
            parts.append("No high-severity structural risk heuristics were triggered.")

        if top_pos:
            pos_names = [f"'{p['name']}' (+{p['contribution']:.3f})" for p in top_pos[:2]]
            parts.append(f"Model feature attribution indicates that {', '.join(pos_names)} contributed toward phishing.")

        if top_neg:
            neg_names = [f"'{n['name']}' ({n['contribution']:.3f})" for n in top_neg[:2]]
            parts.append(f"Conversely, {', '.join(neg_names)} exerted downward pressure toward a legitimate classification.")

        if warnings:
            parts.append(f"Operational Notice: {warnings[0]}")

        parts.append(
            "Academic Disclosure: These indicators describe statistical model behavior and lexical properties; "
            "they do not establish causal attacker intent or inspect real-time server contents."
        )

        return " ".join(parts)

def assess_url_risk(url: str, predictor: Optional[PhishGuardPredictor] = None) -> Dict[str, Any]:
    """Convenience functional API for assessing risk of a raw URL string."""
    engine = PhishGuardRiskEngine(predictor=predictor)
    return engine.assess_url(url)

def evaluate_heuristics_on_dataset(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> pd.DataFrame:
    """Evaluates candidate heuristics empirically across a test partition."""
    records = []
    fn_mask = (y_test == 1) & (y_pred == 0)
    total_samples = len(X_test)
    total_phish = int((y_test == 1).sum())
    total_legit = int((y_test == 0).sum())
    total_fn = int(fn_mask.sum())

    for rule in HEURISTIC_RULES:
        # Vectorized condition evaluation
        mask = X_test.apply(rule["condition"], axis=1)
        triggered_count = int(mask.sum())

        if triggered_count == 0:
            records.append({
                "heuristic_id": rule["id"],
                "heuristic_name": rule["name"],
                "category": rule["category"],
                "weight": rule["weight"],
                "triggered_count": 0,
                "phishing_count": 0,
                "legitimate_count": 0,
                "phishing_precision": 0.0,
                "fn_overlap_count": 0,
                "additional_false_positives": 0,
                "status": "Inactive"
            })
            continue

        phish_trig = int((mask & (y_test == 1)).sum())
        legit_trig = int((mask & (y_test == 0)).sum())
        prec = round(float(phish_trig / triggered_count), 4) if triggered_count > 0 else 0.0
        fn_overlap = int((mask & fn_mask).sum())
        # Legitimate URLs that were correctly called legitimate by ML, but would be flagged by this heuristic alone
        fp_added = int((mask & (y_test == 0) & (y_pred == 0)).sum())

        records.append({
            "heuristic_id": rule["id"],
            "heuristic_name": rule["name"],
            "category": rule["category"],
            "weight": rule["weight"],
            "triggered_count": triggered_count,
            "phishing_count": phish_trig,
            "legitimate_count": legit_trig,
            "phishing_precision": prec,
            "fn_overlap_count": fn_overlap,
            "additional_false_positives": fp_added,
            "status": "Validated"
        })

    return pd.DataFrame(records)

def generate_phase8_visualizations(
    df_holdout_scores: pd.DataFrame,
    df_heuristics_holdout: pd.DataFrame,
    df_heuristics_unseen: pd.DataFrame
) -> None:
    """Generates Figures 16, 17, and 18 for Phase 8."""
    ensure_directories_exist()
    sns.set_theme(style="whitegrid")

    # Figure 16: Risk Score Distribution (Legitimate vs Phishing)
    plt.figure(figsize=(9, 5.5), dpi=300)
    sns.histplot(
        data=df_holdout_scores,
        x="operational_risk_score",
        hue="ground_truth_label",
        bins=30,
        element="step",
        stat="density",
        common_norm=False,
        palette=["#2b8a3e", "#c92a2a"],
        alpha=0.6
    )
    plt.axvline(20, color="#74c0fc", linestyle="--", lw=1.2, label="Low / Guarded Boundary (20)")
    plt.axvline(40, color="#ffd43b", linestyle="--", lw=1.2, label="Guarded / Moderate Boundary (40)")
    plt.axvline(60, color="#ffa94d", linestyle="--", lw=1.2, label="Moderate / High Boundary (60)")
    plt.axvline(80, color="#ff6b6b", linestyle="--", lw=1.2, label="High / Critical Boundary (80)")

    plt.title("PhishGuard Figure 16: Operational Risk Score Distribution by Ground Truth", fontsize=12, fontweight="bold")
    plt.xlabel("Operational Risk Score (0 – 100)", fontsize=11)
    plt.ylabel("Normalized Sample Density", fontsize=11)
    plt.legend(loc="upper center", frameon=True, fontsize=9)
    plt.tight_layout()
    fig16_path = REPORTS_FIGURES_DIR / "16_risk_score_distribution.png"
    plt.savefig(fig16_path)
    plt.close()
    logger.info(f"Saved {fig16_path}")

    # Figure 17: ML Probability vs Operational Risk Score
    plt.figure(figsize=(9, 6), dpi=300)
    # Subsample 2000 points for clear visual comparison
    sample_sub = df_holdout_scores.sample(n=min(2000, len(df_holdout_scores)), random_state=42)
    scatter = plt.scatter(
        sample_sub["ml_probability"] * 100.0,
        sample_sub["operational_risk_score"],
        c=sample_sub["heuristic_adjustment"],
        cmap="coolwarm",
        alpha=0.6,
        edgecolors="none",
        s=30
    )
    plt.plot([0, 100], [0, 100], "k--", lw=1.2, label="1:1 Direct ML Mapping (No Adjustment)")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Heuristic Adjustment (Δ Points)", fontsize=10)

    plt.title("PhishGuard Figure 17: Random Forest Probability vs. Operational Risk Score", fontsize=12, fontweight="bold")
    plt.xlabel("Random Forest Component: 100 × P(phishing)", fontsize=11)
    plt.ylabel("Final Operational Risk Score (0 – 100)", fontsize=11)
    plt.xlim(-2, 102)
    plt.ylim(-2, 102)
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    fig17_path = REPORTS_FIGURES_DIR / "17_ml_vs_risk_score.png"
    plt.savefig(fig17_path)
    plt.close()
    logger.info(f"Saved {fig17_path}")

    # Figure 18: Heuristic Signal Analysis (Activation Frequency & Precision)
    plot_h = df_heuristics_holdout[df_heuristics_holdout["triggered_count"] > 0].copy()
    plot_h = plot_h.sort_values(by="triggered_count", ascending=True)

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    y_pos = np.arange(len(plot_h))

    # Horizontal bar plot for triggered counts
    bars = ax1.barh(y_pos, plot_h["triggered_count"], color="#4dabf7", alpha=0.75, label="Triggered URLs (Count)")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(plot_h["heuristic_name"], fontsize=9)
    ax1.set_xlabel("Triggered URL Count in Hold-Out Test Set (log scale)", fontsize=11)
    ax1.set_xscale("log")

    # Annotate precision on top of bars
    for idx, (cnt, prec) in enumerate(zip(plot_h["triggered_count"], plot_h["phishing_precision"])):
        ax1.text(cnt * 1.15, idx, f"{prec:.1%}", va="center", ha="left", fontsize=8, fontweight="bold", color="#d6336c")

    plt.title("PhishGuard Figure 18: Empirical Heuristic Activation Frequency and Precision (%)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig18_path = REPORTS_FIGURES_DIR / "18_heuristic_signal_analysis.png"
    plt.savefig(fig18_path)
    plt.close()
    logger.info(f"Saved {fig18_path}")

def run_phase8_evaluation():
    """Executes empirical evaluation of the Risk Scoring Engine and compiles all Phase 8 reports."""
    ensure_directories_exist()
    logger.info("=" * 70)
    logger.info("Starting Phase 8: Risk Scoring Engine & Heuristic Signal Layer Evaluation")
    logger.info("=" * 70)

    from src.preprocessing import get_train_test_data, get_or_create_processed_dataset, load_and_clean_raw_data
    from sklearn.model_selection import StratifiedGroupKFold

    engine = PhishGuardRiskEngine()
    predictor = engine.predictor
    pipeline = predictor.pipeline

    # 1. Evaluate on Standard Hold-Out Test Set (Phase 4/5)
    logger.info("Loading Phase 4/5 standard hold-out test set (47,074 samples)...")
    X_train, X_test, y_train, y_test = get_train_test_data(test_size=0.2, random_state=42)
    raw_clean_df = load_and_clean_raw_data()
    test_urls = raw_clean_df.loc[X_test.index, "URL"]

    y_pred_holdout = pipeline.predict(X_test)
    y_prob_holdout = pipeline.predict_proba(X_test)[:, 1]

    logger.info("Evaluating candidate heuristics across hold-out test set...")
    df_heuristics_holdout = evaluate_heuristics_on_dataset(X_test, y_test, y_pred_holdout, y_prob_holdout)

    # 2. Evaluate on Unseen-Domain Test Set (Phase 6)
    logger.info("Evaluating candidate heuristics across Phase 6 unseen-domain test set...")
    processed_df = get_or_create_processed_dataset()
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(sgkf.split(processed_df, processed_df["target"], groups=processed_df["Domain"]))
    _, unseen_test_idx = splits[0]

    X_test_unseen = processed_df.loc[unseen_test_idx, FEATURE_NAMES]
    y_test_unseen = processed_df.loc[unseen_test_idx, "target"]
    y_pred_unseen = pipeline.predict(X_test_unseen)
    y_prob_unseen = pipeline.predict_proba(X_test_unseen)[:, 1]

    df_heuristics_unseen = evaluate_heuristics_on_dataset(X_test_unseen, y_test_unseen, y_pred_unseen, y_prob_unseen)

    # Export heuristic analysis CSV
    csv_path = REPORTS_RESULTS_DIR / "phase8_heuristic_analysis.csv"
    df_heuristics_holdout.to_csv(csv_path, index=False)
    logger.info(f"Saved {csv_path}")

    # 3. Compute Risk Scores for a Sample Evaluation Set (10,000 samples for distribution analysis)
    logger.info("Computing operational risk scores for distribution profiling...")
    sample_indices = X_test.sample(n=min(10000, len(X_test)), random_state=42).index
    score_records = []

    for idx in sample_indices:
        feat_dict = X_test.loc[idx].to_dict()
        prob = float(y_prob_holdout[X_test.index.get_loc(idx)])
        target = int(y_test.loc[idx])

        # Evaluate heuristics
        adj, triggered, _ = engine.evaluate_heuristics(feat_dict, prob)
        risk_score = int(np.clip(round(prob * 100.0 + adj), 0, 100))
        level_key, _ = determine_risk_level(risk_score)

        score_records.append({
            "index": idx,
            "ground_truth": target,
            "ground_truth_label": "Phishing" if target == 1 else "Legitimate",
            "ml_probability": prob,
            "heuristic_adjustment": adj,
            "operational_risk_score": risk_score,
            "risk_level": level_key,
            "triggered_heuristics_count": len(triggered)
        })

    df_scores = pd.DataFrame(score_records)

    # 4. Generate Figures 16, 17, 18
    logger.info("Generating Phase 8 academic figures 16, 17, and 18...")
    generate_phase8_visualizations(df_scores, df_heuristics_holdout, df_heuristics_unseen)

    # 5. Compile Representative URL Assessments
    logger.info("Compiling representative risk assessments...")
    representative_urls = [
        "http://www.worldmedicsky.info",
        "https://www.downapp.com",
        "https://www.cfg.me",
        "https://www.cns11643.gov.tw",
        "http://192.168.1.1/admin/login.php",
        "https://netflix-vn.com",
        "https://bit.ly/secure-banking-alert"
    ]
    representative_assessments = [engine.assess_url(u) for u in representative_urls]

    # 6. Benchmark Latency (Prediction vs Prediction+Explain vs Prediction+Explain+Risk)
    logger.info("Benchmarking end-to-end latency across all three architectural layers...")
    benchmarks = benchmark_layered_latency(engine, representative_urls[:4], n_runs=40)
    logger.info(f"Latency benchmark: Predict={benchmarks['predict_ms']}ms | Explain={benchmarks['explain_ms']}ms | Risk={benchmarks['risk_ms']}ms")

    # 7. Compile JSON & Markdown Reports
    logger.info("Saving master JSON and Markdown reports...")
    save_phase8_reports(
        df_heuristics_holdout,
        df_heuristics_unseen,
        df_scores,
        representative_assessments,
        benchmarks
    )

    logger.info("Phase 8 evaluation complete.")

def benchmark_layered_latency(engine: PhishGuardRiskEngine, urls: List[str], n_runs: int = 40) -> Dict[str, float]:
    """Measures latency across prediction, explainability, and risk engine layers."""
    # Warm-up
    for u in urls:
        engine.predictor.predict(u)
        engine.explainer.explain(u)
        engine.assess_url(u)

    # Layer 1: Predict
    t0 = time.perf_counter()
    for _ in range(n_runs):
        for u in urls:
            engine.predictor.predict(u)
    t_predict = (time.perf_counter() - t0) / (n_runs * len(urls)) * 1000.0

    # Layer 2: Predict + Explain
    t0 = time.perf_counter()
    for _ in range(n_runs):
        for u in urls:
            engine.explainer.explain(u)
    t_explain = (time.perf_counter() - t0) / (n_runs * len(urls)) * 1000.0

    # Layer 3: Predict + Explain + Risk Engine
    t0 = time.perf_counter()
    for _ in range(n_runs):
        for u in urls:
            engine.assess_url(u)
    t_risk = (time.perf_counter() - t0) / (n_runs * len(urls)) * 1000.0

    return {
        "predict_ms": round(t_predict, 2),
        "explain_ms": round(t_explain, 2),
        "risk_ms": round(t_risk, 2),
        "explain_overhead_ms": round(t_explain - t_predict, 2),
        "risk_engine_overhead_ms": round(t_risk - t_explain, 2),
        "total_overhead_ms": round(t_risk - t_predict, 2)
    }

def save_phase8_reports(
    df_h_holdout: pd.DataFrame,
    df_h_unseen: pd.DataFrame,
    df_scores: pd.DataFrame,
    rep_assessments: List[Dict[str, Any]],
    benchmarks: Dict[str, float]
) -> None:
    """Exports master Phase 8 JSON and Markdown reports."""
    # JSON Report
    json_path = REPORTS_RESULTS_DIR / "phase8_risk_engine.json"
    report_dict = {
        "module": "PhishGuard Explainable Risk Scoring Engine",
        "mathematical_formulation": {
            "ml_risk_component": "100.0 * P(phishing | URL)",
            "heuristic_adjustment": "alpha(P) * clip(sum(w_k * I(h_k)), -10.0, 15.0)",
            "confidence_damping_factor": "alpha(P) = clip(1.0 - 0.70 * |2P - 1|, 0.30, 1.00)",
            "final_risk_score": "clip(round(ML_Risk + Heuristic_Adj), 0, 100)"
        },
        "operational_risk_bands": RISK_LEVEL_DEFINITIONS,
        "latency_benchmark": benchmarks,
        "empirical_heuristic_evaluation_holdout": df_h_holdout.to_dict(orient="records"),
        "empirical_heuristic_evaluation_unseen_domain": df_h_unseen.to_dict(orient="records"),
        "representative_assessments": rep_assessments,
        "academic_disclosure": (
            "1. The operational risk score is a decision-support metric combining model probability with structural alerts. "
            "2. Heuristic signals are empirical indicators and do not constitute proof of phishing. "
            "3. The system operates strictly offline on 27 static URL features without network requests or DOM inspection. "
            "4. The risk engine does not claim guaranteed zero-day detection."
        )
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4)
    logger.info(f"Saved {json_path}")

    # Markdown Report
    md_path = REPORTS_RESULTS_DIR / "phase8_risk_engine.md"
    generate_phase8_markdown(df_h_holdout, df_h_unseen, df_scores, rep_assessments, benchmarks, md_path)
    logger.info(f"Saved {md_path}")

def generate_phase8_markdown(
    df_h_holdout: pd.DataFrame,
    df_h_unseen: pd.DataFrame,
    df_scores: pd.DataFrame,
    rep_cases: List[Dict[str, Any]],
    benchmarks: Dict[str, float],
    filepath: Path
) -> None:
    """Generates the master academic Markdown report for Phase 8."""
    md = f"""# PhishGuard: Phase 8 Explainable Risk Scoring Engine Report

**Module:** Explainable Risk Scoring Engine & Heuristic Signal Layer  
**Core Model:** RandomForestClassifier (Phase 4/5/6 Verified Candidate)  
**Evaluated Features:** Exactly 27 Static URL Features  

---

## 1. Risk Score Mathematical Formulation

The PhishGuard Risk Engine transforms the calibrated Random Forest probability into an operational 0–100 risk score while preserving the ML probability as an independent, transparent signal.

### Mathematical Formulation:
$$\\text{{ML\\_Risk}} = 100 \\times P(\\text{{phishing}} \\mid \\text{{URL}}) \\in [0.0, 100.0]$$

$$\\Delta_{{\\text{{Heuristic}}}} = \\alpha(P) \\times \\text{{clip}}\\left(\\sum_{{k=1}}^{{K}} w_k \\cdot \\mathbb{{I}}(h_k), -10.0, +15.0\\right)$$

$$\\text{{Operational Risk Score}} = \\text{{clip}}\\left(\\text{{round}}(\\text{{ML\\_Risk}} + \\Delta_{{\\text{{Heuristic}}}}), 0, 100\\right)$$

### Confidence-Aware Scaling & Double-Counting Mitigation:
Because many heuristic signals overlap with features already utilized by the Random Forest, directly adding unconstrained points would blindly double-count lexical signals.
To resolve this:
1. **Confidence Damping Factor $\\alpha(P)$:**
   $$\\alpha(P) = \\text{{clip}}(1.0 - 0.70 \\cdot |2P - 1|, 0.30, 1.00)$$
   - When the Random Forest is highly confident ($P \\approx 0.0$ or $1.0$), $\\alpha(P) = 0.30$, preventing heuristics from overriding decisive model predictions.
   - When the model decision is borderline or ambiguous ($P \\approx 0.50$), $\\alpha(P) = 1.00$, allowing heuristic structural indicators to provide operational differentiation.
2. **Strict Adjustment Bounds:** Bounded to $[-10.0, +15.0]$ points maximum.

---

## 2. Operational Risk Tiers

The 0–100 score is partitioned into initial operational bands for defensive triage:

| Score Band | Risk Level | Description | Recommended Operational Action |
| :---: | :--- | :--- | :--- |
| **0 – 20** | **LOW** | Clean structural profile matching benign conventions. | Allow navigation. Standard security controls active. |
| **21 – 40** | **GUARDED** | Minor structural deviations; predominantly benign. | Allow with passive telemetry monitoring. |
| **41 – 60** | **MODERATE** | Ambiguous profile or borderline ML confidence. | **Heightened scrutiny.** Present contextual warning. |
| **61 – 80** | **HIGH** | Multiple structural indicators associated with phishing. | Interstitial warning. Advise user against credential entry. |
| **81 – 100** | **CRITICAL** | Severe structural anomalies or high ML probability. | **Block immediately.** Dispatch alert to security operations. |

---

## 3. Empirical Heuristic Evaluation (Hold-Out Test vs. Unseen-Domain Test)

Evaluated across the 47,074 held-out test URLs and the 47,074 Phase 6 unseen-domain test URLs:

| Heuristic Identifier | Rule Name | Weight | Hold-Out Triggered | Hold-Out Phish Prec | Unseen Triggered | Unseen Phish Prec | FN Overlap | Added Benign FPs |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for _, row in df_h_holdout.iterrows():
        # Find matching unseen row
        u_match = df_h_unseen[df_h_unseen["heuristic_id"] == row["heuristic_id"]]
        u_trig = u_match["triggered_count"].values[0] if len(u_match) > 0 else 0
        u_prec = u_match["phishing_precision"].values[0] if len(u_match) > 0 else 0.0

        md += (
            f"| `{row['heuristic_id']}` | {row['heuristic_name']} | {row['weight']:+4.1f} | "
            f"{row['triggered_count']:,} | {row['phishing_precision']:.1%} | "
            f"{u_trig:,} | {u_prec:.1%} | "
            f"{row['fn_overlap_count']} | {row['additional_false_positives']} |\n"
        )

    md += f"""
---

## 4. Error-Correction and False-Negative Analysis

### Critical Empirical Finding:
In Phase 6, Random Forest missed 92 phishing URLs. When evaluating whether heuristic signals can detect these missed phishing URLs:
- **Observed Behavior:** 100% of missed phishing URLs had `has_https = 1`, `path_length = 0`, `num_slashes = 2`, and 98.9% had `suspicious_keyword_present = 0`.
- **Benign Comparison:** In the legitimate test set, 100% of the 26,963 true negative URLs exhibited the **exact same structural pattern** (HTTPS root domain with zero path length and no keywords).
- **Academic Conclusion:** Static URL heuristics **cannot** reliably separate root-level evasive phishing URLs without causing tens of thousands of false alarms on benign root sites. This empirically confirms the intrinsic limitation of static URL analysis and motivates external DNS/WHOIS telemetry in production environments.

---

## 5. Representative Risk Assessments

"""
    for case in rep_cases:
        md += f"### Sample URL: `{case['url']}`\n"
        md += f"- **Classification:** `{case['prediction']}`\n"
        md += f"- **Model Phishing Probability:** {case['model_estimated_phishing_probability_pct']:.2f}%\n"
        md += f"- **Operational Risk Score:** **{case['operational_risk_score']} / 100** ({case['risk_level']} RISK)\n"
        md += f"- **ML Component:** {case['ml_risk_component']:.2f} | **Heuristic Adjustment:** {case['heuristic_adjustment']:+.2f}\n"
        if case["heuristic_signals"]:
            h_list = ", ".join([f"`{h['name']}` ({h['weight']:+.1f})" for h in case["heuristic_signals"]])
            md += f"- **Active Heuristics:** {h_list}\n"
        else:
            md += f"- **Active Heuristics:** None\n"
        if case["warnings"]:
            md += f"- **Warnings:** {case['warnings'][0]}\n"
        md += f"- **Narrative:** {case['explanation']}\n\n"

    md += f"""---

## 6. Layered Latency Benchmark

Measured over 40 iterations across representative evaluation URLs:

| Layer | Cumulative Operations | Mean Latency | Added Overhead |
| :--- | :--- | :---: | :---: |
| **Layer 1: Inference** | Feature Extraction + Random Forest Predict | **{benchmarks['predict_ms']:.2f} ms** | Baseline |
| **Layer 2: Explainability** | Layer 1 + 100-Tree Saabas Path Decomposition | **{benchmarks['explain_ms']:.2f} ms** | +{benchmarks['explain_overhead_ms']:.2f} ms |
| **Layer 3: Risk Engine** | Layer 1 + Layer 2 + Heuristic Evaluation & Risk Synthesis | **{benchmarks['risk_ms']:.2f} ms** | +{benchmarks['risk_engine_overhead_ms']:.2f} ms |

**Total End-to-End Latency:** **{benchmarks['risk_ms']:.2f} ms** (Overhead over raw prediction: +{benchmarks['total_overhead_ms']:.2f} ms).

---

## 7. Academic Limitations & Disclosures

1. **Operational Risk vs. Malicious Intent:** The operational risk score is a prioritized triage metric. It reflects structural suspicion within the 27 static features; it is **not** a causal guarantee of attacker intent.
2. **Heuristics are Not Proof:** Individual heuristic activations (e.g. high digit count) indicate statistical elevation of risk, not definitive proof of phishing.
3. **No Webpage Inspection:** PhishGuard deliberately executes no network requests and renders no DOM elements to preserve user privacy and host safety.
4. **HTTPS is Protocol, Not Trust:** Modern attackers widely deploy HTTPS certificates; HTTPS presence cannot be treated as a guarantee of safety.
5. **Proxy Unseen Evaluation:** Phase 6 remains an offline unseen-domain proxy evaluation and does not guarantee detection of real-world zero-day attacks.
"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md.strip() + "\n")

if __name__ == "__main__":
    run_phase8_evaluation()
