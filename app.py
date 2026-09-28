"""
PhishGuard: Explainable Machine Learning-Based Phishing URL Detection and Risk Analysis System.
Streamlit Web Application Entrypoint (Phase 9).

Provides a cybersecurity analysis dashboard:
- Canonical 27 static URL feature extraction (zero network requests, strictly offline)
- Random Forest phishing probability inference
- Phase 7 Saabas decision-path local feature attributions
- Phase 8 confidence-damped operational risk scoring (0–100) and risk tiers
- Transparent explainability narrative and academic disclosures
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    MODELS_DIR,
    REPORTS_FIGURES_DIR,
    REPORTS_RESULTS_DIR
)
from src.feature_extractor import (
    FEATURE_NAMES,
    extract_url_features,
    normalize_url,
    is_apex_domain_without_www
)
from src.explainability import (
    PhishGuardExplainer,
    FEATURE_METADATA,
    get_global_feature_importance
)
from src.risk_engine import (
    PhishGuardRiskEngine,
    RISK_LEVEL_DEFINITIONS,
    HEURISTIC_RULES
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PhishGuard — Explainable URL Phishing Risk Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Cached Resources
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading PhishGuard Pipeline Artifacts...")
def load_risk_engine() -> PhishGuardRiskEngine:
    """Loads and caches the complete PhishGuard Predictor, Explainer, and Risk Engine."""
    return PhishGuardRiskEngine()

@st.cache_data
def load_global_importance_df() -> pd.DataFrame:
    """Loads cached global feature importance table."""
    return get_global_feature_importance()

# -----------------------------------------------------------------------------
# Benchmark Sample URLs for Demonstrations
# -----------------------------------------------------------------------------
DEMO_URLS = {
    "Select a benchmark sample...": "",
    "🔴 Confirmed Phishing (Unencrypted HTTP Lure)": "http://www.worldmedicsky.info",
    "🔴 Phishing (Direct IP Host Evasion)": "http://192.168.1.1/admin/login.php",
    "🔴 Phishing (Shortened URL Redirection)": "https://bit.ly/secure-banking-alert",
    "🟡 Evasive Phishing (Root HTTPS Evasion - Test FN)": "https://www.cfg.me",
    "🟡 Legitimate Multi-Digit (Government Repository - Test FP)": "https://www.cns11643.gov.tw",
    "🟢 Confirmed Legitimate (Clean Root HTTPS)": "https://www.downapp.com",
    "🟢 Confirmed Legitimate (Standard Portal)": "https://www.wikipedia.org"
}

# -----------------------------------------------------------------------------
# Custom Styling Helpers
# -----------------------------------------------------------------------------
def get_risk_tier_color(risk_level: str) -> str:
    colors = {
        "LOW": "#2b8a3e",        # Green
        "GUARDED": "#1c7ed6",    # Blue
        "MODERATE": "#f59f00",   # Amber / Orange
        "HIGH": "#e8590c",       # Deep Orange
        "CRITICAL": "#c92a2a"    # Crimson Red
    }
    return colors.get(risk_level, "#495057")

def get_prediction_badge_html(prediction: str) -> str:
    if "PHISHING" in prediction.upper():
        return (
            '<span style="background-color: #ffe3e3; color: #c92a2a; border: 1px solid #ffa8a8; '
            'padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 1.1rem; '
            'display: inline-block;">⚠️ POTENTIAL PHISHING</span>'
        )
    else:
        return (
            '<span style="background-color: #d3f9d8; color: #2b8a3e; border: 1px solid #8ce99a; '
            'padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 1.1rem; '
            'display: inline-block;">✅ LEGITIMATE (BENIGN)</span>'
        )

def get_risk_tier_badge_html(risk_level: str) -> str:
    color = get_risk_tier_color(risk_level)
    return (
        f'<span style="background-color: {color}; color: white; '
        f'padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 1.1rem; '
        f'display: inline-block;">{risk_level} RISK</span>'
    )

# -----------------------------------------------------------------------------
# Plotting Helpers
# -----------------------------------------------------------------------------
def plot_local_attributions(attributions: List[Dict[str, Any]], bias: float) -> plt.Figure:
    """Generates horizontal bar chart of local feature contributions."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)

    # Sort top 10 features by magnitude
    top_items = sorted(attributions, key=lambda x: x["absolute_contribution"], reverse=True)[:10]
    top_items = top_items[::-1]  # reverse for bottom-to-top rendering

    names = [f["name"] for f in top_items]
    values = [f["contribution"] for f in top_items]
    colors = ["#c92a2a" if v > 0 else "#2b8a3e" for v in values]

    bars = ax.barh(names, values, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="#495057", linestyle="-", lw=1.0)

    # Annotate values
    for bar, val in zip(bars, values):
        x_pos = bar.get_width()
        ha = "left" if x_pos >= 0 else "right"
        offset = 0.005 if x_pos >= 0 else -0.005
        ax.text(x_pos + offset, bar.get_y() + bar.get_height() / 2, f"{val:+.3f}",
                va="center", ha=ha, fontsize=8, fontweight="bold", color="#333333")

    ax.set_xlabel("Local Probability Shift ΔP (Saabas Attribution)", fontsize=10, fontweight="bold")
    ax.set_title(f"Top Local Feature Contributions (Ensemble Prior Bias: {bias:.1%})", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig

# -----------------------------------------------------------------------------
# Main Application Layout
# -----------------------------------------------------------------------------
def main():
    # Load Risk Engine
    try:
        engine = load_risk_engine()
    except Exception as e:
        st.error(f"Failed to load PhishGuard model artifacts: {e}")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.markdown("## 🛡️ PhishGuard System")
        st.caption("Explainable ML Phishing URL Detection & Risk Analysis")
        st.markdown("---")

        st.markdown("### 📊 Architecture & State")
        st.markdown(
            "- **Model:** `RandomForestClassifier`\n"
            "- **Features:** 27 Static URL Features\n"
            "- **Evaluation:** 5-Fold Stratified Group K-Fold\n"
            "- **Domain Leakage:** Strict 0.00% Domain Overlap\n"
            "- **Explainability:** Saabas Tree Decomposition\n"
            "- **Execution Mode:** Strictly Local / Offline\n"
            "- **Automated Tests:** 112/112 Passing"
        )
        st.markdown("---")

        st.markdown("### 🚦 Risk Tier Legend")
        for key, info in RISK_LEVEL_DEFINITIONS.items():
            col_dot = get_risk_tier_color(key)
            st.markdown(
                f"<span style='color:{col_dot}; font-weight:bold;'>● {key} ({info['range'][0]}–{info['range'][1]}):</span> "
                f"<small>{info['description']}</small>",
                unsafe_allow_html=True
            )
        st.markdown("---")

        st.markdown("### ⚠️ Security Notice")
        st.info(
            "PhishGuard evaluates raw URL strings safely in offline mode. "
            "No HTTP requests, DNS lookups, or webpage renderings occur."
        )

    # Main Header
    st.title("🛡️ PhishGuard: Explainable URL Phishing Risk Analyzer")
    st.markdown(
        "**An Explainable Machine Learning-Based Phishing URL Detection and Risk Analysis System**  \n"
        "*Zero Network Access • 27 Lexical & Structural Features • Exact Tree Attribution • Bounded Operational Risk*"
    )
    st.markdown("---")

    # URL Input Section
    col_input, col_demo = st.columns([3, 2])

    with col_demo:
        demo_choice = st.selectbox(
            "Quick Demo: Select Benchmark Sample URL",
            options=list(DEMO_URLS.keys()),
            index=0
        )

    with col_input:
        default_val = DEMO_URLS[demo_choice] if demo_choice != "Select a benchmark sample..." else ""
        url_input = st.text_input(
            "Enter URL for Static Security & Risk Analysis:",
            value=default_val,
            placeholder="e.g. https://www.example.com/login"
        )

    col_btn, col_hint = st.columns([1, 4])
    with col_btn:
        analyze_clicked = st.button("🔍 Analyze URL Risk", type="primary", use_container_width=True)
    with col_hint:
        st.caption("Press button or Enter to extract features, compute model probability, and generate local attribution.")

    # Execution Trigger
    if analyze_clicked or (url_input and url_input.strip()):
        input_clean = url_input.strip()

        norm_url, was_schemeless = normalize_url(input_clean)
        is_apex = is_apex_domain_without_www(input_clean)

        if was_schemeless:
            st.info(
                f"ℹ️ **Schemeless Input Normalized:** The entered URL was interpreted with default scheme `{norm_url}` "
                "for feature parsing, while preserving the user-entered hostname."
            )

        if is_apex:
            st.warning(
                "⚠️ **Dataset Representation Notice:** The PhishGuard benchmark dataset contains legitimate URLs "
                "exclusively in canonical `https://www.domain.tld` format. Apex-domain URLs such as "
                f"`{norm_url}` are not represented as legitimate samples in the frozen training dataset "
                "and may therefore receive unreliable model predictions. This is a known dataset representation "
                "limitation, not a deployment error."
            )

        with st.spinner("Extracting 27 static features and computing risk analysis..."):
            try:
                assessment = engine.assess_url(input_clean)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.info("Ensure the URL input is a valid string. PhishGuard handles malformed links safely.")
                return

        # ---------------------------------------------------------------------
        # SECTION A: VERDICT & METRIC CARDS
        # ---------------------------------------------------------------------
        st.markdown("## 1. Security Verdict & Risk Level")

        col_v1, col_v2, col_v3, col_v4 = st.columns(4)

        with col_v1:
            st.markdown("**ML Binary Verdict**")
            st.markdown(get_prediction_badge_html(assessment["prediction"]), unsafe_allow_html=True)
            st.caption(f"Classifier: `{assessment.get('explanation_details', {}).get('model_name', 'RandomForestClassifier')}`")

        with col_v2:
            st.markdown("**Operational Risk Tier**")
            st.markdown(get_risk_tier_badge_html(assessment["risk_level"]), unsafe_allow_html=True)
            st.caption(assessment["risk_level_details"]["label"])

        with col_v3:
            st.markdown("**Operational Risk Score**")
            score = assessment["operational_risk_score"]
            st.metric(
                label="Risk Score",
                value=f"{score} / 100",
                delta=f"{assessment['heuristic_adjustment']:+.1f} Heuristic" if assessment['heuristic_adjustment'] != 0 else "0.0 Base"
            )

        with col_v4:
            st.markdown("**Model Phishing Probability**")
            prob_pct = assessment["model_estimated_phishing_probability_pct"]
            st.metric(
                label="Phishing Probability",
                value=f"{prob_pct:.1f}%",
                delta=f"Legitimate: {100.0 - prob_pct:.1f}%"
            )

        # Operational Warnings
        if assessment["warnings"]:
            for w in assessment["warnings"]:
                st.warning(f"🔔 **Operational Warning:** {w}")

        st.markdown("---")

        # ---------------------------------------------------------------------
        # SECTION B: RISK BREAKDOWN & COMPOSITION
        # ---------------------------------------------------------------------
        st.markdown("## 2. Risk Score Formulation & Composition")

        st.markdown(
            "PhishGuard distinguishes the raw **ML Probability** from the **Operational Risk Score**. "
            "Heuristic adjustments are confidence-damped via $\\alpha(P)$ to prevent double-counting."
        )

        col_b1, col_b2, col_b3 = st.columns([1, 1, 2])

        with col_b1:
            st.markdown("#### 1. ML Component")
            st.markdown(f"**`{assessment['ml_risk_component']:.2f}`** pts")
            st.caption("Formulation: $100 \\times P(\\text{phishing})$")

        with col_b2:
            st.markdown("#### 2. Heuristic Adjustment")
            adj = assessment["heuristic_adjustment"]
            adj_color = "#c92a2a" if adj > 0 else "#2b8a3e" if adj < 0 else "#495057"
            st.markdown(f"<span style='color:{adj_color}; font-size:1.5rem; font-weight:bold;'>{adj:+.2f} pts</span>", unsafe_allow_html=True)
            st.caption("Bounded adjustment: $\\Delta \\in [-10.0, +15.0]$")

        with col_b3:
            st.markdown("#### 3. Final Operational Score")
            st.progress(score / 100.0)
            st.caption(f"**Recommended Action:** {assessment['risk_level_details']['recommended_action']}")

        # Triggered Heuristics Table
        if assessment["heuristic_signals"]:
            st.markdown("##### ⚡ Triggered Heuristic Signals")
            h_df = pd.DataFrame(assessment["heuristic_signals"])[["name", "category", "weight", "description"]]
            h_df.columns = ["Rule Name", "Category", "Weight (pts)", "Signal Description"]
            st.dataframe(h_df, use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ No high-severity structural risk heuristics were triggered for this URL.")

        st.markdown("---")

        # ---------------------------------------------------------------------
        # SECTION C: EXPLAINABILITY & LOCAL ATTRIBUTION (PHASE 7)
        # ---------------------------------------------------------------------
        st.markdown("## 3. Local Model Attribution (Saabas Tree Decomposition)")

        st.markdown(
            "Every prediction is decomposed across all 100 trees in the Random Forest. "
            "Bars indicate how much each feature shifted the conditional probability from the baseline prior."
        )

        explanation_details = assessment.get("explanation_details", {})
        bias = explanation_details.get("baseline_prior_bias", 0.4999)
        all_contribs = explanation_details.get("top_contributing_features", [])

        col_chart, col_contribs = st.columns([3, 2])

        with col_chart:
            if all_contribs:
                fig = plot_local_attributions(all_contribs, bias)
                st.pyplot(fig)
                plt.close(fig)

        with col_contribs:
            st.markdown("##### 📌 Key Attribution Drivers")
            pos_contribs = assessment["local_feature_attributions"]["top_positive_contributors"]
            neg_contribs = assessment["local_feature_attributions"]["top_negative_contributors"]

            if pos_contribs:
                st.markdown("**Pushed Toward Phishing (Red):**")
                for p in pos_contribs[:3]:
                    st.markdown(f"- **{p['name']}** (`{p['value']}`): `{p['contribution']:+.3f}`")

            if neg_contribs:
                st.markdown("**Pushed Toward Legitimate (Green):**")
                for n in neg_contribs[:3]:
                    st.markdown(f"- **{n['name']}** (`{n['value']}`): `{n['contribution']:+.3f}`")

            recon_err = explanation_details.get("mathematical_consistency", {}).get("reconstruction_error", 0.0)
            st.caption(f"Mathematical Reconstruction Error: `{recon_err:.2e}`")

        # Human Narrative
        st.markdown("##### 📝 Comprehensive Narrative Summary")
        st.info(assessment["explanation"])

        st.markdown("---")

        # ---------------------------------------------------------------------
        # SECTION D: FULL 27-FEATURE PROFILE
        # ---------------------------------------------------------------------
        st.markdown("## 4. Canonical 27 Static Feature Profile")
        with st.expander("🔍 Click to view all 27 extracted features and descriptions", expanded=False):
            feature_records = []
            extracted_features = assessment.get("features", {})
            for feat in FEATURE_NAMES:
                meta = FEATURE_METADATA.get(feat, {})
                feature_records.append({
                    "Feature Identifier": feat,
                    "Display Name": meta.get("name", feat),
                    "Category": meta.get("category", "General"),
                    "Extracted Value": extracted_features.get(feat, 0),
                    "Description": meta.get("description", "")
                })
            df_feat = pd.DataFrame(feature_records)
            st.dataframe(df_feat, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ---------------------------------------------------------------------
        # SECTION E: GLOBAL FEATURE IMPORTANCE & FIGURES
        # ---------------------------------------------------------------------
        st.markdown("## 5. Global Model Intelligence & Research Context")
        with st.expander("📈 Click to view Global Feature Importance and Experimental Findings", expanded=False):
            tab_imp, tab_unseen = st.tabs(["Global Feature Importance (Figure 15)", "Unseen-Domain Findings"])

            with tab_imp:
                st.markdown(
                    "> **Academic Disclosure:** Global Gini importance reflects overall decision tree split purity across "
                    "the training distribution. It does **not** represent causal evidence or per-instance attribution."
                )
                df_global = load_global_importance_df()
                fig15_file = REPORTS_FIGURES_DIR / "15_global_feature_importance.png"
                if fig15_file.exists():
                    st.image(str(fig15_file), caption="Figure 15: Global Random Forest Feature Importance (All 27 Features)")
                else:
                    st.dataframe(df_global.head(15), use_container_width=True, hide_index=True)

            with tab_unseen:
                st.markdown(
                    "#### Unseen-Domain Generalization (Phase 6)\n"
                    "- Evaluated under strict domain grouping (**0.00% domain overlap** across 44,014 held-out domains).\n"
                    "- Retained **>99.5% Phishing Recall** on unseen domains (Generalization gap < 0.0002).\n"
                    "- **Identified Blind Spot:** Root-level HTTPS URLs with zero subdomains and no keywords mimic benign structure. "
                    "This empirical finding motivates Phase 8 multi-tiered risk scoring."
                )

        st.markdown("---")

        # ---------------------------------------------------------------------
        # SECTION F: ACADEMIC DISCLOSURES & LIMITATIONS
        # ---------------------------------------------------------------------
        st.markdown("## 6. Academic Disclosures & Operational Limitations")
        st.warning(
            "**Crucial Academic Disclosures:**\n\n"
            "1. **Static URL-Only Scope:** PhishGuard operates purely on lexical and structural features extracted from raw URL strings. "
            "It deliberately does not fetch webpage content, execute JavaScript, parse HTML/DOM trees, query WHOIS, or contact DNS servers.\n"
            "2. **Operational Risk vs. Malicious Intent:** The operational risk score is a decision-support and triage metric. "
            "It reflects structural and lexical suspicion; it does **not** constitute proof of malicious attacker intent.\n"
            "3. **HTTPS Does Not Guarantee Safety:** Modern adversaries extensively deploy HTTPS encryption. Presence of HTTPS should never be taken as proof of legitimacy.\n"
            "4. **No Real-World Zero-Day Detection Claim:** Phase 6 provides an offline unseen-domain generalization proxy evaluation; "
            "it does not guarantee detection of all zero-day attacks in live production environments."
        )

    else:
        # Initial Landing State
        st.info("👈 Enter a URL above or choose a sample from the quick demo dropdown to begin analysis.")

        st.markdown("### How PhishGuard Analyzes URLs")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.markdown("#### 1. Safe Static Extraction")
            st.markdown(
                "Extracts 27 lexical and structural characteristics without connecting to the remote host. "
                "Zero risk of exploit execution or payload delivery."
            )
        with col_c2:
            st.markdown("#### 2. Calibrated Random Forest")
            st.markdown(
                "Evaluates the feature vector against 100 decision trees trained with strict domain separation, "
                "yielding continuous, highly calibrated phishing probabilities."
            )
        with col_c3:
            st.markdown("#### 3. Local Tree Explainability")
            st.markdown(
                "Decomposes probability into exact feature contributions via the Saabas tree traversal algorithm, "
                "showing exactly which tokens pushed the verdict."
            )

if __name__ == "__main__":
    main()
