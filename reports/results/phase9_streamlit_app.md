# PhishGuard — Phase 9: Streamlit Web Application Report

## Executive Summary
Phase 9 delivers a production-grade, interactive cybersecurity analysis dashboard implemented in **Streamlit** (`app.py`). The web interface operationalizes the complete PhishGuard pipeline—integrating the verified 27-feature URL extraction engine, the frozen `RandomForestClassifier` deployment candidate, the Phase 7 Saabas local feature attribution engine, and the Phase 8 confidence-aware heuristic risk scoring layer into an intuitive visual analyst dashboard.

The application adheres strictly to all academic constraints established in Phases 1–8:
- **Zero modification** to underlying datasets, features, models, weights, or metrics.
- **Strictly offline operation**: Zero network, DNS, WHOIS, or HTTP/DOM requests during execution.
- **Strict academic terminology**: Replaces misleading commercial claims ("100% zero-day detection", "malicious intent") with precise academic descriptors ("operational risk score", "lexical signal attribution", "offline unseen-domain proxy evaluation").

---

## 1. Application Architecture

```
User Input (Interactive Text / Benchmark Dropdown)
                         │
                         ▼
        Feature Extractor (src/feature_extractor.py)
              [27 Static Lexical & Structural Features]
                         │
                         ▼
          Deployment Model (models/phishing_model.joblib)
              [RandomForestClassifier Pipeline]
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
Phase 7 Saabas Attribution         Phase 8 Risk Engine
 (src/explainability.py)           (src/risk_engine.py)
   Exact Tree Decomposition          Confidence Damping:
   φ_i feature attributions            R = clamp(100·P + (1 - 2|P-0.5|)·Δ, 0, 100)
        │                                 │
        └────────────────┬────────────────┘
                         ▼
               Streamlit UI Dashboard (app.py)
       - High-Level Classification & Risk Tier Badges
       - Mathematical Risk Breakdown (ML Prob vs Heuristic Delta)
       - Matplotlib Saabas Local Attribution Waterfall Chart
       - Exact Reconstruction Audit (|Sum(φ) - (P - Base)| < 1e-15)
       - Canonical 27-Feature Profile Table (Expandable)
       - Global Model Intelligence (Unseen-Domain Metrics & Figure 15)
       - Prominent Academic Disclaimers & Security Limitations Banner
```

---

## 2. Core Dashboard Components & Design

### 2.1 Caching & Performance
- **Resource Caching (`@st.cache_resource`)**: The `PhishGuardRiskEngine` and global feature importance table are cached in memory upon initial boot.
- **Inference Latency**: Sub-100 ms end-to-end evaluation per URL (feature extraction + Random Forest inference + Saabas tree decomposition + risk engine computation).

### 2.2 Input Controls & Benchmark Quick-Select
Analysts can either enter custom URLs or select from pre-configured benchmark demonstration cases covering each operational tier:
1. `https://www.google.com/search?q=machine+learning` (Low Risk, Benign)
2. `https://github.com/torvalds/linux/blob/master/README.md` (Low Risk, Benign)
3. `http://login.secure-bank.verification-account.com/auth/login.php` (Critical Risk, Phishing)
4. `http://192.168.1.100/paypal/login.html` (Critical Risk, Phishing)
5. `https://bit.ly/3xY8z9K` (Guarded Risk, URL Shortener)
6. `http://service-paypal.com.account-update.info/login/` (Critical Risk, Phishing)

### 2.3 Verdict & Metric Badges
- **Operational Risk Score**: Large numeric KPI ($0–100$) color-coded according to the 5 operational tiers:
  - **LOW** ($[0, 20)$): `#22c55e` (Emerald Green)
  - **GUARDED** ($[20, 40)$): `#3b82f6` (Sky Blue)
  - **MODERATE** ($[40, 60)$): `#f59e0b` (Amber Orange)
  - **HIGH** ($[60, 80)$): `#f97316` (Deep Orange)
  - **CRITICAL** ($[80, 100]$): `#ef4444` (Crimson Red)
- **Classification Verdict**: Formal binary label (`SUSPECTED PHISHING` vs `LEGITIMATE`) with high-contrast custom CSS badges.
- **Model Probabilities**: Distinct breakdown of raw Random Forest phishing probability $P(\text{Phishing})$ vs $P(\text{Legitimate})$.

### 2.4 Mathematical Risk Decomposition
The dashboard explicitly reveals the exact contribution of the ML classifier versus the heuristic signal layer:
$$\text{Operational Risk} = \text{clamp}\left(100 \times P_{\text{RF}} + (1 - 2|P_{\text{RF}} - 0.5|) \times \Delta_{\text{heuristics}},\, 0,\, 100\right)$$
- Displays the raw base score ($100 \times P_{\text{RF}}$).
- Displays the confidence damping coefficient $(1 - 2|P_{\text{RF}} - 0.5|)$.
- Displays the net heuristic adjustment ($\Delta_{\text{heuristics}}$) along with an active heuristic signal badge list.

### 2.5 Local Attribution Visualization & Reconstruction Audit
- **Attribution Chart**: Horizontal bar chart generated via Matplotlib/Seaborn illustrating top positive (crimson red) and negative (emerald green) feature contributions ($\phi_i$).
- **Saabas Mathematical Audit**: Explicit verification of tree additivity:
  $$\sum_{i=1}^{27} \phi_i = P_{\text{RF}} - \text{baseline}$$
  The dashboard reports the numerical absolute error ($|\sum \phi_i - (P_{\text{RF}} - P_{\text{base}})|$), verifying convergence to double-precision machine precision ($< 10^{-15}$).

### 2.6 Canonical 27 Feature Profile Table
An interactive expandable drawer provides a complete audit table of all 27 features extracted for the evaluated URL:
- Feature name
- Extracted value
- Category (Lexical, Structural, Security, Token, Ratios)
- Engineering description

### 2.7 Global Intelligence & Unseen-Domain Generalization
Embeds key research findings directly into the interface:
- Unseen-Domain Holdout Metrics: Accuracy ($99.79\%$), Precision ($99.97\%$), Recall ($99.54\%$), ROC-AUC ($0.9983$), FNR ($0.46\%$).
- 5-Fold StratifiedGroupKFold stability summary.
- Embedded Figure 15 (Global Feature Importance).

---

## 3. Security, Privacy, and Offline Guarantee

1. **Zero Outbound Connections**: The application runs completely isolated from external networks. No HTTP clients (`requests`, `urllib`), DNS resolvers (`socket.gethostbyname`), or WHOIS lookups are executed.
2. **Safe Static Analysis**: Analyzes only the lexical string representation of URLs. Malicious binaries, drive-by scripts, and payload endpoints are never executed, downloaded, or rendered.
3. **Graceful Error Handling**: Robust defensive handling for blank strings, malformed schemas, overly long inputs, or invalid URI encodings, preventing unhandled exceptions or app crashes.

---

## 4. Academic Disclosures & Limitation Disclaimers

A prominent banner in the dashboard highlights the academic boundaries of the research:
- **No Causal Attack Intent**: Lexical features measure statistical structural anomalies, not cryptographic or forensic intent.
- **Unseen-Domain Proxy**: Strong unseen-domain generalization demonstrates resilience against domain memorization, but does not constitute a guarantee against all real-world zero-day campaigns.
- **Cloud Infrastructure Blind Spot**: Benign cloud platforms (e.g. AWS S3, Google Docs, Firebase) abused for phishing can exhibit clean lexical characteristics that static URL analysis alone cannot reliably identify without content or context signals.

---

## 5. Verification & Test Suite Integration

The Phase 9 Streamlit web application was verified via dedicated automated tests in `tests/test_app.py`:
- `test_risk_color_mapping`: Verifies valid hex color mapping across all 5 operational tiers.
- `test_render_badge_html`: Verifies proper CSS class and escaping for HTML status badges.
- `test_engine_caching`: Verifies single-instance initialization and cached resource stability.
- `test_assess_url_stream_integration`: Verifies end-to-end evaluation of benchmark demo URLs.
- `test_local_attribution_plot_generation`: Verifies error-free Matplotlib figure generation.
- `test_offline_guarantee_in_app`: Verifies mock socket network isolation during app execution.
- `test_empty_url_handling`: Verifies graceful non-crashing handling of empty strings.
- `test_malformed_url_handling`: Verifies graceful evaluation of invalid schemas and long input strings.
- `test_feature_table_structure`: Verifies complete 27-row extraction across all canonical features.

**Full Test Suite Summary:**
- **Total Tests**: 52 passed
- **Status**: 100% passing across 6 test modules (`test_features.py`, `test_prediction.py`, `test_unseen_domain.py`, `test_explainability.py`, `test_risk_engine.py`, `test_app.py`)
- **Regressions**: 0
