# PhishGuard: Phase 8 Explainable Risk Scoring Engine Report

**Module:** Explainable Risk Scoring Engine & Heuristic Signal Layer  
**Core Model:** RandomForestClassifier (Phase 4/5/6 Verified Candidate)  
**Evaluated Features:** Exactly 27 Static URL Features  

---

## 1. Risk Score Mathematical Formulation

The PhishGuard Risk Engine transforms the calibrated Random Forest probability into an operational 0–100 risk score while preserving the ML probability as an independent, transparent signal.

### Mathematical Formulation:
$$\text{ML\_Risk} = 100 \times P(\text{phishing} \mid \text{URL}) \in [0.0, 100.0]$$

$$\Delta_{\text{Heuristic}} = \alpha(P) \times \text{clip}\left(\sum_{k=1}^{K} w_k \cdot \mathbb{I}(h_k), -10.0, +15.0\right)$$

$$\text{Operational Risk Score} = \text{clip}\left(\text{round}(\text{ML\_Risk} + \Delta_{\text{Heuristic}}), 0, 100\right)$$

### Confidence-Aware Scaling & Double-Counting Mitigation:
Because many heuristic signals overlap with features already utilized by the Random Forest, directly adding unconstrained points would blindly double-count lexical signals.
To resolve this:
1. **Confidence Damping Factor $\alpha(P)$:**
   $$\alpha(P) = \text{clip}(1.0 - 0.70 \cdot |2P - 1|, 0.30, 1.00)$$
   - When the Random Forest is highly confident ($P \approx 0.0$ or $1.0$), $\alpha(P) = 0.30$, preventing heuristics from overriding decisive model predictions.
   - When the model decision is borderline or ambiguous ($P \approx 0.50$), $\alpha(P) = 1.00$, allowing heuristic structural indicators to provide operational differentiation.
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
| `H_IP_HOST` | Direct IP Address Host | +8.0 | 117 | 100.0% | 88 | 100.0% | 0 | 0 |
| `H_AT_SYMBOL` | @ Symbol Credential Delimiter | +8.0 | 296 | 100.0% | 329 | 100.0% | 0 | 0 |
| `H_URL_SHORTENER` | URL Shortener Service | +6.0 | 123 | 100.0% | 21 | 95.2% | 1 | 0 |
| `H_HEX_ENCODING` | Hexadecimal Percent-Encoding | +4.0 | 167 | 100.0% | 197 | 100.0% | 0 | 0 |
| `H_EXCESS_SUBDOMAINS` | Excessive Subdomain Hierarchy | +4.0 | 573 | 98.6% | 572 | 97.9% | 3 | 8 |
| `H_HIGH_DIGIT_RATIO` | Elevated Numeric Character Density | +4.0 | 1,939 | 99.8% | 1,872 | 99.9% | 0 | 2 |
| `H_SUSPICIOUS_KEYWORD` | Authentication Keyword Concentration | +3.0 | 1,699 | 91.9% | 1,606 | 92.8% | 1 | 137 |
| `H_MULTIPLE_HYPHENS` | Multiple Hyphen Delimiters | +2.0 | 1,449 | 98.2% | 1,471 | 97.8% | 0 | 26 |
| `H_EXCESSIVE_LENGTH` | Abnormally Long URL | +3.0 | 2,312 | 100.0% | 2,338 | 100.0% | 0 | 0 |
| `H_UNENCRYPTED_HTTP` | Unencrypted HTTP Scheme | +4.0 | 10,301 | 100.0% | 10,110 | 100.0% | 0 | 0 |
| `H_STANDARD_BENIGN_PROFILE` | Standard Benign Root Structure | -3.0 | 26,003 | 0.7% | 26,015 | 0.7% | 68 | 25829 |

---

## 4. Error-Correction and False-Negative Analysis

### Critical Empirical Finding:
In Phase 6, Random Forest missed 92 phishing URLs. When evaluating whether heuristic signals can detect these missed phishing URLs:
- **Observed Behavior:** 100% of missed phishing URLs had `has_https = 1`, `path_length = 0`, `num_slashes = 2`, and 98.9% had `suspicious_keyword_present = 0`.
- **Benign Comparison:** In the legitimate test set, 100% of the 26,963 true negative URLs exhibited the **exact same structural pattern** (HTTPS root domain with zero path length and no keywords).
- **Academic Conclusion:** Static URL heuristics **cannot** reliably separate root-level evasive phishing URLs without causing tens of thousands of false alarms on benign root sites. This empirically confirms the intrinsic limitation of static URL analysis and motivates external DNS/WHOIS telemetry in production environments.

---

## 5. Representative Risk Assessments

### Sample URL: `http://www.worldmedicsky.info`
- **Classification:** `POTENTIAL PHISHING`
- **Model Phishing Probability:** 100.00%
- **Operational Risk Score:** **100 / 100** (CRITICAL RISK)
- **ML Component:** 100.00 | **Heuristic Adjustment:** +1.20
- **Active Heuristics:** `Unencrypted HTTP Scheme` (+4.0)
- **Warnings:** Unencrypted connection: Legitimate modern services overwhelmingly mandate HTTPS.
- **Narrative:** The URL receives an Operational Risk Score of 100/100 (CRITICAL RISK), reflecting a model-estimated phishing probability of 100.0%. Active structural risk signals include: 'Unencrypted HTTP Scheme'. Model feature attribution indicates that 'HTTPS Scheme Indicator' (+0.737) contributed toward phishing. Conversely, 'Slash Count' (-0.053), 'Path Length' (-0.050) exerted downward pressure toward a legitimate classification. Operational Notice: Unencrypted connection: Legitimate modern services overwhelmingly mandate HTTPS. Academic Disclosure: These indicators describe statistical model behavior and lexical properties; they do not establish causal attacker intent or inspect real-time server contents.

### Sample URL: `https://www.downapp.com`
- **Classification:** `LEGITIMATE (BENIGN)`
- **Model Phishing Probability:** 0.19%
- **Operational Risk Score:** **0 / 100** (LOW RISK)
- **ML Component:** 0.19 | **Heuristic Adjustment:** -0.91
- **Active Heuristics:** `Standard Benign Root Structure` (-3.0)
- **Warnings:** Standard root structure: Typical of legitimate corporate domains, but also adopted by evasive landing pages.
- **Narrative:** The URL receives an Operational Risk Score of 0/100 (LOW RISK), reflecting a model-estimated phishing probability of 0.2%. Active structural risk signals include: 'Standard Benign Root Structure'. Model feature attribution indicates that 'Special Character Ratio' (+0.007) contributed toward phishing. Conversely, 'HTTPS Scheme Indicator' (-0.223), 'Path Length' (-0.068) exerted downward pressure toward a legitimate classification. Operational Notice: Standard root structure: Typical of legitimate corporate domains, but also adopted by evasive landing pages. Academic Disclosure: These indicators describe statistical model behavior and lexical properties; they do not establish causal attacker intent or inspect real-time server contents.

### Sample URL: `https://www.cfg.me`
- **Classification:** `LEGITIMATE (BENIGN)`
- **Model Phishing Probability:** 0.30%
- **Operational Risk Score:** **0 / 100** (LOW RISK)
- **ML Component:** 0.30 | **Heuristic Adjustment:** -0.91
- **Active Heuristics:** `Standard Benign Root Structure` (-3.0)
- **Warnings:** Standard root structure: Typical of legitimate corporate domains, but also adopted by evasive landing pages.
- **Narrative:** The URL receives an Operational Risk Score of 0/100 (LOW RISK), reflecting a model-estimated phishing probability of 0.3%. Active structural risk signals include: 'Standard Benign Root Structure'. Model feature attribution indicates that 'Special Character Ratio' (+0.015) contributed toward phishing. Conversely, 'HTTPS Scheme Indicator' (-0.228), 'Path Length' (-0.068) exerted downward pressure toward a legitimate classification. Operational Notice: Standard root structure: Typical of legitimate corporate domains, but also adopted by evasive landing pages. Academic Disclosure: These indicators describe statistical model behavior and lexical properties; they do not establish causal attacker intent or inspect real-time server contents.

### Sample URL: `https://www.cns11643.gov.tw`
- **Classification:** `POTENTIAL PHISHING`
- **Model Phishing Probability:** 52.93%
- **Operational Risk Score:** **53 / 100** (MODERATE RISK)
- **ML Component:** 52.93 | **Heuristic Adjustment:** +0.00
- **Active Heuristics:** None
- **Warnings:** Borderline Model Decision: Model probability (52.9%) is near the classification threshold (0.50). Heightened scrutiny and manual verification recommended.
- **Narrative:** The URL receives an Operational Risk Score of 53/100 (MODERATE RISK), reflecting a model-estimated phishing probability of 52.9%. No high-severity structural risk heuristics were triggered. Model feature attribution indicates that 'Digit Ratio' (+0.298), 'Digit Count' (+0.286) contributed toward phishing. Conversely, 'HTTPS Scheme Indicator' (-0.239), 'Slash Count' (-0.112) exerted downward pressure toward a legitimate classification. Operational Notice: Borderline Model Decision: Model probability (52.9%) is near the classification threshold (0.50). Heightened scrutiny and manual verification recommended. Academic Disclosure: These indicators describe statistical model behavior and lexical properties; they do not establish causal attacker intent or inspect real-time server contents.

### Sample URL: `http://192.168.1.1/admin/login.php`
- **Classification:** `POTENTIAL PHISHING`
- **Model Phishing Probability:** 100.00%
- **Operational Risk Score:** **100 / 100** (CRITICAL RISK)
- **ML Component:** 100.00 | **Heuristic Adjustment:** +4.50
- **Active Heuristics:** `Direct IP Address Host` (+8.0), `Elevated Numeric Character Density` (+4.0), `Authentication Keyword Concentration` (+3.0), `Unencrypted HTTP Scheme` (+4.0)
- **Warnings:** Direct IP host: Bypasses standard domain reputation and registration oversight.
- **Narrative:** The URL receives an Operational Risk Score of 100/100 (CRITICAL RISK), reflecting a model-estimated phishing probability of 100.0%. Active structural risk signals include: 'Direct IP Address Host', 'Elevated Numeric Character Density', 'Authentication Keyword Concentration'. Model feature attribution indicates that 'Path Length' (+0.120), 'Slash Count' (+0.104) contributed toward phishing. Conversely, 'Hostname Length' (-0.006), 'Hyphen Count' (-0.005) exerted downward pressure toward a legitimate classification. Operational Notice: Direct IP host: Bypasses standard domain reputation and registration oversight. Academic Disclosure: These indicators describe statistical model behavior and lexical properties; they do not establish causal attacker intent or inspect real-time server contents.

### Sample URL: `https://netflix-vn.com`
- **Classification:** `POTENTIAL PHISHING`
- **Model Phishing Probability:** 100.00%
- **Operational Risk Score:** **100 / 100** (CRITICAL RISK)
- **ML Component:** 100.00 | **Heuristic Adjustment:** +0.00
- **Active Heuristics:** None
- **Narrative:** The URL receives an Operational Risk Score of 100/100 (CRITICAL RISK), reflecting a model-estimated phishing probability of 100.0%. No high-severity structural risk heuristics were triggered. Model feature attribution indicates that 'Dot Count' (+0.727), 'Hyphen Count' (+0.093) contributed toward phishing. Conversely, 'HTTPS Scheme Indicator' (-0.147), 'Path Length' (-0.068) exerted downward pressure toward a legitimate classification. Academic Disclosure: These indicators describe statistical model behavior and lexical properties; they do not establish causal attacker intent or inspect real-time server contents.

### Sample URL: `https://bit.ly/secure-banking-alert`
- **Classification:** `POTENTIAL PHISHING`
- **Model Phishing Probability:** 100.00%
- **Operational Risk Score:** **100 / 100** (CRITICAL RISK)
- **ML Component:** 100.00 | **Heuristic Adjustment:** +2.70
- **Active Heuristics:** `URL Shortener Service` (+6.0), `Authentication Keyword Concentration` (+3.0)
- **Warnings:** Shortened URL: Destination domain is obscured, preventing visual domain verification.
- **Narrative:** The URL receives an Operational Risk Score of 100/100 (CRITICAL RISK), reflecting a model-estimated phishing probability of 100.0%. Active structural risk signals include: 'URL Shortener Service', 'Authentication Keyword Concentration'. Model feature attribution indicates that 'Slash Count' (+0.180), 'Path Length' (+0.178) contributed toward phishing. Conversely, 'HTTPS Scheme Indicator' (-0.040), 'Digit Ratio' (-0.036) exerted downward pressure toward a legitimate classification. Operational Notice: Shortened URL: Destination domain is obscured, preventing visual domain verification. Academic Disclosure: These indicators describe statistical model behavior and lexical properties; they do not establish causal attacker intent or inspect real-time server contents.

---

## 6. Layered Latency Benchmark

Measured over 40 iterations across representative evaluation URLs:

| Layer | Cumulative Operations | Mean Latency | Added Overhead |
| :--- | :--- | :---: | :---: |
| **Layer 1: Inference** | Feature Extraction + Random Forest Predict | **47.85 ms** | Baseline |
| **Layer 2: Explainability** | Layer 1 + 100-Tree Saabas Path Decomposition | **69.61 ms** | +21.76 ms |
| **Layer 3: Risk Engine** | Layer 1 + Layer 2 + Heuristic Evaluation & Risk Synthesis | **115.28 ms** | +45.67 ms |

**Total End-to-End Latency:** **115.28 ms** (Overhead over raw prediction: +67.43 ms).

---

## 7. Academic Limitations & Disclosures

1. **Operational Risk vs. Malicious Intent:** The operational risk score is a prioritized triage metric. It reflects structural suspicion within the 27 static features; it is **not** a causal guarantee of attacker intent.
2. **Heuristics are Not Proof:** Individual heuristic activations (e.g. high digit count) indicate statistical elevation of risk, not definitive proof of phishing.
3. **No Webpage Inspection:** PhishGuard deliberately executes no network requests and renders no DOM elements to preserve user privacy and host safety.
4. **HTTPS is Protocol, Not Trust:** Modern attackers widely deploy HTTPS certificates; HTTPS presence cannot be treated as a guarantee of safety.
5. **Proxy Unseen Evaluation:** Phase 6 remains an offline unseen-domain proxy evaluation and does not guarantee detection of real-world zero-day attacks.
