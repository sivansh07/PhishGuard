# PhishGuard: Phase 7 Explainability Engine Report

**Module:** Explainability Engine  
**Deployment Model:** RandomForestClassifier  
**Evaluated Features:** Exactly 27 static URL features (No webpage content, DOM, network, or reputation lookup)  

---

## 1. Global Feature Importance Analysis

Global feature importance is measured via mean Gini impurity reduction across all 100 decision trees in the fitted Random Forest.

> **Critical Academic Distinction:** Global feature importance reflects model behavior across the training distribution. It is **not** a per-instance explanation and does **not** constitute causal evidence of phishing behavior.

| Rank | Feature Identifier | Feature Display Name | Category | Gini Importance | Cumulative Importance |
| :---: | :--- | :--- | :--- | :---: | :---: |
| 1 | `has_https` | HTTPS Scheme Indicator | Security & Protocol | 0.4446 | 0.4446 |
| 2 | `path_length` | Path Length | Length & Hierarchy | 0.1478 | 0.5924 |
| 3 | `num_slashes` | Slash Count | Delimiter & Punctuation | 0.1331 | 0.7256 |
| 4 | `digit_ratio` | Digit Ratio | Character Distribution | 0.0635 | 0.7890 |
| 5 | `num_special_chars` | Special Character Count | Character Distribution | 0.0476 | 0.8366 |
| 6 | `num_digits` | Digit Count | Character Distribution | 0.0405 | 0.8771 |
| 7 | `url_length` | URL Length | Length & Hierarchy | 0.0347 | 0.9118 |
| 8 | `num_subdomains` | Subdomain Count | Domain & Infrastructure | 0.0252 | 0.9370 |
| 9 | `num_dots` | Dot Count | Delimiter & Punctuation | 0.0226 | 0.9596 |
| 10 | `hostname_length` | Hostname Length | Length & Hierarchy | 0.0174 | 0.9770 |
| 11 | `num_hyphens` | Hyphen Count | Delimiter & Punctuation | 0.0117 | 0.9887 |
| 12 | `special_char_ratio` | Special Character Ratio | Character Distribution | 0.0075 | 0.9962 |
| 13 | `suspicious_keyword_present` | Suspicious Keyword Flag | Lexical Semantics | 0.0019 | 0.9980 |
| 14 | `suspicious_keyword_count` | Suspicious Keyword Count | Lexical Semantics | 0.0015 | 0.9996 |
| 15 | `num_question_marks` | Question Mark Count | Delimiter & Punctuation | 0.0002 | 0.9998 |
| 16 | `has_url_shortener` | URL Shortener Domain | Domain & Infrastructure | 0.0001 | 0.9999 |
| 17 | `query_length` | Query Length | Length & Hierarchy | 0.0000 | 0.9999 |
| 18 | `has_query` | Query Component Presence | Length & Hierarchy | 0.0000 | 1.0000 |
| 19 | `has_fragment` | URL Fragment Presence | Length & Hierarchy | 0.0000 | 1.0000 |
| 20 | `has_at_symbol` | @ Symbol Delimiter | Security & Protocol | 0.0000 | 1.0000 |
| 21 | `num_underscores` | Underscore Count | Delimiter & Punctuation | 0.0000 | 1.0000 |
| 22 | `num_equal_signs` | Equal Sign Count | Delimiter & Punctuation | 0.0000 | 1.0000 |
| 23 | `has_ip_address` | Direct IP Address Host | Security & Protocol | 0.0000 | 1.0000 |
| 24 | `num_percent_signs` | Percent Sign Count | Delimiter & Punctuation | 0.0000 | 1.0000 |
| 25 | `num_ampersands` | Ampersand Count | Delimiter & Punctuation | 0.0000 | 1.0000 |
| 26 | `has_port` | Explicit Non-Standard Port | Security & Protocol | 0.0000 | 1.0000 |
| 27 | `is_encoded` | Percent-Encoding Flag | Security & Protocol | 0.0000 | 1.0000 |

---

## 2. Local Attribution Methodology (Saabas Decision Path Decomposition)

Local feature attributions are computed using the **Saabas Tree Path Decomposition** algorithm:
1. For every tree in the Random Forest, the decision path of a given sample $x$ is traced from root to leaf.
2. At each intermediate split node $u$ branching on feature $j$ to child node $v$, the shift in conditional phishing probability $\Delta = P(\text{phishing} \mid v) - P(\text{phishing} \mid u)$ is attributed directly to feature $j$.
3. Averaged across the entire forest ensemble:
   $$P(\text{phishing} \mid \mathbf{x}) = \text{Ensemble Bias} + \sum_{j=1}^{27} \text{Contribution}_j(\mathbf{x})$$
4. The ensemble baseline prior bias for this model is **49.99%**.
5. Local attributions provide an exact, signed mathematical decomposition with machine-epsilon reconstruction error ($< 10^{-15}$).

---

## 3. Inference and Explanation Latency Benchmark

Benchmarked over 50 iterations per representative URL:

| Metric | Measured Latency |
| :--- | :--- |
| **Standard Prediction Latency (Extract + Model Predict)** | **35.04 ms** |
| **Prediction + Local Explainability (Extract + Predict + Tree Decomposition)** | **74.73 ms** |
| **Explainability Overhead** | **+39.69 ms** |

*Finding: The explainability engine adds minimal overhead (~39.7 ms) without introducing heavy third-party dependencies, preserving real-time evaluation capability.*

---

## 4. False-Negative Explainability Findings

By inspecting missed phishing URLs from the held-out test split, the explainability engine identified the precise mathematical mechanism of evasion:
- **Observed Model Behavior:** Missed phishing URLs consistently possess `has_https = 1`, `path_length = 0`, `num_slashes = 2`, and `suspicious_keyword_present = 0`. Because these features carry the highest negative attributions in the model, their cumulative effect overrides minor lexical indicators and pushes the probability below 0.05.
- **Security Interpretation:** Adversaries who register clean domains, use HTTPS, and host payloads at the root path bypass static lexical analysis because their URL structure is indistinguishable from standard benign sites.
- **System Impact:** This finding reinforces the academic thesis that static URL classifiers have an intrinsic boundary and directly motivates the Phase 8 Risk Engine.

---

## 5. Academic Disclosures & Limitations

1. **Non-Causal Nature:** Neither global feature importance nor local feature attributions demonstrate causal relationships. They describe the decision boundaries of the trained classifier.
2. **URL-Only Perspective:** The system deliberately refrains from visiting target servers, evaluating page DOM, or executing JavaScript to ensure zero operational risk.
3. **No Attacker Intent:** A high feature attribution (e.g. for `num_subdomains`) indicates statistical correlation within the training distribution, not subjective attacker intent.
