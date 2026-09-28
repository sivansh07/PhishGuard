# PhishGuard: Phase 7 Local Explanations and False-Negative Analysis

> **Academic Disclosure:** Local feature attributions represent model decision shifts decomposed across the 27 static URL features. They do not constitute causal explanations of phishing behavior.

---

## 1. Representative Evaluation Cases

### Case: True Positive (Confirmed Phishing)
- **URL:** `http://www.worldmedicsky.info`
- **Ground Truth:** Phishing
- **Model Prediction:** POTENTIAL PHISHING (Estimated Phishing Probability: **100.00%**)
- **Ensemble Prior Bias:** 49.99%
- **Reconstruction Error:** `1.11e-16`

| Contributing Feature | Category | Feature Value | Direction | Attribution |
| :--- | :--- | :--- | :--- | :--- |
| **HTTPS Scheme Indicator** | Security & Protocol | `0` | increases_phishing | `+0.7365` |
| **Slash Count** | Delimiter & Punctuation | `2` | decreases_phishing | `-0.0528` |
| **Path Length** | Length & Hierarchy | `0` | decreases_phishing | `-0.0498` |
| **Digit Ratio** | Character Distribution | `0.0` | decreases_phishing | `-0.0294` |
| **Special Character Count** | Character Distribution | `5` | decreases_phishing | `-0.0222` |
| **Digit Count** | Character Distribution | `0` | decreases_phishing | `-0.0213` |

**Human-Readable Summary:** Prediction: POTENTIAL PHISHING (Estimated Phishing Probability: 100.00%). Baseline Training Prior: The model begins with an ensemble prior bias of 49.99%. Features Increasing Phishing Probability: 'HTTPS Scheme Indicator' (value: 0, attribution: +0.7365) shifted the prediction upward toward phishing. Features Decreasing Phishing Probability: 'Slash Count' (value: 2, attribution: -0.0528), 'Path Length' (value: 0, attribution: -0.0498), 'Digit Ratio' (value: 0.0, attribution: -0.0294) shifted the prediction downward toward legitimate. Academic Disclosure: Feature contributions reflect model internal attribution across the 27 static URL features and do not constitute proof of attacker intent or external webpage safety.

---

### Case: True Negative (Confirmed Legitimate)
- **URL:** `https://www.downapp.com`
- **Ground Truth:** Legitimate
- **Model Prediction:** LEGITIMATE (BENIGN) (Estimated Phishing Probability: **0.19%**)
- **Ensemble Prior Bias:** 49.99%
- **Reconstruction Error:** `3.86e-17`

| Contributing Feature | Category | Feature Value | Direction | Attribution |
| :--- | :--- | :--- | :--- | :--- |
| **HTTPS Scheme Indicator** | Security & Protocol | `1` | decreases_phishing | `-0.2229` |
| **Path Length** | Length & Hierarchy | `0` | decreases_phishing | `-0.0676` |
| **Slash Count** | Delimiter & Punctuation | `2` | decreases_phishing | `-0.0620` |
| **Digit Ratio** | Character Distribution | `0.0` | decreases_phishing | `-0.0356` |
| **Special Character Count** | Character Distribution | `5` | decreases_phishing | `-0.0279` |
| **Digit Count** | Character Distribution | `0` | decreases_phishing | `-0.0223` |

**Human-Readable Summary:** Prediction: LEGITIMATE (BENIGN) (Estimated Phishing Probability: 0.19%). Baseline Training Prior: The model begins with an ensemble prior bias of 49.99%. Features Increasing Phishing Probability: 'Special Character Ratio' (value: 0.2174, attribution: +0.0071) shifted the prediction upward toward phishing. Features Decreasing Phishing Probability: 'HTTPS Scheme Indicator' (value: 1, attribution: -0.2229), 'Path Length' (value: 0, attribution: -0.0676), 'Slash Count' (value: 2, attribution: -0.0620) shifted the prediction downward toward legitimate. Academic Disclosure: Feature contributions reflect model internal attribution across the 27 static URL features and do not constitute proof of attacker intent or external webpage safety.

---

### Case: False Negative (Missed Phishing - Evasion Case)
- **URL:** `https://www.cfg.me`
- **Ground Truth:** Phishing
- **Model Prediction:** LEGITIMATE (BENIGN) (Estimated Phishing Probability: **0.30%**)
- **Ensemble Prior Bias:** 49.99%
- **Reconstruction Error:** `1.04e-17`

| Contributing Feature | Category | Feature Value | Direction | Attribution |
| :--- | :--- | :--- | :--- | :--- |
| **HTTPS Scheme Indicator** | Security & Protocol | `1` | decreases_phishing | `-0.2282` |
| **Path Length** | Length & Hierarchy | `0` | decreases_phishing | `-0.0676` |
| **Slash Count** | Delimiter & Punctuation | `2` | decreases_phishing | `-0.0620` |
| **Digit Ratio** | Character Distribution | `0.0` | decreases_phishing | `-0.0354` |
| **Special Character Count** | Character Distribution | `5` | decreases_phishing | `-0.0286` |
| **Hostname Length** | Length & Hierarchy | `10` | decreases_phishing | `-0.0234` |

**Human-Readable Summary:** Prediction: LEGITIMATE (BENIGN) (Estimated Phishing Probability: 0.30%). Baseline Training Prior: The model begins with an ensemble prior bias of 49.99%. Features Increasing Phishing Probability: 'Special Character Ratio' (value: 0.2778, attribution: +0.0150) shifted the prediction upward toward phishing. Features Decreasing Phishing Probability: 'HTTPS Scheme Indicator' (value: 1, attribution: -0.2282), 'Path Length' (value: 0, attribution: -0.0676), 'Slash Count' (value: 2, attribution: -0.0620) shifted the prediction downward toward legitimate. Academic Disclosure: Feature contributions reflect model internal attribution across the 27 static URL features and do not constitute proof of attacker intent or external webpage safety.

---

### Case: False Positive (Legitimate Flagged as Phishing)
- **URL:** `https://www.cns11643.gov.tw`
- **Ground Truth:** Legitimate
- **Model Prediction:** POTENTIAL PHISHING (Estimated Phishing Probability: **52.93%**)
- **Ensemble Prior Bias:** 49.99%
- **Reconstruction Error:** `2.22e-16`

| Contributing Feature | Category | Feature Value | Direction | Attribution |
| :--- | :--- | :--- | :--- | :--- |
| **Digit Ratio** | Character Distribution | `0.1852` | increases_phishing | `+0.2983` |
| **Digit Count** | Character Distribution | `5` | increases_phishing | `+0.2858` |
| **HTTPS Scheme Indicator** | Security & Protocol | `1` | decreases_phishing | `-0.2391` |
| **Slash Count** | Delimiter & Punctuation | `2` | decreases_phishing | `-0.1117` |
| **Path Length** | Length & Hierarchy | `0` | decreases_phishing | `-0.0927` |
| **URL Length** | Length & Hierarchy | `27` | decreases_phishing | `-0.0716` |

**Human-Readable Summary:** Prediction: POTENTIAL PHISHING (Estimated Phishing Probability: 52.93%). Baseline Training Prior: The model begins with an ensemble prior bias of 49.99%. Features Increasing Phishing Probability: 'Digit Ratio' (value: 0.1852, attribution: +0.2983), 'Digit Count' (value: 5, attribution: +0.2858), 'Subdomain Count' (value: 1, attribution: +0.0586) shifted the prediction upward toward phishing. Features Decreasing Phishing Probability: 'HTTPS Scheme Indicator' (value: 1, attribution: -0.2391), 'Slash Count' (value: 2, attribution: -0.1117), 'Path Length' (value: 0, attribution: -0.0927) shifted the prediction downward toward legitimate. Academic Disclosure: Feature contributions reflect model internal attribution across the 27 static URL features and do not constitute proof of attacker intent or external webpage safety.

---

## 2. False-Negative Explainability Study (Missed Phishing Links)

In Phase 5 and Phase 6, the model missed ~90 phishing URLs out of 20,104 test phishing samples. Here we explain the structural mechanics behind these misses:

#### False Negative Sample 1: `https://www.cfg.me`
- **Predicted Probability:** 0.30% (Classified as: LEGITIMATE (BENIGN))
- **Key Negative Contributors (Pushed toward Legitimate):**
  - **HTTPS Scheme Indicator** (Value: `1`): Contributed `-0.2282`
  - **Path Length** (Value: `0`): Contributed `-0.0676`
  - **Slash Count** (Value: `2`): Contributed `-0.0620`
  - **Digit Ratio** (Value: `0.0`): Contributed `-0.0354`
- **Key Positive Contributors (Pushed toward Phishing):**
  - **Special Character Ratio** (Value: `0.2778`): Contributed `+0.0150`

- **Observed Model Behavior:** The URL exhibited `has_https=1`, `path_length=0`, `num_slashes=2`, and `suspicious_keyword_present=0`. Because these features carry the highest global importance for legitimate classifications, their combination decisively pushed the predicted probability below the decision threshold.
- **Security Interpretation:** Adversaries who register clean domain names, deploy standard SSL certificates (HTTPS), and avoid credential tokens in the URL string successfully evade static lexical analysis. This establishes an intrinsic limitation of URL-only detection and motivates the multi-layered PhishGuard Risk Engine in Phase 8.

#### False Negative Sample 2: `https://www.michelonturismo.com.br`
- **Predicted Probability:** 0.98% (Classified as: LEGITIMATE (BENIGN))
- **Key Negative Contributors (Pushed toward Legitimate):**
  - **HTTPS Scheme Indicator** (Value: `1`): Contributed `-0.2615`
  - **Path Length** (Value: `0`): Contributed `-0.0884`
  - **Slash Count** (Value: `2`): Contributed `-0.0743`
  - **Digit Ratio** (Value: `0.0`): Contributed `-0.0458`
- **Key Positive Contributors (Pushed toward Phishing):**
  - **Subdomain Count** (Value: `1`): Contributed `+0.0666`
  - **Hostname Length** (Value: `26`): Contributed `+0.0103`

- **Observed Model Behavior:** The URL exhibited `has_https=1`, `path_length=0`, `num_slashes=2`, and `suspicious_keyword_present=0`. Because these features carry the highest global importance for legitimate classifications, their combination decisively pushed the predicted probability below the decision threshold.
- **Security Interpretation:** Adversaries who register clean domain names, deploy standard SSL certificates (HTTPS), and avoid credential tokens in the URL string successfully evade static lexical analysis. This establishes an intrinsic limitation of URL-only detection and motivates the multi-layered PhishGuard Risk Engine in Phase 8.

#### False Negative Sample 3: `https://www.tostudydrycleaning.ru`
- **Predicted Probability:** 0.25% (Classified as: LEGITIMATE (BENIGN))
- **Key Negative Contributors (Pushed toward Legitimate):**
  - **HTTPS Scheme Indicator** (Value: `1`): Contributed `-0.2015`
  - **Path Length** (Value: `0`): Contributed `-0.0675`
  - **Slash Count** (Value: `2`): Contributed `-0.0615`
  - **Digit Ratio** (Value: `0.0`): Contributed `-0.0347`
- **Key Positive Contributors (Pushed toward Phishing):**
  - *None: No feature exerted significant positive attribution.*

- **Observed Model Behavior:** The URL exhibited `has_https=1`, `path_length=0`, `num_slashes=2`, and `suspicious_keyword_present=0`. Because these features carry the highest global importance for legitimate classifications, their combination decisively pushed the predicted probability below the decision threshold.
- **Security Interpretation:** Adversaries who register clean domain names, deploy standard SSL certificates (HTTPS), and avoid credential tokens in the URL string successfully evade static lexical analysis. This establishes an intrinsic limitation of URL-only detection and motivates the multi-layered PhishGuard Risk Engine in Phase 8.

#### False Negative Sample 4: `https://www.netflix-vn.com`
- **Predicted Probability:** 0.50% (Classified as: LEGITIMATE (BENIGN))
- **Key Negative Contributors (Pushed toward Legitimate):**
  - **HTTPS Scheme Indicator** (Value: `1`): Contributed `-0.2498`
  - **Path Length** (Value: `0`): Contributed `-0.0710`
  - **Slash Count** (Value: `2`): Contributed `-0.0630`
  - **Digit Ratio** (Value: `0.0`): Contributed `-0.0418`
- **Key Positive Contributors (Pushed toward Phishing):**
  - **Hyphen Count** (Value: `1`): Contributed `+0.0442`
  - **Special Character Ratio** (Value: `0.2308`): Contributed `+0.0064`

- **Observed Model Behavior:** The URL exhibited `has_https=1`, `path_length=0`, `num_slashes=2`, and `suspicious_keyword_present=0`. Because these features carry the highest global importance for legitimate classifications, their combination decisively pushed the predicted probability below the decision threshold.
- **Security Interpretation:** Adversaries who register clean domain names, deploy standard SSL certificates (HTTPS), and avoid credential tokens in the URL string successfully evade static lexical analysis. This establishes an intrinsic limitation of URL-only detection and motivates the multi-layered PhishGuard Risk Engine in Phase 8.

#### False Negative Sample 5: `https://www.instagram-apple.com`
- **Predicted Probability:** 1.34% (Classified as: LEGITIMATE (BENIGN))
- **Key Negative Contributors (Pushed toward Legitimate):**
  - **HTTPS Scheme Indicator** (Value: `1`): Contributed `-0.2226`
  - **Path Length** (Value: `0`): Contributed `-0.0701`
  - **Slash Count** (Value: `2`): Contributed `-0.0639`
  - **Digit Ratio** (Value: `0.0`): Contributed `-0.0417`
- **Key Positive Contributors (Pushed toward Phishing):**
  - **Hyphen Count** (Value: `1`): Contributed `+0.0422`

- **Observed Model Behavior:** The URL exhibited `has_https=1`, `path_length=0`, `num_slashes=2`, and `suspicious_keyword_present=0`. Because these features carry the highest global importance for legitimate classifications, their combination decisively pushed the predicted probability below the decision threshold.
- **Security Interpretation:** Adversaries who register clean domain names, deploy standard SSL certificates (HTTPS), and avoid credential tokens in the URL string successfully evade static lexical analysis. This establishes an intrinsic limitation of URL-only detection and motivates the multi-layered PhishGuard Risk Engine in Phase 8.
