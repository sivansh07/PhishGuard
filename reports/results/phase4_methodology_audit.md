# PhishGuard: Phase 4 Methodological & Data Leakage Audit Report

**Experiment Label:** Standard Random Stratified Hold-Out Evaluation  
**Status:** PASSED - No data leakage detected. Baselines established cleanly.  

---

## 1. Data Leakage & Feature Independence Verification

| Verification Item | Requirement | Audit Result | Status |
| :--- | :--- | :--- | :--- |
| **Target Feature Leakage** | Target/Label cannot be used as an input feature | No target/label column present in `FEATURE_NAMES` | **PASS** |
| **Webpage Feature Exclusion** | Exclude all 29 HTML/DOM/JS features | 0 excluded webpage features present in feature set | **PASS** |
| **Feature Extraction Isolation** | Operates strictly on raw URL string | `extract_url_features(url)` takes string only, no external calls | **PASS** |
| **Dataset Deduplication** | Remove duplicate URLs prior to splitting | 425 duplicate URLs dropped prior to train/test split | **PASS** |
| **Scaler Isolation** | Scaling fitted strictly on training data | `StandardScaler` encapsulated in scikit-learn `Pipeline` | **PASS** |

---

## 2. Canonical 27 URL-Derived Features

The feature extractor produces exactly **27 features**:
```
url_length, hostname_length, path_length, query_length, num_dots, num_slashes, num_hyphens, num_underscores, num_digits, num_special_chars, num_question_marks, num_equal_signs, num_ampersands, num_percent_signs, has_https, has_ip_address, has_at_symbol, has_port, has_fragment, has_query, has_url_shortener, num_subdomains, suspicious_keyword_count, suspicious_keyword_present, digit_ratio, special_char_ratio, is_encoded
```
*Extractor schema matches pipeline training input:* **YES**

---

## 3. Probability Terminology & Calibration Disclosure

* **Explicit Calibration:** **False**
* **Method:** Classifier raw predict_proba() (sigmoid output for LogisticRegression; leaf empirical fractions for DecisionTree / RandomForest).
* **Terminology Standard:** All outputs and documentation refer to **"predicted class probability"** or **"predicted phishing probability"**, never claiming post-hoc calibration (e.g. Platt scaling / Isotonic regression).

---

## 4. Random Forest Feature Importance Methodology

* **Calculation:** Normalized Mean Decrease in Impurity (Gini Importance) across all 100 decision trees.
* **Top Discriminator:** `has_https` (44.46%)
* **Methodological Caution:** Gini importance reflects empirical correlation and node split purity within the training set. It is **not** causal proof that protocol type alone defines a phishing attack.

---

## 5. Domain Overlap Audit (Explaining the 99.8% Random-Split Metric)

In this standard random stratified hold-out evaluation:
* **Training Instances:** 188,296 (176,979 unique domains)
* **Testing Instances:** 47,074 (45,186 unique domains)
* **Overlapping Domains:** **2,079 domains**
* **Test URLs Belonging to Seen Domains:** **3,785 / 47,074 (8.04%)**

### Key Methodological Insight:
While **91.96%** of test URLs belong to domains exclusive to the test partition, **8.04%** share domains with the training partition. More importantly, benign URLs in this benchmark exhibit strong syntactic regularity (consistent path structures and 100% HTTPS adoption), allowing tree-based classifiers to partition the feature space with extreme precision. This highlights the necessity of the **Unseen-Domain Grouped Evaluation** scheduled for Phase 6.

---

## 6. Diagnostic HTTPS Ablation Experiment

To verify that the model is not merely a superficial "HTTPS detector", we trained and compared two identical Random Forest models:

| Metric | Model A (All 27 Features) | Model B (Without `has_https` - 26 Features) | Delta (Model B - Model A) |
| :--- | :--- | :--- | :--- |
| **Number of Features** | 27 | 26 | -1 |
| **Accuracy** | 0.9979 | 0.9932 | -0.0047 |
| **Precision** | 0.9997 | 0.9960 | -0.0037 |
| **Recall (Phishing)** | 0.9955 | 0.9881 | -0.0074 |
| **F1-Score** | 0.9976 | 0.9920 | -0.0056 |
| **ROC-AUC** | 0.9982 | 0.9985 | +0.0003 |
| **False Negatives (Missed Phish)** | 90 | 239 | **+149** |
| **False Positives (False Alarms)** | 7 | 80 | +73 |

### Ablation Takeaway:
* Without `has_https`, accuracy drops by only **0.47%** (to 99.32%) and recall remains at **98.81%**.
* However, False Negatives increase from 90 to 239 (+149 missed phishing attacks).
* **Conclusion:** The model utilizes the rich combination of the remaining 26 features (path length, slash counts, digit ratios, subdomains, special character counts) to identify threats, confirming structural and lexical efficacy.

---

## 7. Inference Pipeline Consistency

| Test URL | Prediction | Predicted Phish Probability | Features Count | Schema Match |
| :--- | :--- | :--- | :--- | :--- |
| `https://www.google.com` | LEGITIMATE (BENIGN) | 0.0026 | 27 | **PASS** |
| `http://verify-bank-update.com/login` | POTENTIAL PHISHING | 1.0 | 27 | **PASS** |
| `http://192.168.0.1/admin` | POTENTIAL PHISHING | 1.0 | 27 | **PASS** |

---

## 8. Operational Security Language Standards

* We do not claim a formal mathematical cost-sensitive loss matrix has been fitted.
* Instead, we emphasize the **operational security asymmetry**: A false negative allows an adversary to harvest credentials or execute exploits, whereas a false positive generates manageable user friction.
