"""
PhishGuard Feature Engineering Demonstration & Statistical Validation.

Demonstrates extract_url_features on canonical phishing and benign URLs,
and validates feature distributions across a statistical sample of PhiUSIIL URLs.
"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extractor import extract_url_features, FEATURE_NAMES
from src.utils import DATA_RAW_DIR, setup_logger

logger = setup_logger("FeatureDemo")

CANONICAL_URLS = [
    ("Legitimate University", "https://www.cam.ac.uk"),
    ("Legitimate Search", "https://www.google.com/search?q=machine+learning"),
    ("Phishing - IP Address", "http://192.168.1.1/admin/login"),
    ("Phishing - Credential Harvesting", "http://secure-paypal-verify-account.update-login.com"),
    ("Phishing - Obfuscated UserInfo (@)", "http://paypal.com@malicious-phish.ru/verify"),
    ("Phishing - Subdomain Overload", "http://login.verification.banking.security.update.example-phish.net"),
    ("Shortened Link", "https://bit.ly/3xY9AbZ"),
]

def demonstrate_canonical_urls():
    print("=" * 80)
    print("           PhishGuard: Canonical URL Feature Extraction")
    print("=" * 80)
    for category, url in CANONICAL_URLS:
        feats = extract_url_features(url)
        print(f"\n[Category] {category}")
        print(f" URL: {url}")
        print(f"  -> Length: {feats['url_length']}, Hostname: {feats['hostname_length']}, Path: {feats['path_length']}")
        print(f"  -> HTTPS: {feats['has_https']}, IP: {feats['has_ip_address']}, @ Symbol: {feats['has_at_symbol']}")
        print(f"  -> Shortener: {feats['has_url_shortener']}, Subdomains: {feats['num_subdomains']}")
        print(f"  -> Keywords Present: {feats['suspicious_keyword_present']} (count: {feats['suspicious_keyword_count']})")
        print(f"  -> Dots: {feats['num_dots']}, Hyphens: {feats['num_hyphens']}, Digit Ratio: {feats['digit_ratio']}")

def validate_dataset_sample():
    csv_candidates = list(DATA_RAW_DIR.glob("*.csv"))
    if not csv_candidates:
        logger.warning("Raw dataset CSV not found. Skipping dataset statistical validation.")
        return

    logger.info("Computing feature separation on balanced sample of 2,000 URLs...")
    df = pd.read_csv(csv_candidates[0], usecols=["URL", "label"])

    # Sample 1000 benign (1) and 1000 phishing (0)
    benign_sample = df[df["label"] == 1].sample(1000, random_state=42)
    phish_sample = df[df["label"] == 0].sample(1000, random_state=42)

    df_sample = pd.concat([benign_sample, phish_sample]).reset_index(drop=True)
    # Target: 0 = Benign, 1 = Phishing
    df_sample["target"] = df_sample["label"].apply(lambda x: 1 if x == 0 else 0)

    extracted = [extract_url_features(u) for u in df_sample["URL"]]
    df_feats = pd.DataFrame(extracted)
    df_feats["target"] = df_sample["target"]

    summary = df_feats.groupby("target")[[
        "url_length", "has_https", "has_ip_address", "num_subdomains",
        "num_dots", "num_hyphens", "suspicious_keyword_present", "special_char_ratio"
    ]].mean().T

    summary.columns = ["Legitimate (Mean)", "Phishing (Mean)"]
    summary["Delta (Phish - Legit)"] = summary["Phishing (Mean)"] - summary["Legitimate (Mean)"]

    print("\n" + "=" * 80)
    print("    Statistical Feature Separation: 2,000 Sampled URLs (1,000 Phishing vs 1,000 Benign)")
    print("=" * 80)
    print(summary.round(4).to_string())
    print("=" * 80)

if __name__ == "__main__":
    demonstrate_canonical_urls()
    validate_dataset_sample()
