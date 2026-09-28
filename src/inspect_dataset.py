"""
PhishGuard Dataset Inspection & Exploration Script.

Inspects the raw PhiUSIIL dataset, identifies URL/target columns, evaluates
data quality, separates URL-only static features from webpage-dependent features,
and outputs statistical summaries and class distribution charts.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    DATA_RAW_DIR,
    REPORTS_FIGURES_DIR,
    REPORTS_RESULTS_DIR,
    setup_logger,
    ensure_directories_exist
)

logger = setup_logger("DatasetInspection")

def inspect_phiusiil_dataset() -> dict:
    """Performs rigorous inspection and categorization of the raw dataset.

    Returns:
        Dictionary containing comprehensive inspection findings.
    """
    ensure_directories_exist()
    csv_candidates = list(DATA_RAW_DIR.glob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No CSV file found in {DATA_RAW_DIR}. Run download_dataset.py first.")

    dataset_path = csv_candidates[0]
    logger.info(f"Loading raw dataset from {dataset_path}...")

    # Load dataset
    df = pd.read_csv(dataset_path)
    total_rows, total_cols = df.shape
    logger.info(f"Dataset successfully loaded. Dimensions: {total_rows} rows x {total_cols} columns.")

    # 1. Identify URL and Label columns
    url_col = "URL" if "URL" in df.columns else None
    label_col = "label" if "label" in df.columns else None

    if not url_col or not label_col:
        raise ValueError("Critical columns (URL or label) not found in dataset!")

    # 2. Check label distribution in raw data
    raw_label_counts = df[label_col].value_counts().to_dict()
    # In PhiUSIIL: 1 = Legitimate (134,850), 0 = Phishing (100,945)
    legitimate_count = raw_label_counts.get(1, 0)
    phishing_count = raw_label_counts.get(0, 0)

    # 3. Data Integrity & Missing Values
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0].to_dict()

    duplicate_urls = df[url_col].duplicated().sum()
    unique_domains = df["Domain"].nunique() if "Domain" in df.columns else None

    # 4. Feature Taxonomy & Modality Classification
    webpage_dependent_features = [
        "LineOfCode", "LargestLineLength", "HasTitle", "Title",
        "DomainTitleMatchScore", "URLTitleMatchScore", "HasFavicon",
        "Robots", "IsResponsive", "NoOfURLRedirect", "NoOfSelfRedirect",
        "HasDescription", "NoOfPopup", "NoOfiFrame", "HasExternalFormSubmit",
        "HasSocialNet", "HasSubmitButton", "HasHiddenFields",
        "HasPasswordField", "Bank", "Pay", "Crypto", "HasCopyrightInfo",
        "NoOfImage", "NoOfCSS", "NoOfJS", "NoOfSelfRef", "NoOfEmptyRef",
        "NoOfExternalRef"
    ]

    url_lexical_dataset_features = [
        "URLLength", "DomainLength", "IsDomainIP", "TLD", "URLSimilarityIndex",
        "CharContinuationRate", "TLDLegitimateProb", "URLCharProb", "TLDLength",
        "NoOfSubDomain", "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio",
        "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL",
        "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
        "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL", "IsHTTPS"
    ]

    identifier_features = ["FILENAME", "URL", "Domain"]

    # 5. Generate Inspection Report Dict
    report = {
        "dataset_filename": dataset_path.name,
        "dataset_size_mb": round(dataset_path.stat().st_size / (1024 * 1024), 2),
        "total_records": int(total_rows),
        "total_columns": int(total_cols),
        "url_column": url_col,
        "label_column": label_col,
        "raw_label_distribution": {
            "raw_1_legitimate": int(legitimate_count),
            "raw_0_phishing": int(phishing_count)
        },
        "mapped_target_standard": {
            "0_legitimate": int(legitimate_count),
            "1_phishing": int(phishing_count),
            "phishing_percentage": round((phishing_count / total_rows) * 100, 2),
            "legitimate_percentage": round((legitimate_count / total_rows) * 100, 2)
        },
        "unique_urls": int(df[url_col].nunique()),
        "duplicate_urls": int(duplicate_urls),
        "unique_domains": int(unique_domains) if unique_domains else "N/A",
        "columns_with_missing_values": cols_with_nulls,
        "feature_taxonomy": {
            "identifiers_count": len(identifier_features),
            "identifiers": identifier_features,
            "url_lexical_features_in_dataset_count": len(url_lexical_dataset_features),
            "url_lexical_features_in_dataset": url_lexical_dataset_features,
            "webpage_dependent_features_count": len(webpage_dependent_features),
            "webpage_dependent_features": webpage_dependent_features,
            "exclusion_rationale": (
                "Webpage-dependent features (DOM elements, scripts, titles, images, form submits) "
                "require dynamically visiting and rendering external sites. In a zero-day detection context, "
                "this creates severe security hazards (drive-by downloads) and fails when phishing pages have "
                "already been taken offline. Therefore, PhishGuard strictly restricts analysis to client-safe, "
                "zero-execution URL lexical and structural indicators."
            )
        }
    }

    # Save JSON report
    report_file = REPORTS_RESULTS_DIR / "dataset_inspection_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    logger.info(f"Inspection report saved to {report_file}")

    # 6. Generate and save Class Distribution Figure
    plot_class_distribution(legitimate_count, phishing_count)

    return report

def plot_class_distribution(legitimate_count: int, phishing_count: int) -> None:
    """Generates an academic-grade figure visualizing dataset class distribution."""
    plt.figure(figsize=(7, 5), dpi=300)
    sns.set_theme(style="whitegrid")

    categories = ["Legitimate (Benign)", "Phishing (Malicious)"]
    counts = [legitimate_count, phishing_count]
    colors = ["#2b8a3e", "#c92a2a"]

    bars = plt.bar(categories, counts, color=colors, width=0.5, edgecolor="black", linewidth=1.2)

    for bar in bars:
        height = bar.get_height()
        percentage = (height / sum(counts)) * 100
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (sum(counts) * 0.015),
            f"{height:,}\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    plt.title("PhiUSIIL Dataset: Ground Truth Class Distribution", fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Number of URL Samples", fontsize=11)
    plt.ylim(0, max(counts) * 1.18)
    plt.tight_layout()

    fig_path = REPORTS_FIGURES_DIR / "01_class_distribution.png"
    plt.savefig(fig_path)
    plt.close()
    logger.info(f"Class distribution chart saved to {fig_path}")

if __name__ == "__main__":
    report = inspect_phiusiil_dataset()
    print("\n" + "=" * 65)
    print("        PhishGuard: Phase 2 Dataset Inspection Summary")
    print("=" * 65)
    print(f"Dataset File:       {report['dataset_filename']} ({report['dataset_size_mb']} MB)")
    print(f"Total URL Records:  {report['total_records']:,}")
    print(f"Unique Domains:     {report['unique_domains']:,}")
    print(f"Duplicate URLs:     {report['duplicate_urls']:,}")
    print("\nTarget Distribution (Cybersecurity Standard):")
    print(f"  - Legitimate (0): {report['mapped_target_standard']['0_legitimate']:,} ({report['mapped_target_standard']['legitimate_percentage']}%)")
    print(f"  - Phishing   (1): {report['mapped_target_standard']['1_phishing']:,} ({report['mapped_target_standard']['phishing_percentage']}%)")
    print("\nFeature Taxonomy:")
    print(f"  - Identifiers:               {report['feature_taxonomy']['identifiers_count']} columns")
    print(f"  - Pre-computed URL Lexical:  {report['feature_taxonomy']['url_lexical_features_in_dataset_count']} columns")
    print(f"  - Webpage-Dependent:         {report['feature_taxonomy']['webpage_dependent_features_count']} columns [EXCLUDED]")
    print("=" * 65)
