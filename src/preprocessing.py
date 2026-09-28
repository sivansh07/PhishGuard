"""
PhishGuard Dataset Preprocessing and Data Cleaning Module.

Handles loading of the raw dataset, target mapping, duplicate removal,
URL feature matrix extraction, caching, and stratified train/test splitting.
"""

import sys
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import DATA_RAW_DIR, DATA_PROCESSED_DIR, setup_logger
from src.feature_extractor import extract_url_features, FEATURE_NAMES

logger = setup_logger("Preprocessing")

PROCESSED_DATA_FILE = DATA_PROCESSED_DIR / "phiusiil_features_processed.parquet"
PROCESSED_DATA_CSV = DATA_PROCESSED_DIR / "phiusiil_features_processed.csv"

def load_and_clean_raw_data(csv_path: Optional[Path] = None, drop_duplicates: bool = True) -> pd.DataFrame:
    """Loads raw dataset, standardizes target label, and removes duplicate URLs.

    Args:
        csv_path: Path to raw CSV file. Defaults to first CSV in DATA_RAW_DIR.
        drop_duplicates: Whether to drop duplicate URL rows.

    Returns:
        Cleaned DataFrame containing ['URL', 'Domain', 'target'].
    """
    if csv_path is None:
        csv_files = list(DATA_RAW_DIR.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV file discovered in {DATA_RAW_DIR}.")
        csv_path = csv_files[0]

    logger.info(f"Loading raw data from {csv_path.name}...")
    df = pd.read_csv(csv_path, usecols=["URL", "Domain", "label"])
    initial_count = len(df)

    if drop_duplicates:
        df = df.drop_duplicates(subset=["URL"]).reset_index(drop=True)
        dropped_count = initial_count - len(df)
        logger.info(f"Removed {dropped_count} duplicate URLs ({initial_count} -> {len(df)} records).")

    # Cybersecurity Standard Target Mapping:
    # In raw PhiUSIIL: 1 = Legitimate, 0 = Phishing.
    # We map: 1 = Phishing (Malicious / Positive threat class), 0 = Legitimate (Benign).
    df["target"] = df["label"].apply(lambda x: 1 if x == 0 else 0)
    df.drop(columns=["label"], inplace=True)

    phish_count = (df["target"] == 1).sum()
    legit_count = (df["target"] == 0).sum()
    logger.info(f"Class distribution: Legitimate (0) = {legit_count:,}, Phishing (1) = {phish_count:,}")

    return df

def build_feature_dataframe(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Extracts the 27 URL-derived features for every URL in the dataframe.

    Uses extract_url_features from src.feature_extractor exclusively.
    Does NOT use any webpage-dependent features.

    Args:
        df_clean: Cleaned DataFrame with ['URL', 'Domain', 'target'].

    Returns:
        DataFrame containing all 27 extracted features, 'Domain', and 'target'.
    """
    logger.info(f"Extracting 27 lexical and structural features for {len(df_clean):,} URLs...")
    extracted_records = [extract_url_features(url) for url in df_clean["URL"]]

    df_feats = pd.DataFrame(extracted_records, columns=FEATURE_NAMES)
    df_feats["Domain"] = df_clean["Domain"].values
    df_feats["target"] = df_clean["target"].values

    logger.info(f"Feature matrix built successfully. Dimensions: {df_feats.shape}")
    return df_feats

def get_or_create_processed_dataset(force_recompute: bool = False) -> pd.DataFrame:
    """Retrieves processed feature dataset from cache or extracts and caches it.

    Args:
        force_recompute: If True, ignores existing cache and re-extracts features.

    Returns:
        Processed feature DataFrame.
    """
    if not force_recompute and PROCESSED_DATA_FILE.exists():
        logger.info(f"Loading cached feature dataset from {PROCESSED_DATA_FILE}...")
        try:
            return pd.read_parquet(PROCESSED_DATA_FILE)
        except Exception as e:
            logger.warning(f"Failed to read parquet ({e}). Rebuilding feature dataset...")

    # Build fresh dataset
    df_clean = load_and_clean_raw_data()
    df_processed = build_feature_dataframe(df_clean)

    # Save cache
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df_processed.to_parquet(PROCESSED_DATA_FILE, index=False)
        logger.info(f"Cached processed dataset to {PROCESSED_DATA_FILE}")
    except Exception as e:
        logger.warning(f"Could not save parquet ({e}). Saving as CSV...")
        df_processed.to_csv(PROCESSED_DATA_CSV, index=False)
        logger.info(f"Cached processed dataset to {PROCESSED_DATA_CSV}")

    return df_processed

def get_train_test_data(
    test_size: float = 0.2,
    random_state: int = 42,
    force_recompute: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepares stratified train-test splits on the 27 URL-derived features.

    Ensures no feature scaling is performed globally to prevent data leakage.

    Args:
        test_size: Proportion of dataset for test split (default 0.2).
        random_state: Random seed for reproducibility.
        force_recompute: Whether to re-extract features from raw dataset.

    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    df = get_or_create_processed_dataset(force_recompute=force_recompute)

    X = df[FEATURE_NAMES]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info(f"Train split: {X_train.shape[0]:,} samples | Test split: {X_test.shape[0]:,} samples")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df_proc = get_or_create_processed_dataset()
    print("Processed Dataset Summary:")
    print(df_proc.info())
    print("\nTarget Balance:")
    print(df_proc["target"].value_counts(normalize=True))
