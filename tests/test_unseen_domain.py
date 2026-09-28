"""
Unit and Integration Tests for Unseen-Domain Grouped Evaluation.
Verifies domain partition separation, zero domain leakage, and feature schema consistency.
"""

import pytest
import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extractor import FEATURE_NAMES
from src.preprocessing import get_or_create_processed_dataset

@pytest.fixture(scope="module")
def processed_data():
    """Loads cached processed feature matrix."""
    df = get_or_create_processed_dataset()
    return df

def test_feature_count(processed_data):
    """Verifies that exactly 27 static URL features are present in the dataset schema."""
    assert len(FEATURE_NAMES) == 27
    for f in FEATURE_NAMES:
        assert f in processed_data.columns

def test_zero_domain_overlap(processed_data):
    """Verifies that StratifiedGroupKFold produces partitions with strictly ZERO domain overlap."""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(sgkf.split(processed_data, processed_data["target"], groups=processed_data["Domain"]))

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_domains = set(processed_data.loc[train_idx, "Domain"])
        test_domains = set(processed_data.loc[test_idx, "Domain"])
        overlap = train_domains.intersection(test_domains)

        assert len(overlap) == 0, f"Fold {fold_idx + 1} has {len(overlap)} overlapping domains!"

def test_grouped_split_integrity(processed_data):
    """Verifies that train and test indices sum up to the total dataset size without duplication."""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(sgkf.split(processed_data, processed_data["target"], groups=processed_data["Domain"]))

    train_idx, test_idx = splits[0]
    total_len = len(train_idx) + len(test_idx)
    assert total_len == len(processed_data)

    index_intersection = set(train_idx).intersection(set(test_idx))
    assert len(index_intersection) == 0

def test_stratified_balance(processed_data):
    """Verifies that grouped splitting preserves the approximate class ratio in both partitions."""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(sgkf.split(processed_data, processed_data["target"], groups=processed_data["Domain"]))

    train_idx, test_idx = splits[0]
    train_phish_pct = processed_data.loc[train_idx, "target"].mean()
    test_phish_pct = processed_data.loc[test_idx, "target"].mean()

    # Both partitions should closely approximate the overall 42.7% phishing ratio
    assert abs(train_phish_pct - 0.427) < 0.01
    assert abs(test_phish_pct - 0.427) < 0.01
