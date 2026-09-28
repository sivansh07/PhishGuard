"""
Unit tests for PhishGuard URL Feature Extractor.
Validates safe, robust static feature extraction across diverse URL topologies.
"""

import pytest
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extractor import extract_url_features, FEATURE_NAMES

def test_valid_benign_url():
    """Validates standard benign HTTPS URL extraction."""
    url = "https://www.example.com/about/contact.html"
    features = extract_url_features(url)

    assert isinstance(features, dict)
    assert features["has_https"] == 1
    assert features["has_ip_address"] == 0
    assert features["has_at_symbol"] == 0
    assert features["url_length"] == len(url)
    assert features["num_dots"] == 3
    assert features["num_subdomains"] == 0
    assert features["suspicious_keyword_present"] == 0

def test_malformed_url():
    """Validates that malformed URL strings fail gracefully without throwing exceptions."""
    url = "ht!tp:///??weird:::pattern%%%..com"
    features = extract_url_features(url)

    assert isinstance(features, dict)
    assert features["url_length"] == len(url)
    assert features["num_percent_signs"] == 3
    assert features["num_question_marks"] == 2
    assert features["is_encoded"] == 1

def test_empty_url():
    """Validates graceful handling of empty or None strings."""
    features = extract_url_features("")
    assert features["url_length"] == 0
    assert features["has_https"] == 0
    assert features["suspicious_keyword_present"] == 0

    features_none = extract_url_features(None)
    assert features_none["url_length"] == 0

def test_url_containing_ip():
    """Validates IP address detection in hostname."""
    url = "http://192.168.1.100/admin/panel"
    features = extract_url_features(url)

    assert features["has_ip_address"] == 1
    assert features["has_https"] == 0

def test_url_containing_at_symbol():
    """Validates detection of '@' credential trick in URL."""
    url = "http://google.com@phishing-target.com/login"
    features = extract_url_features(url)

    assert features["has_at_symbol"] == 1
    assert features["suspicious_keyword_present"] == 1
    assert features["suspicious_keyword_count"] >= 1

def test_https_url():
    """Validates HTTPS scheme detection."""
    https_url = "https://secure-portal.com"
    http_url = "http://insecure-portal.com"

    assert extract_url_features(https_url)["has_https"] == 1
    assert extract_url_features(http_url)["has_https"] == 0

def test_suspicious_keyword_url():
    """Validates multi-keyword detection."""
    url = "http://verify-bank-account-security-update.com/signin/confirm"
    features = extract_url_features(url)

    assert features["suspicious_keyword_present"] == 1
    # verify, bank, account, security, update, signin, confirm -> 7 keywords
    assert features["suspicious_keyword_count"] >= 5

def test_long_url():
    """Validates extraction on abnormally long URLs commonly used in payload embedding."""
    url = "https://example.com/login?" + "token=" + "a" * 500
    features = extract_url_features(url)

    assert features["url_length"] > 500
    assert features["has_query"] == 1
    assert features["num_equal_signs"] == 1

def test_url_shortener():
    """Validates identification of known URL shortening domains."""
    url = "https://bit.ly/3xY9AbZ"
    features = extract_url_features(url)

    assert features["has_url_shortener"] == 1

def test_subdomains():
    """Validates subdomain count calculation."""
    url = "http://portal.login.accounts.service.example.com"
    features = extract_url_features(url)

    # subdomains: portal, login, accounts, service -> 4 subdomains
    assert features["num_subdomains"] >= 3

def test_feature_consistency():
    """Ensures all extracted dictionaries match the canonical FEATURE_NAMES schema."""
    url = "https://test.com"
    features = extract_url_features(url)
    assert set(features.keys()) == set(FEATURE_NAMES)
