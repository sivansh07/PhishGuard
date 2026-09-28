"""
PhishGuard URL Feature Extraction Module.

Extracts lexical, structural, and security indicators directly from URL strings
without executing, fetching, or visiting suspicious endpoints.
"""

import re
from urllib.parse import urlparse, urlsplit
from typing import Dict, Any, Tuple

# Common URL shortening domains used to obscure destinations
KNOWN_SHORTENERS = frozenset({
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "bit.do", "cutt.ly", "shorturl.at",
    "tiny.cc", "rebrand.ly", "lnkd.in", "t.ly", "v.gd"
})

# High-frequency social engineering and credential harvesting keywords
SUSPICIOUS_KEYWORDS = (
    "login", "signin", "verify", "verification", "account",
    "update", "secure", "security", "password", "bank",
    "payment", "confirm", "wallet", "authenticate"
)

# Compiled regex patterns for high performance and thread safety
IPV4_REGEX = re.compile(
    r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
    r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
)

SPECIAL_CHARS_REGEX = re.compile(r"[^a-zA-Z0-9\s]")

def normalize_url(raw_url: str) -> Tuple[str, bool]:
    """Safely normalizes schemeless user input by prepending 'https://' by default.

    Preserves the hostname exactly and does NOT alter already scheme-qualified URLs.
    Does NOT add 'www.' automatically.

    Args:
        raw_url: The incoming raw URL string.

    Returns:
        Tuple of (normalized_url: str, was_schemeless: bool).
    """
    if not isinstance(raw_url, str):
        raw_url = "" if raw_url is None else str(raw_url)

    url = raw_url.strip()
    if not url:
        return "", False

    # If scheme indicator is already present, leave unchanged
    if "://" in url or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        return url, False

    return f"https://{url}", True

def is_apex_domain_without_www(url: str) -> bool:
    """Determines whether a URL targets an apex domain without the 'www.' prefix.

    The PhishGuard benchmark dataset curated legitimate URLs exclusively in the
    canonical 'https://www.domain.tld' format. Apex-domain URLs (e.g.
    https://github.com) are not represented as legitimate samples in the training
    dataset.

    Args:
        url: The URL string to inspect.

    Returns:
        True if the hostname is a non-IP domain that omits 'www.' and has no subdomains.
    """
    if not url:
        return False

    norm_url, _ = normalize_url(url)
    try:
        parsed = urlsplit(norm_url)
        hostname = (parsed.hostname or "").lower()
    except Exception:
        return False

    if not hostname or IPV4_REGEX.match(hostname):
        return False

    if hostname.startswith("www."):
        return False

    # Split domain labels
    parts = hostname.split(".")
    if len(parts) < 2:
        return False

    # Check subdomain count matching canonical feature definition
    subdomains = parts
    if subdomains and subdomains[0] == "www":
        subdomains = subdomains[1:]
    num_subdomains = max(0, len(subdomains) - 2) if len(subdomains) >= 2 else 0

    return num_subdomains == 0

def _normalize_and_split_url(raw_url: str):
    """Safely normalizes and parses a URL string without network calls.

    Args:
        raw_url: The incoming URL string.

    Returns:
        Tuple of (normalized_url, parsed_split_result, hostname)
    """
    norm_url, _ = normalize_url(raw_url)

    if not norm_url:
        try:
            parsed = urlsplit("http://empty.invalid")
        except Exception:
            parsed = urlsplit("")
        return "", parsed, ""

    try:
        parsed = urlsplit(norm_url)
        hostname = (parsed.hostname or "").lower()
    except Exception:
        # Fallback for heavily malformed URL strings
        parsed = urlsplit("http://malformed.invalid")
        hostname = ""

    return norm_url, parsed, hostname

def extract_url_features(url: str) -> Dict[str, Any]:
    """Safely extracts comprehensive lexical and structural features from a URL string.

    Args:
        url: The raw URL string to analyze.

    Returns:
        Dictionary mapping feature names to numerical (int/float) values.
    """
    url_str, parsed, hostname = _normalize_and_split_url(url)
    url_len = len(url_str)

    if url_len == 0:
        # Graceful fallback for empty URL strings
        return {
            "url_length": 0,
            "hostname_length": 0,
            "path_length": 0,
            "query_length": 0,
            "num_dots": 0,
            "num_slashes": 0,
            "num_hyphens": 0,
            "num_underscores": 0,
            "num_digits": 0,
            "num_special_chars": 0,
            "num_question_marks": 0,
            "num_equal_signs": 0,
            "num_ampersands": 0,
            "num_percent_signs": 0,
            "has_https": 0,
            "has_ip_address": 0,
            "has_at_symbol": 0,
            "has_port": 0,
            "has_fragment": 0,
            "has_query": 0,
            "has_url_shortener": 0,
            "num_subdomains": 0,
            "suspicious_keyword_count": 0,
            "suspicious_keyword_present": 0,
            "digit_ratio": 0.0,
            "special_char_ratio": 0.0,
            "is_encoded": 0
        }

    # Component lengths
    path_len = len(parsed.path) if parsed.path else 0
    query_len = len(parsed.query) if parsed.query else 0
    hostname_len = len(hostname)

    # Basic character counts
    num_dots = url_str.count(".")
    num_slashes = url_str.count("/")
    num_hyphens = url_str.count("-")
    num_underscores = url_str.count("_")
    num_digits = sum(1 for c in url_str if c.isdigit())
    num_question_marks = url_str.count("?")
    num_equal_signs = url_str.count("=")
    num_ampersands = url_str.count("&")
    num_percent_signs = url_str.count("%")
    num_special_chars = len(SPECIAL_CHARS_REGEX.findall(url_str))

    # Security-related indicators
    has_https = 1 if url_str.lower().startswith("https://") else 0
    has_ip = 1 if IPV4_REGEX.match(hostname) else 0
    has_at = 1 if "@" in url_str else 0

    has_port = 0
    if parsed.netloc and ":" in parsed.netloc:
        port_part = parsed.netloc.split(":")[-1]
        if port_part.isdigit() and port_part not in ("80", "443"):
            has_port = 1

    has_fragment = 1 if bool(parsed.fragment) else 0
    has_query = 1 if query_len > 0 else 0

    # Shortener detection
    base_domain = ".".join(hostname.split(".")[-2:]) if "." in hostname else hostname
    has_shortener = 1 if (hostname in KNOWN_SHORTENERS or base_domain in KNOWN_SHORTENERS) else 0

    # Subdomain calculation
    # e.g., "login.bank.example.com" -> subdomains: ["login", "bank"] -> count = 2
    # e.g., "www.example.com" -> strip www -> count = 0
    subdomains = hostname.split(".")
    if subdomains and subdomains[0] == "www":
        subdomains = subdomains[1:]
    # Subtract domain name and TLD (2 parts) if present
    num_subdomains = max(0, len(subdomains) - 2) if len(subdomains) >= 2 else 0

    # Suspicious text / keyword detection
    url_lower = url_str.lower()
    keyword_count = sum(url_lower.count(kw) for kw in SUSPICIOUS_KEYWORDS)
    keyword_present = 1 if keyword_count > 0 else 0

    # Ratios
    digit_ratio = round(num_digits / url_len, 4) if url_len > 0 else 0.0
    special_char_ratio = round(num_special_chars / url_len, 4) if url_len > 0 else 0.0
    is_encoded = 1 if num_percent_signs > 0 else 0

    return {
        "url_length": url_len,
        "hostname_length": hostname_len,
        "path_length": path_len,
        "query_length": query_len,
        "num_dots": num_dots,
        "num_slashes": num_slashes,
        "num_hyphens": num_hyphens,
        "num_underscores": num_underscores,
        "num_digits": num_digits,
        "num_special_chars": num_special_chars,
        "num_question_marks": num_question_marks,
        "num_equal_signs": num_equal_signs,
        "num_ampersands": num_ampersands,
        "num_percent_signs": num_percent_signs,
        "has_https": has_https,
        "has_ip_address": has_ip,
        "has_at_symbol": has_at,
        "has_port": has_port,
        "has_fragment": has_fragment,
        "has_query": has_query,
        "has_url_shortener": has_shortener,
        "num_subdomains": num_subdomains,
        "suspicious_keyword_count": keyword_count,
        "suspicious_keyword_present": keyword_present,
        "digit_ratio": digit_ratio,
        "special_char_ratio": special_char_ratio,
        "is_encoded": is_encoded
    }

# Canonical list of feature names in fixed order for ML pipelines
FEATURE_NAMES = list(extract_url_features("https://example.com").keys())
