"""
PhishGuard Utility Functions.

Provides filesystem paths, configuration constants, logging helpers,
and safe string normalization functions.
"""

from pathlib import Path
import logging

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Standard Directories
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_RESULTS_DIR = PROJECT_ROOT / "reports" / "results"

def setup_logger(name: str = "PhishGuard", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a standardized console logger.

    Args:
        name: Name of the logger instance.
        level: Logging severity level (e.g. logging.INFO).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

def ensure_directories_exist() -> None:
    """Verifies that all standard project directories exist on disk."""
    for d in [
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        MODELS_DIR,
        REPORTS_FIGURES_DIR,
        REPORTS_RESULTS_DIR
    ]:
        d.mkdir(parents=True, exist_ok=True)
