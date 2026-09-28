"""
PhishGuard Inference Engine.

Loads the serialized model pipeline, extracts features from raw URL strings,
and provides predicted phishing probability and binary predictions.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extractor import extract_url_features, FEATURE_NAMES
from src.model import load_pipeline, DEFAULT_MODEL_FILE

class PhishGuardPredictor:
    """End-to-end inference handler for incoming URLs."""

    def __init__(self, model_path: Optional[Path] = None):
        """Initializes predictor and loads serialized model pipeline.

        Args:
            model_path: Optional path to joblib model. Defaults to DEFAULT_MODEL_FILE.
        """
        self.model_path = model_path or DEFAULT_MODEL_FILE
        self.pipeline = load_pipeline(self.model_path)
        self.model_name = self.pipeline.named_steps["classifier"].__class__.__name__

    def predict(self, url: str) -> Dict[str, Any]:
        """Runs static feature extraction and model prediction on a URL string.

        Args:
            url: The raw URL string to test.

        Returns:
            Dictionary containing prediction, probability, and extracted features.
        """
        features_dict = extract_url_features(url)
        features_df = pd.DataFrame([features_dict], columns=FEATURE_NAMES)

        pred_class = int(self.pipeline.predict(features_df)[0])
        pred_proba = float(self.pipeline.predict_proba(features_df)[0, 1])

        label = "POTENTIAL PHISHING" if pred_class == 1 else "LEGITIMATE (BENIGN)"

        return {
            "url": url,
            "prediction": label,
            "is_phishing": bool(pred_class == 1),
            "phishing_probability": round(pred_proba, 4),
            "phishing_probability_pct": round(pred_proba * 100, 2),
            "model_used": self.model_name,
            "features": features_dict
        }
