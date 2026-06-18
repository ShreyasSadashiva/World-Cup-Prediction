"""
XGBoost classifier for match outcome prediction.

Outcomes: H (home win) / D (draw) / A (away win)
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

MODEL_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_PATH = MODEL_DIR / "wc_predictor.joblib"

_CLASSES = ["A", "D", "H"]
_LABEL_MAP = {"H": "Home Win", "D": "Draw", "A": "Away Win"}


def _model_dir() -> Path:
    MODEL_DIR.mkdir(exist_ok=True)
    return MODEL_DIR


def save(model, feature_cols: list[str]) -> None:
    _model_dir()
    le = LabelEncoder().fit(_CLASSES)
    joblib.dump({"model": model, "encoder": le, "feature_cols": feature_cols}, MODEL_PATH)
    print(f"[Model] Saved → {MODEL_PATH}")


def load():
    """Return (model, encoder, feature_cols) or (None, None, None) if not trained."""
    if not MODEL_PATH.exists():
        return None, None, None
    data = joblib.load(MODEL_PATH)
    return data["model"], data["encoder"], data["feature_cols"]


def is_trained() -> bool:
    return MODEL_PATH.exists()


def predict(features: dict) -> dict:
    """
    Given a feature dict, return probabilities and predicted outcome.

    Returns:
        {
            "home_win": float,
            "draw":     float,
            "away_win": float,
            "predicted": "Home Win" | "Draw" | "Away Win",
            "confidence": float,
        }
    """
    model, encoder, feature_cols = load()
    if model is None:
        return {
            "home_win": 0.40, "draw": 0.25, "away_win": 0.35,
            "predicted": "Home Win", "confidence": 0.40,
            "model_available": False,
        }

    X = pd.DataFrame([features])[feature_cols].fillna(0.0)
    proba = model.predict_proba(X)[0]
    classes = encoder.classes_  # ["A", "D", "H"] in sorted order

    result: dict = {"model_available": True}
    for cls, prob in zip(classes, proba):
        if cls == "H":
            result["home_win"] = float(prob)
        elif cls == "D":
            result["draw"] = float(prob)
        else:
            result["away_win"] = float(prob)

    best_idx = int(np.argmax(proba))
    result["predicted"] = _LABEL_MAP[classes[best_idx]]
    result["confidence"] = float(proba[best_idx])
    return result
