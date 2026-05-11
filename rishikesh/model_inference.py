"""
model_inference.py
------------------
Core inference engine for the Behavioral Risk Detection Engine.

The main model is loaded ONCE at module import time. All calls to
get_risk_score() reuse the same in-memory model object — no per-call disk I/O.

CRITICAL: The Feature_Vector must NEVER be re-scaled. Mugeesh's pipeline
delivers values that are already normalized. Passing them directly to the
model is the correct behavior.

Public API
----------
get_risk_score(features: list) -> dict
    Accepts a list of exactly 20 numeric values and returns a 6-field
    Risk_Prediction dict.
"""

import math
import os

import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Module-level model loading (once at import time)
# ---------------------------------------------------------------------------

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "main_model.pkl")

# This will raise FileNotFoundError at import time if the file is missing.
# Run train_main_model.py first to generate the model file.
_model = joblib.load(_MODEL_PATH)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_BEHAVIOR_TYPES = frozenset([
    "NORMAL",
    "REVENGE_TRADING",
    "OVERTRADING",
    "IMPULSIVE_ENTRY",
    "FATIGUE_TRADING",
    "TILT",
])

# Maps sklearn class label → sub_scores dict key
_KEY_MAP: dict[str, str] = {
    "NORMAL":           "normal_probability",
    "REVENGE_TRADING":  "revenge_trading_probability",
    "OVERTRADING":      "overtrading_probability",
    "IMPULSIVE_ENTRY":  "impulsive_entry_probability",
    "FATIGUE_TRADING":  "fatigue_trading_probability",
    "TILT":             "tilt_probability",
}

# Feature index for trade_count_session (used in OVERTRADING alert)
_TRADE_COUNT_IDX = 1


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_features(features: list) -> None:
    """
    Validate the Feature_Vector before passing it to the model.

    Raises
    ------
    ValueError
        - If len(features) != 20
        - If any element is not int or float
        - If any element is NaN or inf
    """
    if len(features) != 20:
        raise ValueError(
            f"Expected exactly 20 features, got {len(features)}. "
            "The Feature_Vector must contain exactly 20 pre-scaled floats."
        )

    for i, v in enumerate(features):
        if not isinstance(v, (int, float)):
            raise ValueError(
                f"Feature at index {i} is not numeric: "
                f"type={type(v).__name__}, value={v!r}. "
                "All features must be int or float."
            )
        if not math.isfinite(v):
            raise ValueError(
                f"Feature at index {i} contains NaN or inf: value={v}. "
                "All features must be finite numeric values."
            )


# ---------------------------------------------------------------------------
# Alert message generation
# ---------------------------------------------------------------------------

def _build_alert_message(behavior_type: str, features: list) -> str:
    """Return the human-readable alert message for the detected behavior type."""
    if behavior_type == "NORMAL":
        return "Trading behavior looks healthy. Continue as planned."
    elif behavior_type == "REVENGE_TRADING":
        return "Possible emotional trading detected after a loss. Consider stepping back."
    elif behavior_type == "OVERTRADING":
        n = int(features[_TRADE_COUNT_IDX])
        return f"You have placed {n} trades this session. Overtrading detected \u2014 step away."
    elif behavior_type == "IMPULSIVE_ENTRY":
        return "Trade entry detected without proper setup. Slow down and review your plan."
    elif behavior_type == "FATIGUE_TRADING":
        return "Trading fatigue detected. You may be trading during off-hours or after a long session."
    elif behavior_type == "TILT":
        return "Multiple risk signals detected simultaneously. This is a high-risk state. Trade blocked."
    else:
        # Should never reach here given validated model output
        return f"Unknown behavior type detected: {behavior_type}."


# ---------------------------------------------------------------------------
# Intervention level
# ---------------------------------------------------------------------------

def _get_intervention_level(risk_score: int) -> str:
    """
    Derive intervention level from risk_score.

    0–39   → NONE
    40–69  → WARN
    70–100 → BLOCK
    """
    if risk_score <= 39:
        return "NONE"
    elif risk_score <= 69:
        return "WARN"
    else:
        return "BLOCK"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_risk_score(features: list) -> dict:
    """
    Compute a behavioral risk prediction from a 20-float Feature_Vector.

    Parameters
    ----------
    features : list
        Exactly 20 numeric (int or float) values, pre-scaled by Mugeesh's
        pipeline. Must contain no NaN or inf values. Must NOT be re-scaled.

    Returns
    -------
    dict with exactly 6 keys:
        risk_score         : int   in [0, 100]
        behavior_type      : str   one of 6 Behavior_Type labels
        confidence         : float in [0.0, 1.0]
        sub_scores         : dict  6 probability keys summing to ~1.0
        alert_message      : str   non-empty human-readable message
        intervention_level : str   "NONE" | "WARN" | "BLOCK"

    Raises
    ------
    ValueError
        If features has wrong length, contains non-numeric values, or
        contains NaN/inf values.
    """
    # --- Validate input (raises ValueError on any violation) ---
    _validate_features(features)

    # --- Run inference (no re-scaling — pass features directly) ---
    proba = _model.predict_proba([features])[0]   # shape: (n_classes,)
    classes = _model.classes_                      # array of class label strings

    # --- Build sub_scores dict ---
    sub_scores: dict[str, float] = {
        _KEY_MAP[str(cls)]: float(p)
        for cls, p in zip(classes, proba)
    }

    # --- Determine dominant behavior ---
    best_idx = int(np.argmax(proba))
    highest_prob = float(proba[best_idx])
    behavior_type = str(classes[best_idx])

    # --- Derived fields ---
    risk_score = int(highest_prob * 100)
    confidence = highest_prob
    intervention_level = _get_intervention_level(risk_score)
    alert_message = _build_alert_message(behavior_type, features)

    return {
        "risk_score":         risk_score,
        "behavior_type":      behavior_type,
        "confidence":         confidence,
        "sub_scores":         sub_scores,
        "alert_message":      alert_message,
        "intervention_level": intervention_level,
    }
