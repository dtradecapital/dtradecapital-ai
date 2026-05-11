"""
tests/test_inference.py
-----------------------
Property-based tests for get_risk_score() using the Hypothesis library.

All 9 correctness properties (P1–P9) are verified here, plus example-based
smoke tests for specific edge cases.

Run from workspace root:
    python -m pytest rishikesh/tests/test_inference.py -v
"""

import math
import sys
import os

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

# Ensure rishikesh/ is on the path when running from workspace root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import model_inference
from model_inference import get_risk_score

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Valid Feature_Vector: exactly 20 finite floats
valid_feature_vector = st.lists(
    st.floats(
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
        min_value=-1e6,
        max_value=1e6,
    ),
    min_size=20,
    max_size=20,
)

# Invalid length: anything that is NOT 20 elements
invalid_length_vector = st.one_of(
    st.lists(
        st.floats(allow_nan=False, allow_infinity=False),
        min_size=0,
        max_size=19,
    ),
    st.lists(
        st.floats(allow_nan=False, allow_infinity=False),
        min_size=21,
        max_size=40,
    ),
)

VALID_BEHAVIOR_TYPES = {
    "NORMAL",
    "REVENGE_TRADING",
    "OVERTRADING",
    "IMPULSIVE_ENTRY",
    "FATIGUE_TRADING",
    "TILT",
}

REQUIRED_OUTPUT_KEYS = {
    "risk_score",
    "behavior_type",
    "confidence",
    "sub_scores",
    "alert_message",
    "intervention_level",
}

REQUIRED_SUB_SCORE_KEYS = {
    "normal_probability",
    "revenge_trading_probability",
    "overtrading_probability",
    "impulsive_entry_probability",
    "fatigue_trading_probability",
    "tilt_probability",
}

VALID_INTERVENTION_LEVELS = {"NONE", "WARN", "BLOCK"}


# ---------------------------------------------------------------------------
# Property 1: Output has exactly 6 keys
# Feature: behavioral-risk-detection-engine, Property 1: Output has exactly 6 keys
# ---------------------------------------------------------------------------

@given(valid_feature_vector)
@settings(max_examples=100)
def test_output_has_exactly_6_keys(features):
    """P1: For any valid 20-float input, output always has all 6 required keys."""
    result = get_risk_score(features)
    assert set(result.keys()) == REQUIRED_OUTPUT_KEYS, (
        f"Expected keys {REQUIRED_OUTPUT_KEYS}, got {set(result.keys())}"
    )


# ---------------------------------------------------------------------------
# Property 2: risk_score is int in [0, 100]
# Feature: behavioral-risk-detection-engine, Property 2: risk_score is int in [0, 100]
# ---------------------------------------------------------------------------

@given(valid_feature_vector)
@settings(max_examples=100)
def test_risk_score_is_int_in_range(features):
    """P2: risk_score is always a Python int between 0 and 100 inclusive."""
    result = get_risk_score(features)
    assert isinstance(result["risk_score"], int), (
        f"risk_score must be int, got {type(result['risk_score'])}"
    )
    assert 0 <= result["risk_score"] <= 100, (
        f"risk_score {result['risk_score']} is outside [0, 100]"
    )


# ---------------------------------------------------------------------------
# Property 3: confidence is float in [0.0, 1.0]
# Feature: behavioral-risk-detection-engine, Property 3: confidence is float in [0.0, 1.0]
# ---------------------------------------------------------------------------

@given(valid_feature_vector)
@settings(max_examples=100)
def test_confidence_is_float_in_range(features):
    """P3: confidence is always a float between 0.0 and 1.0 inclusive."""
    result = get_risk_score(features)
    assert isinstance(result["confidence"], float), (
        f"confidence must be float, got {type(result['confidence'])}"
    )
    assert 0.0 <= result["confidence"] <= 1.0, (
        f"confidence {result['confidence']} is outside [0.0, 1.0]"
    )


# ---------------------------------------------------------------------------
# Property 4: behavior_type is one of 6 valid strings
# Feature: behavioral-risk-detection-engine, Property 4: behavior_type is one of 6 valid strings
# ---------------------------------------------------------------------------

@given(valid_feature_vector)
@settings(max_examples=100)
def test_behavior_type_is_valid(features):
    """P4: behavior_type is always one of the 6 valid Behavior_Type strings."""
    result = get_risk_score(features)
    assert result["behavior_type"] in VALID_BEHAVIOR_TYPES, (
        f"behavior_type '{result['behavior_type']}' is not in {VALID_BEHAVIOR_TYPES}"
    )


# ---------------------------------------------------------------------------
# Property 5: intervention_level is consistent with risk_score
# Feature: behavioral-risk-detection-engine, Property 5: intervention_level consistent with risk_score
# ---------------------------------------------------------------------------

@given(valid_feature_vector)
@settings(max_examples=100)
def test_intervention_level_consistent(features):
    """P5: intervention_level always matches the risk_score threshold rules."""
    result = get_risk_score(features)
    risk = result["risk_score"]
    level = result["intervention_level"]

    if 0 <= risk <= 39:
        assert level == "NONE", f"risk={risk} should give NONE, got {level}"
    elif 40 <= risk <= 69:
        assert level == "WARN", f"risk={risk} should give WARN, got {level}"
    elif 70 <= risk <= 100:
        assert level == "BLOCK", f"risk={risk} should give BLOCK, got {level}"
    else:
        pytest.fail(f"risk_score {risk} is outside [0, 100]")


# ---------------------------------------------------------------------------
# Property 6: sub_scores has 6 keys summing to 1.0 ±0.01
# Feature: behavioral-risk-detection-engine, Property 6: sub_scores has 6 keys summing to 1.0 ±0.01
# ---------------------------------------------------------------------------

@given(valid_feature_vector)
@settings(max_examples=100)
def test_sub_scores_structure_and_sum(features):
    """P6: sub_scores always has exactly 6 keys and values sum to ~1.0."""
    result = get_risk_score(features)
    sub = result["sub_scores"]

    assert set(sub.keys()) == REQUIRED_SUB_SCORE_KEYS, (
        f"sub_scores keys mismatch: expected {REQUIRED_SUB_SCORE_KEYS}, got {set(sub.keys())}"
    )

    total = sum(sub.values())
    assert abs(total - 1.0) <= 0.01, (
        f"sub_scores values sum to {total:.6f}, expected 1.0 ±0.01"
    )

    for key, val in sub.items():
        assert isinstance(val, float), f"sub_scores['{key}'] must be float, got {type(val)}"
        assert 0.0 <= val <= 1.0, f"sub_scores['{key}'] = {val} is outside [0.0, 1.0]"


# ---------------------------------------------------------------------------
# Property 7: behavior_type matches highest sub_scores key
# Feature: behavioral-risk-detection-engine, Property 7: behavior_type matches argmax of sub_scores
# ---------------------------------------------------------------------------

_BEHAVIOR_TO_SUB_KEY = {
    "NORMAL":           "normal_probability",
    "REVENGE_TRADING":  "revenge_trading_probability",
    "OVERTRADING":      "overtrading_probability",
    "IMPULSIVE_ENTRY":  "impulsive_entry_probability",
    "FATIGUE_TRADING":  "fatigue_trading_probability",
    "TILT":             "tilt_probability",
}


@given(valid_feature_vector)
@settings(max_examples=100)
def test_behavior_type_matches_argmax(features):
    """P7: behavior_type always corresponds to the highest sub_scores value."""
    result = get_risk_score(features)
    sub = result["sub_scores"]
    expected_key = max(sub, key=sub.get)
    actual_sub_key = _BEHAVIOR_TO_SUB_KEY[result["behavior_type"]]
    assert actual_sub_key == expected_key, (
        f"behavior_type '{result['behavior_type']}' maps to sub_scores key "
        f"'{actual_sub_key}', but highest sub_scores key is '{expected_key}' "
        f"(value={sub[expected_key]:.4f})"
    )


# ---------------------------------------------------------------------------
# Property 8: alert_message is a non-empty string
# Feature: behavioral-risk-detection-engine, Property 8: alert_message is non-empty string
# ---------------------------------------------------------------------------

@given(valid_feature_vector)
@settings(max_examples=100)
def test_alert_message_nonempty(features):
    """P8: alert_message is always a non-empty string."""
    result = get_risk_score(features)
    assert isinstance(result["alert_message"], str), (
        f"alert_message must be str, got {type(result['alert_message'])}"
    )
    assert len(result["alert_message"]) > 0, "alert_message must not be empty"


# ---------------------------------------------------------------------------
# Property 9: Invalid-length input raises ValueError
# Feature: behavioral-risk-detection-engine, Property 9: invalid length raises ValueError
# ---------------------------------------------------------------------------

@given(invalid_length_vector)
@settings(max_examples=100)
def test_invalid_length_raises_valueerror(features):
    """P9: Any Feature_Vector with length != 20 raises ValueError."""
    assume(len(features) != 20)
    with pytest.raises(ValueError, match="20"):
        get_risk_score(features)


# ---------------------------------------------------------------------------
# Example-based / smoke tests
# ---------------------------------------------------------------------------

def _make_valid_features(trade_count=5):
    """Return a valid 20-float Feature_Vector with a specific trade count."""
    return [0.3, float(trade_count), 1.0, 1.0, 447.0, 2.5, 1.8,
            0.92, 0.06, 3.0, 4.0, 7.0, 2.0, -70.0, -210.0,
            45.2, 0.042, 1.0, 1.5, 0.78]


def test_nan_input_raises_valueerror():
    """NaN in any position must raise ValueError."""
    features = _make_valid_features()
    features[5] = float("nan")
    with pytest.raises(ValueError, match="NaN or inf"):
        get_risk_score(features)


def test_inf_input_raises_valueerror():
    """Infinity in any position must raise ValueError."""
    features = _make_valid_features()
    features[0] = float("inf")
    with pytest.raises(ValueError, match="NaN or inf"):
        get_risk_score(features)


def test_neg_inf_input_raises_valueerror():
    """-Infinity in any position must raise ValueError."""
    features = _make_valid_features()
    features[3] = float("-inf")
    with pytest.raises(ValueError, match="NaN or inf"):
        get_risk_score(features)


def test_non_numeric_element_raises_valueerror():
    """A string element must raise ValueError."""
    features = _make_valid_features()
    features[2] = "bad"
    with pytest.raises(ValueError, match="not numeric"):
        get_risk_score(features)


def test_none_element_raises_valueerror():
    """None element must raise ValueError."""
    features = _make_valid_features()
    features[0] = None
    with pytest.raises(ValueError, match="not numeric"):
        get_risk_score(features)


def test_overtrading_alert_contains_trade_count():
    """OVERTRADING alert must embed int(features[1]) as {n}."""
    # Force OVERTRADING by using features that strongly signal it
    # (high trade count, high trades_per_hour, long session)
    features = [0.4, 35.0, 0.0, 0.0, 450.0, 1.5, 1.2,
                0.5, 0.05, 12.0, 1.0, 10.0, 2.0, -20.0, -100.0,
                50.0, 0.03, 0.0, 1.0, 0.6]
    result = get_risk_score(features)
    if result["behavior_type"] == "OVERTRADING":
        expected_n = int(features[1])
        assert str(expected_n) in result["alert_message"], (
            f"OVERTRADING alert should contain '{expected_n}', "
            f"got: {result['alert_message']}"
        )


def test_exact_alert_messages_for_fixed_types():
    """Verify exact alert message strings for non-OVERTRADING behavior types."""
    expected_messages = {
        "NORMAL":         "Trading behavior looks healthy. Continue as planned.",
        "REVENGE_TRADING": "Possible emotional trading detected after a loss. Consider stepping back.",
        "IMPULSIVE_ENTRY": "Trade entry detected without proper setup. Slow down and review your plan.",
        "FATIGUE_TRADING": "Trading fatigue detected. You may be trading during off-hours or after a long session.",
        "TILT":           "Multiple risk signals detected simultaneously. This is a high-risk state. Trade blocked.",
    }
    features = _make_valid_features()
    result = get_risk_score(features)
    bt = result["behavior_type"]
    if bt in expected_messages:
        assert result["alert_message"] == expected_messages[bt], (
            f"Alert message mismatch for {bt}:\n"
            f"  expected: {expected_messages[bt]!r}\n"
            f"  got:      {result['alert_message']!r}"
        )


def test_model_loaded_at_import():
    """The module-level _model must be loaded (not None) at import time."""
    assert model_inference._model is not None, (
        "model_inference._model should be loaded at import time"
    )


def test_model_identity_stable_across_calls():
    """The same model object must be reused across multiple calls."""
    model_id_before = id(model_inference._model)
    get_risk_score(_make_valid_features())
    get_risk_score(_make_valid_features(trade_count=10))
    model_id_after = id(model_inference._model)
    assert model_id_before == model_id_after, (
        "Model object identity changed between calls — model is being reloaded per call"
    )


def test_full_output_example():
    """Smoke test: the example from the spec returns a valid 6-field dict."""
    features = [0.3, 14, 1, 1, 447, 2.5, 1.8, 0.92, 0.06, 3,
                4, 7, 2, -70.0, -210.0, 45.2, 0.042, 1, 1.5, 0.78]
    result = get_risk_score(features)
    assert set(result.keys()) == REQUIRED_OUTPUT_KEYS
    assert isinstance(result["risk_score"], int)
    assert 0 <= result["risk_score"] <= 100
    assert result["behavior_type"] in VALID_BEHAVIOR_TYPES
    assert result["intervention_level"] in VALID_INTERVENTION_LEVELS
