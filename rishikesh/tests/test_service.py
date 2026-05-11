"""
tests/test_service.py
---------------------
Property-based tests for ModelService using the Hypothesis library.

Covers correctness properties P10–P13:
  P10: Same user_id:trade_id always returns identical cached result
  P11: Different trade_ids are independently cached
  P12: New prediction appends exactly one log line
  P13: Cached prediction does not append to log

Run from workspace root:
    python -m pytest rishikesh/tests/test_service.py -v
"""

import logging
import os
import sys
import tempfile
import time
from unittest.mock import patch

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

# Ensure rishikesh/ is on the path when running from workspace root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_service import ModelService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_OUTPUT_KEYS = {
    "risk_score",
    "behavior_type",
    "confidence",
    "sub_scores",
    "alert_message",
    "intervention_level",
}

VALID_FEATURES = [
    0.3, 14.0, 1.0, 1.0, 447.0, 2.5, 1.8, 0.92, 0.06, 3.0,
    4.0, 7.0, 2.0, -70.0, -210.0, 45.2, 0.042, 1.0, 1.5, 0.78,
]


def _make_service(tmp_dir: str):
    """
    Create a ModelService with a log file isolated to tmp_dir.
    Returns (svc, log_path). Use as a context manager:
        with _make_service(tmp_dir) as (svc, log_path): ...
    """
    log_path = os.path.join(tmp_dir, "predictions.log")
    svc = ModelService(log_path=log_path)
    return svc, log_path


def _count_log_lines(log_path: str) -> int:
    """Return the number of non-empty lines in the log file."""
    if not os.path.exists(log_path):
        return 0
    with open(log_path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Restrict to alphanumeric characters to avoid hypothesis generating
# strings with characters that cause issues in cache keys or log lines
user_id_strategy = st.text(
    min_size=1, max_size=20,
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"))
)
trade_id_strategy = st.text(
    min_size=1, max_size=20,
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"))
)


# ---------------------------------------------------------------------------
# Property 10: Same user_id:trade_id returns identical cached result
# Feature: behavioral-risk-detection-engine, Property 10: cache correctness
# ---------------------------------------------------------------------------

@given(user_id_strategy, trade_id_strategy)
@settings(max_examples=50)
def test_same_pair_returns_identical_result(user_id, trade_id):
    """P10: Calling predict twice with the same pair returns identical dicts."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        svc, _ = _make_service(tmp_dir)
        try:
            result1 = svc.predict(user_id, trade_id, VALID_FEATURES)
            result2 = svc.predict(user_id, trade_id, VALID_FEATURES)
            assert result1 == result2, (
                f"Cache miss on second call for user_id={user_id!r}, trade_id={trade_id!r}. "
                f"Results differ:\n  first:  {result1}\n  second: {result2}"
            )
        finally:
            svc.close()


# ---------------------------------------------------------------------------
# Property 11: Different trade_ids are independently cached
# Feature: behavioral-risk-detection-engine, Property 11: no cross-contamination
# ---------------------------------------------------------------------------

@given(user_id_strategy, trade_id_strategy, trade_id_strategy)
@settings(max_examples=50)
def test_different_trade_ids_are_independent(user_id, trade_id_1, trade_id_2):
    """P11: Different trade_ids for the same user are stored independently."""
    assume(trade_id_1 != trade_id_2)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        svc, _ = _make_service(tmp_dir)
        try:
            result1 = svc.predict(user_id, trade_id_1, VALID_FEATURES)
            result2 = svc.predict(user_id, trade_id_2, VALID_FEATURES)

            key1 = f"{user_id}:{trade_id_1}"
            key2 = f"{user_id}:{trade_id_2}"
            assert key1 in svc._cache, f"Cache entry missing for key '{key1}'"
            assert key2 in svc._cache, f"Cache entry missing for key '{key2}'"
            assert svc._cache[key1] == result1
            assert svc._cache[key2] == result2
        finally:
            svc.close()


# ---------------------------------------------------------------------------
# Property 12: New prediction appends exactly one log line
# Feature: behavioral-risk-detection-engine, Property 12: new prediction logs one line
# ---------------------------------------------------------------------------

@given(user_id_strategy, trade_id_strategy)
@settings(max_examples=50)
def test_new_prediction_appends_log_line(user_id, trade_id):
    """P12: A new (uncached) prediction appends exactly one line to predictions.log."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        svc, log_path = _make_service(tmp_dir)
        try:
            lines_before = _count_log_lines(log_path)
            svc.predict(user_id, trade_id, VALID_FEATURES)
            svc._file_handler.flush()
            lines_after = _count_log_lines(log_path)
            assert lines_after == lines_before + 1, (
                f"Expected exactly 1 new log line after new prediction, "
                f"got {lines_after - lines_before}"
            )
        finally:
            svc.close()


# ---------------------------------------------------------------------------
# Property 13: Cached prediction does not append to log
# Feature: behavioral-risk-detection-engine, Property 13: cached prediction no log write
# ---------------------------------------------------------------------------

@given(user_id_strategy, trade_id_strategy)
@settings(max_examples=50)
def test_cached_prediction_no_log_write(user_id, trade_id):
    """P13: A cached prediction must NOT append any new line to predictions.log."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        svc, log_path = _make_service(tmp_dir)
        try:
            # First call — populates cache and writes one log line
            svc.predict(user_id, trade_id, VALID_FEATURES)
            svc._file_handler.flush()
            lines_after_first = _count_log_lines(log_path)

            # Second call — must be a cache hit, no new log line
            svc.predict(user_id, trade_id, VALID_FEATURES)
            svc._file_handler.flush()
            lines_after_second = _count_log_lines(log_path)

            assert lines_after_second == lines_after_first, (
                f"Cached prediction wrote {lines_after_second - lines_after_first} "
                f"extra log line(s) — expected 0"
            )
        finally:
            svc.close()


# ---------------------------------------------------------------------------
# Example-based / smoke tests
# ---------------------------------------------------------------------------

def test_predict_returns_all_6_keys():
    """predict() must return a dict with all 6 required Risk_Prediction keys."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        svc, _ = _make_service(tmp_dir)
        try:
            result = svc.predict("U001", "T001", VALID_FEATURES)
            assert set(result.keys()) == REQUIRED_OUTPUT_KEYS
        finally:
            svc.close()


def test_new_instance_has_empty_cache():
    """Two separate ModelService instances must have independent empty caches."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        svc1, _ = _make_service(tmp_dir)
        svc2, _ = _make_service(tmp_dir)
        try:
            svc1.predict("U001", "T001", VALID_FEATURES)
            # svc2 must not see svc1's cache entry
            assert "U001:T001" not in svc2._cache, (
                "svc2 should not share cache with svc1"
            )
        finally:
            svc1.close()
            svc2.close()


def test_log_file_created_if_missing():
    """predictions.log must be auto-created if it does not exist."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        log_path = os.path.join(tmp_dir, "predictions.log")
        assert not os.path.exists(log_path), "Log file should not exist yet"

        svc = ModelService(log_path=log_path)
        try:
            svc.predict("U001", "T001", VALID_FEATURES)
            assert os.path.exists(log_path), (
                "predictions.log should have been auto-created after first prediction"
            )
        finally:
            svc.close()


def test_log_line_format():
    """Log line must match the expected format."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        svc, log_path = _make_service(tmp_dir)
        try:
            svc.predict("U001", "T001", VALID_FEATURES)
            svc._file_handler.flush()
        finally:
            svc.close()

        with open(log_path, "r", encoding="utf-8") as f:
            line = f.readline().strip()

        assert line.startswith("["), f"Log line should start with '[', got: {line!r}"
        assert "trade_id=T001" in line, f"Log line missing trade_id: {line!r}"
        assert "behavior=" in line, f"Log line missing behavior: {line!r}"
        assert "risk=" in line, f"Log line missing risk: {line!r}"
        assert "ms=" in line, f"Log line missing ms: {line!r}"


def test_slow_prediction_emits_warning():
    """When latency > 100ms, a WARNING must be logged."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        svc, _ = _make_service(tmp_dir)
        try:
            call_count = [0]

            def mock_perf_counter():
                call_count[0] += 1
                if call_count[0] == 1:
                    return 0.0      # t0
                return 0.200        # t1 — 200ms later

            with patch("model_service.time.perf_counter", side_effect=mock_perf_counter):
                with patch("model_service.logging.warning") as mock_warn:
                    svc.predict("U001", "T_SLOW", VALID_FEATURES)
                    assert mock_warn.called, (
                        "logging.warning should have been called for a >100ms prediction"
                    )
                    warn_msg = mock_warn.call_args[0][0]
                    assert "T_SLOW" in warn_msg, (
                        f"Warning should mention trade_id, got: {warn_msg!r}"
                    )
        finally:
            svc.close()


def test_different_user_ids_same_trade_id_are_independent():
    """Different user_ids with the same trade_id must be cached independently."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        svc, _ = _make_service(tmp_dir)
        try:
            svc.predict("U001", "T001", VALID_FEATURES)
            svc.predict("U002", "T001", VALID_FEATURES)

            assert "U001:T001" in svc._cache
            assert "U002:T001" in svc._cache
        finally:
            svc.close()
