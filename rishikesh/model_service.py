"""
model_service.py
----------------
ModelService wraps the Inference_Engine (get_risk_score) with:
  - In-memory prediction caching keyed by "{user_id}:{trade_id}"
  - Structured append-only logging to predictions.log
  - Per-prediction latency measurement
  - WARNING log when latency exceeds 100 ms

Usage
-----
    from model_service import ModelService

    svc = ModelService()
    result = svc.predict("U001", "T001", [0.3, 14, 1, 1, 447, ...])
    print(result)
"""

import logging
import os
import time
from datetime import datetime, timezone

from model_inference import get_risk_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_MODULE_DIR = os.path.dirname(__file__)
_LOG_PATH = os.path.join(_MODULE_DIR, "predictions.log")


# ---------------------------------------------------------------------------
# ModelService
# ---------------------------------------------------------------------------

class ModelService:
    """
    Thread-unsafe (single-process) service layer for behavioral risk predictions.

    Attributes
    ----------
    _cache : dict[str, dict]
        In-memory cache keyed by "{user_id}:{trade_id}".
        Instance-scoped — a new ModelService() starts with an empty cache.
    """

    def __init__(self, log_path: str = None) -> None:
        self._cache: dict[str, dict] = {}
        self._log_path = log_path if log_path is not None else _LOG_PATH
        self._setup_logging()

    # ------------------------------------------------------------------
    # Logging setup
    # ------------------------------------------------------------------

    def _setup_logging(self) -> None:
        """
        Configure Python logging to write structured prediction lines to
        predictions.log in append mode. Creates the file (and any missing
        parent directories) automatically.
        """
        log_dir = os.path.dirname(self._log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # Root logger for WARNING-level messages (slow predictions etc.)
        logging.basicConfig(
            level=logging.WARNING,
            format="%(levelname)s %(asctime)s %(message)s",
        )

        # Dedicated file handler for structured prediction log lines
        self._file_handler = logging.FileHandler(self._log_path, mode="a", encoding="utf-8")
        self._file_handler.setLevel(logging.INFO)
        self._file_handler.setFormatter(logging.Formatter("%(message)s"))

        self._pred_logger = logging.getLogger(f"predictions.{id(self)}")
        self._pred_logger.setLevel(logging.INFO)
        self._pred_logger.addHandler(self._file_handler)
        # Prevent propagation to root logger (avoids duplicate console output)
        self._pred_logger.propagate = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, user_id: str, trade_id: str, features: list) -> dict:
        """
        Return a Risk_Prediction dict for the given trade.

        Cache hit  → returns stored result immediately; no inference, no log write.
        Cache miss → calls get_risk_score, stores result, appends one log line.

        Parameters
        ----------
        user_id  : str  Unique identifier for the trader.
        trade_id : str  Unique identifier for the trade event.
        features : list Exactly 20 pre-scaled floats (must NOT be re-scaled).

        Returns
        -------
        dict  Risk_Prediction with 6 fields: risk_score, behavior_type,
              confidence, sub_scores, alert_message, intervention_level.

        Raises
        ------
        ValueError  Propagated from get_risk_score on invalid features.
        """
        cache_key = f"{user_id}:{trade_id}"

        # --- Cache hit: return immediately without any I/O ---
        if cache_key in self._cache:
            return self._cache[cache_key]

        # --- Cache miss: run inference and measure latency ---
        t0 = time.perf_counter()
        result = get_risk_score(features)
        ms_taken = (time.perf_counter() - t0) * 1000.0

        # --- Store in cache ---
        self._cache[cache_key] = result

        # --- Append structured log line ---
        timestamp = datetime.now(timezone.utc).isoformat()
        log_line = (
            f"[{timestamp}] "
            f"trade_id={trade_id} "
            f"behavior={result['behavior_type']} "
            f"risk={result['risk_score']} "
            f"ms={ms_taken:.2f}"
        )
        self._pred_logger.info(log_line)

        # --- Warn on slow predictions ---
        if ms_taken > 100:
            logging.warning(
                f"Slow prediction: trade_id={trade_id} latency={ms_taken:.2f}ms "
                f"(threshold=100ms)"
            )

        return result

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Close the log file handler and release the file lock.
        Call this when the ModelService instance is no longer needed,
        especially in tests on Windows where open file handles prevent
        temporary directory cleanup.
        """
        if self._file_handler:
            self._pred_logger.removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_handler = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
