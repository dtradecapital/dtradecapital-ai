"""
orchestrator/main.py
====================
Team Lead: Lokesh
Day 3-4 Task: Build Integration Orchestrator

Connects the full pipeline:
  Balaji's data_pipeline  →  Mugeesh's feature_engine
       →  Rishikesh's ai_models  →  Varun's API layer

Integration Flow (from spec):
  1. Receive raw trade JSON
  2. validate_trade()   — Balaji
  3. clean_trade()      — Balaji
  4. store_trade()      — Balaji
  5. extract_behavioral_features(trade_history)  — Mugeesh
  6. get_risk_score(feature_vector)              — Rishikesh
  7. Store result in behavioral_scores table     — Varun / API layer
  8. Return final prediction to caller

Total time target: <500ms end-to-end
"""

import time
import logging
import traceback
from typing import Any

from monitoring import OrchestratorMonitor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("orchestrator.main")


# ---------------------------------------------------------------------------
# Lazy imports — each teammate's module lives in its own package.
# We import lazily so the orchestrator can be unit-tested even when a
# downstream service is not yet available.
# ---------------------------------------------------------------------------

def _import_data_pipeline():
    """Balaji's data pipeline."""
    try:
        from data_pipeline.data_pipeline import validate_trade, clean_trade, store_trade
        return validate_trade, clean_trade, store_trade
    except ImportError as exc:
        raise ImportError(
            "data_pipeline module not found. "
            "Make sure Balaji's data_pipeline.py is in the data_pipeline/ package."
        ) from exc


def _import_feature_engine():
    """Mugeesh's feature engine."""
    try:
        from feature_engine.feature_engine import extract_behavioral_features
        return extract_behavioral_features
    except ImportError as exc:
        raise ImportError(
            "feature_engine module not found. "
            "Make sure Mugeesh's feature_engine.py is in the feature_engine/ package."
        ) from exc


def _import_model_service():
    """Rishikesh's AI model service."""
    try:
        from ai_models.model_service import get_risk_score
        return get_risk_score
    except ImportError as exc:
        raise ImportError(
            "ai_models module not found. "
            "Make sure Rishikesh's model_service.py is in the ai_models/ package."
        ) from exc


def _import_api_layer():
    """Varun's API storage helper (store result in behavioral_scores table)."""
    try:
        from api.endpoints import store_behavioral_score
        return store_behavioral_score
    except ImportError as exc:
        raise ImportError(
            "api module not found. "
            "Make sure Varun's endpoints.py is in the api/ package."
        ) from exc


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    Central integration layer.

    Usage
    -----
    orch = Orchestrator()
    result = orch.run(trade_data, trade_history)
    """

    def __init__(self):
        self.monitor = OrchestratorMonitor()
        logger.info("Orchestrator initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, trade_data: dict, trade_history: list[dict]) -> dict:
        """
        Run the full pipeline for a single incoming trade.

        Parameters
        ----------
        trade_data     : dict  — raw trade JSON from frontend/API
        trade_history  : list  — previous trades for this user
                                 (needed by Mugeesh's feature engine)

        Returns
        -------
        dict with keys:
            user_id, trade_id, risk_score, behavior_type, confidence,
            sub_scores, alert_message, intervention_level,
            pipeline_latency_ms, status
        """
        pipeline_start = time.perf_counter()
        user_id  = trade_data.get("user_id",  "UNKNOWN")
        trade_id = trade_data.get("trade_id", "UNKNOWN")

        logger.info("Pipeline START | user=%s trade=%s", user_id, trade_id)

        try:
            # ── STEP 1 ── Validate (Balaji) ────────────────────────────
            cleaned_trade = self._step_validate_and_clean(trade_data)

            # ── STEP 2 ── Store raw trade (Balaji) ─────────────────────
            self._step_store_trade(cleaned_trade)

            # ── STEP 3 ── Extract features (Mugeesh) ───────────────────
            feature_vector = self._step_extract_features(cleaned_trade, trade_history)

            # ── STEP 4 ── Model inference (Rishikesh) ───────────────────
            prediction = self._step_model_inference(feature_vector, user_id, trade_id)

            # ── STEP 5 ── Persist result (Varun) ────────────────────────
            self._step_store_score(prediction)

            # ── Done ────────────────────────────────────────────────────
            elapsed_ms = (time.perf_counter() - pipeline_start) * 1000
            prediction["pipeline_latency_ms"] = round(elapsed_ms, 2)
            prediction["status"] = "success"

            self.monitor.record_success(elapsed_ms)
            logger.info(
                "Pipeline DONE  | user=%s trade=%s | risk=%s behaviour=%s | %.1fms",
                user_id, trade_id,
                prediction.get("risk_score"),
                prediction.get("behavior_type"),
                elapsed_ms,
            )

            if elapsed_ms > 500:
                logger.warning(
                    "SLA BREACH: pipeline took %.1fms (target <500ms) for trade=%s",
                    elapsed_ms, trade_id,
                )

            return prediction

        except PipelineValidationError as exc:
            return self._error_response(user_id, trade_id, exc, "VALIDATION_ERROR", pipeline_start)

        except PipelineFeatureError as exc:
            return self._error_response(user_id, trade_id, exc, "FEATURE_ERROR", pipeline_start)

        except PipelineModelError as exc:
            return self._error_response(user_id, trade_id, exc, "MODEL_ERROR", pipeline_start)

        except PipelineStorageError as exc:
            return self._error_response(user_id, trade_id, exc, "STORAGE_ERROR", pipeline_start)

        except Exception as exc:  # noqa: BLE001
            return self._error_response(user_id, trade_id, exc, "UNEXPECTED_ERROR", pipeline_start)

    # ------------------------------------------------------------------
    # Pipeline Steps
    # ------------------------------------------------------------------

    def _step_validate_and_clean(self, trade_data: dict) -> dict:
        """
        Step 1 + 2 — Balaji's validate_trade() and clean_trade().
        """
        step_start = time.perf_counter()
        logger.info("Step 1/5 | validate + clean trade")

        try:
            validate_trade, clean_trade, _ = _import_data_pipeline()

            # -- Validate --
            validation_result = validate_trade(trade_data)
            if not validation_result.get("valid", False):
                errors = validation_result.get("errors", [])
                raise PipelineValidationError(
                    f"Trade failed validation: {errors}",
                    field=validation_result.get("field"),
                    errors=errors,
                )

            # -- Clean --
            cleaned = clean_trade(trade_data)
            if not cleaned:
                raise PipelineValidationError("clean_trade() returned empty result.")

            self.monitor.record_step("validate_clean", time.perf_counter() - step_start)
            logger.debug("Step 1/5 DONE | %.1fms", (time.perf_counter() - step_start) * 1000)
            return cleaned

        except PipelineValidationError:
            raise
        except Exception as exc:
            raise PipelineValidationError(f"Unexpected error in data pipeline: {exc}") from exc

    def _step_store_trade(self, cleaned_trade: dict) -> None:
        """
        Step 2 — Balaji's store_trade().
        """
        step_start = time.perf_counter()
        logger.info("Step 2/5 | store trade in DB")

        try:
            _, _, store_trade = _import_data_pipeline()
            store_trade(cleaned_trade)
            self.monitor.record_step("store_trade", time.perf_counter() - step_start)
            logger.debug("Step 2/5 DONE | %.1fms", (time.perf_counter() - step_start) * 1000)

        except Exception as exc:
            raise PipelineStorageError(f"store_trade() failed: {exc}") from exc

    def _step_extract_features(self, cleaned_trade: dict, trade_history: list[dict]) -> list[float]:
        """
        Step 3 — Mugeesh's extract_behavioral_features().
        Returns a 20-element feature vector.
        """
        step_start = time.perf_counter()
        logger.info("Step 3/5 | extract behavioural features")

        try:
            extract_behavioral_features = _import_feature_engine()

            # The feature engine needs the current trade appended to history
            full_history = trade_history + [cleaned_trade]
            result = extract_behavioral_features(full_history)

            feature_vector = result.get("features") if isinstance(result, dict) else result

            if not feature_vector or len(feature_vector) != 20:
                raise PipelineFeatureError(
                    f"Expected 20 features, got {len(feature_vector) if feature_vector else 0}. "
                    "Check Mugeesh's extract_behavioral_features() output."
                )

            self.monitor.record_step("extract_features", time.perf_counter() - step_start)
            logger.debug(
                "Step 3/5 DONE | 20 features extracted | %.1fms",
                (time.perf_counter() - step_start) * 1000,
            )
            return feature_vector

        except PipelineFeatureError:
            raise
        except Exception as exc:
            raise PipelineFeatureError(f"Feature extraction failed: {exc}") from exc

    def _step_model_inference(
        self, feature_vector: list[float], user_id: str, trade_id: str
    ) -> dict:
        """
        Step 4 — Rishikesh's get_risk_score().
        Returns the full prediction dict.
        """
        step_start = time.perf_counter()
        logger.info("Step 4/5 | model inference")

        try:
            get_risk_score = _import_model_service()
            prediction = get_risk_score(feature_vector)

            # Inject IDs if model doesn't include them
            prediction.setdefault("user_id",  user_id)
            prediction.setdefault("trade_id", trade_id)

            self._validate_prediction(prediction)

            self.monitor.record_step("model_inference", time.perf_counter() - step_start)
            logger.debug("Step 4/5 DONE | %.1fms", (time.perf_counter() - step_start) * 1000)
            return prediction

        except PipelineModelError:
            raise
        except Exception as exc:
            raise PipelineModelError(f"Model inference failed: {exc}") from exc

    def _step_store_score(self, prediction: dict) -> None:
        """
        Step 5 — Varun's store_behavioral_score() — persists result in
        behavioral_scores table.
        """
        step_start = time.perf_counter()
        logger.info("Step 5/5 | store behavioural score")

        try:
            store_behavioral_score = _import_api_layer()
            store_behavioral_score(prediction)
            self.monitor.record_step("store_score", time.perf_counter() - step_start)
            logger.debug("Step 5/5 DONE | %.1fms", (time.perf_counter() - step_start) * 1000)

        except Exception as exc:
            # Storage failure is non-fatal — log but don't block the response
            logger.error("store_behavioral_score() failed (non-fatal): %s", exc)
            self.monitor.record_step_error("store_score")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    VALID_BEHAVIOR_TYPES = {
        "NORMAL", "REVENGE_TRADING", "OVERTRADING",
        "IMPULSIVE_ENTRY", "FATIGUE_TRADING", "TILT",
    }
    VALID_INTERVENTION_LEVELS = {"NONE", "WARN", "BLOCK"}

    def _validate_prediction(self, prediction: dict) -> None:
        """Sanity-check the model output before returning it downstream."""
        risk_score = prediction.get("risk_score")
        if risk_score is None or not (0 <= risk_score <= 100):
            raise PipelineModelError(
                f"Invalid risk_score '{risk_score}'. Must be 0-100."
            )

        behavior_type = prediction.get("behavior_type", "").upper()
        if behavior_type not in self.VALID_BEHAVIOR_TYPES:
            raise PipelineModelError(
                f"Invalid behavior_type '{behavior_type}'. "
                f"Must be one of {self.VALID_BEHAVIOR_TYPES}."
            )

        intervention = prediction.get("intervention_level", "").upper()
        if intervention not in self.VALID_INTERVENTION_LEVELS:
            raise PipelineModelError(
                f"Invalid intervention_level '{intervention}'. "
                f"Must be one of {self.VALID_INTERVENTION_LEVELS}."
            )

    def _error_response(
        self,
        user_id: str,
        trade_id: str,
        exc: Exception,
        code: str,
        pipeline_start: float,
    ) -> dict:
        elapsed_ms = (time.perf_counter() - pipeline_start) * 1000
        self.monitor.record_failure(code)
        logger.error(
            "Pipeline FAILED | user=%s trade=%s | %s: %s\n%s",
            user_id, trade_id, code, exc, traceback.format_exc(),
        )
        return {
            "status":               "error",
            "error":                True,
            "code":                 code,
            "message":              str(exc),
            "user_id":              user_id,
            "trade_id":             trade_id,
            "pipeline_latency_ms":  round(elapsed_ms, 2),
        }


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class PipelineError(Exception):
    """Base class for all orchestrator pipeline errors."""


class PipelineValidationError(PipelineError):
    def __init__(self, message: str, field: str | None = None, errors: list | None = None):
        super().__init__(message)
        self.field  = field
        self.errors = errors or []


class PipelineFeatureError(PipelineError):
    """Raised when Mugeesh's feature engine fails."""


class PipelineModelError(PipelineError):
    """Raised when Rishikesh's model service fails."""


class PipelineStorageError(PipelineError):
    """Raised when Balaji's or Varun's storage layer fails."""


# ---------------------------------------------------------------------------
# Module-level convenience function (used by Varun's API endpoints)
# ---------------------------------------------------------------------------

_default_orchestrator: Orchestrator | None = None


def run_pipeline(trade_data: dict, trade_history: list[dict]) -> dict:
    """
    Convenience function — creates a shared Orchestrator instance on first
    call and reuses it for every subsequent call (singleton pattern).

    Called by Varun's /analyze-trade endpoint:
        from orchestrator.main import run_pipeline
        result = run_pipeline(trade_data, trade_history)
    """
    global _default_orchestrator  # noqa: PLW0603
    if _default_orchestrator is None:
        _default_orchestrator = Orchestrator()
    return _default_orchestrator.run(trade_data, trade_history)


# ---------------------------------------------------------------------------
# Quick smoke-test (run: python main.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    SAMPLE_TRADE: dict[str, Any] = {
        "trade_id":               "TRD-001",
        "user_id":                "USR-123",
        "symbol":                 "EURUSD",
        "trade_type":             "BUY",
        "open_time":              "2025-11-01 14:23:45",
        "close_time":             "2025-11-01 14:31:12",
        "open_price":             1.08450,
        "close_price":            1.08310,
        "lot_size":               0.50,
        "stop_loss":              1.08200,
        "take_profit":            1.08700,
        "profit_loss":           -70.00,
        "duration_seconds":       447,
        "broker":                 "Pepperstone",
        "platform":               "MT5",
        "session":                "LONDON",
        "account_balance_before": 5000.00,
        "account_balance_after":  4930.00,
    }

    print("Running orchestrator smoke-test …")
    orch   = Orchestrator()
    result = orch.run(SAMPLE_TRADE, trade_history=[])
    print(json.dumps(result, indent=2))
