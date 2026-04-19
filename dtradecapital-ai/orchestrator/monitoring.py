"""
orchestrator/monitoring.py
==========================
Team Lead: Lokesh
Monitors pipeline latency, error rates, and per-step timing.

Collected metrics are printed as structured logs so they can be picked up
by any log-aggregation tool (Datadog, CloudWatch, Grafana Loki, etc.)
without extra dependencies.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict

logger = logging.getLogger("orchestrator.monitoring")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StepMetrics:
    call_count:   int   = 0
    error_count:  int   = 0
    total_ms:     float = 0.0
    min_ms:       float = float("inf")
    max_ms:       float = 0.0

    def record(self, elapsed_seconds: float) -> None:
        ms = elapsed_seconds * 1000
        self.call_count  += 1
        self.total_ms    += ms
        self.min_ms       = min(self.min_ms, ms)
        self.max_ms       = max(self.max_ms, ms)

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.call_count if self.call_count else 0.0

    @property
    def error_rate(self) -> float:
        total = self.call_count + self.error_count
        return self.error_count / total if total else 0.0


@dataclass
class PipelineMetrics:
    success_count:  int   = 0
    failure_count:  int   = 0
    total_ms:       float = 0.0
    min_ms:         float = float("inf")
    max_ms:         float = 0.0
    sla_breach_count: int = 0             # runs that exceeded 500ms
    sla_target_ms:    float = 500.0

    failures_by_code: DefaultDict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def record_success(self, elapsed_ms: float) -> None:
        self.success_count += 1
        self.total_ms      += elapsed_ms
        self.min_ms         = min(self.min_ms, elapsed_ms)
        self.max_ms         = max(self.max_ms, elapsed_ms)
        if elapsed_ms > self.sla_target_ms:
            self.sla_breach_count += 1

    def record_failure(self, code: str) -> None:
        self.failure_count += 1
        self.failures_by_code[code] += 1

    @property
    def total_runs(self) -> int:
        return self.success_count + self.failure_count

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.success_count if self.success_count else 0.0

    @property
    def success_rate(self) -> float:
        return self.success_count / self.total_runs if self.total_runs else 0.0

    @property
    def sla_pass_rate(self) -> float:
        if not self.success_count:
            return 0.0
        return (self.success_count - self.sla_breach_count) / self.success_count


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class OrchestratorMonitor:
    """
    Lightweight in-process metrics collector for the orchestrator pipeline.

    Works without any external dependencies.
    Call monitor.summary() at any time to get a snapshot.
    """

    PIPELINE_STEPS = [
        "validate_clean",
        "store_trade",
        "extract_features",
        "model_inference",
        "store_score",
    ]

    def __init__(self):
        self._pipeline  = PipelineMetrics()
        self._steps: DefaultDict[str, StepMetrics] = defaultdict(StepMetrics)
        self._start_time = time.time()
        logger.info("OrchestratorMonitor started.")

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_success(self, elapsed_ms: float) -> None:
        self._pipeline.record_success(elapsed_ms)
        logger.info(
            "METRIC pipeline_success latency_ms=%.1f sla=%s",
            elapsed_ms,
            "PASS" if elapsed_ms <= self._pipeline.sla_target_ms else "BREACH",
        )

    def record_failure(self, code: str) -> None:
        self._pipeline.record_failure(code)
        logger.warning("METRIC pipeline_failure code=%s", code)

    def record_step(self, step_name: str, elapsed_seconds: float) -> None:
        ms = elapsed_seconds * 1000
        self._steps[step_name].record(elapsed_seconds)
        logger.debug("METRIC step=%s latency_ms=%.1f", step_name, ms)

    def record_step_error(self, step_name: str) -> None:
        self._steps[step_name].error_count += 1
        logger.warning("METRIC step_error step=%s", step_name)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """
        Return a full metrics snapshot as a plain dict.
        Suitable for a health-check endpoint or a periodic log flush.
        """
        uptime_seconds = time.time() - self._start_time

        pipeline = self._pipeline
        step_data = {}
        for step_name, m in self._steps.items():
            step_data[step_name] = {
                "calls":      m.call_count,
                "errors":     m.error_count,
                "avg_ms":     round(m.avg_ms,   2),
                "min_ms":     round(m.min_ms,   2) if m.call_count else None,
                "max_ms":     round(m.max_ms,   2),
                "error_rate": round(m.error_rate, 4),
            }

        return {
            "uptime_seconds":  round(uptime_seconds, 1),
            "pipeline": {
                "total_runs":       pipeline.total_runs,
                "success_count":    pipeline.success_count,
                "failure_count":    pipeline.failure_count,
                "success_rate":     round(pipeline.success_rate, 4),
                "avg_latency_ms":   round(pipeline.avg_ms,  2),
                "min_latency_ms":   round(pipeline.min_ms,  2) if pipeline.success_count else None,
                "max_latency_ms":   round(pipeline.max_ms,  2),
                "sla_target_ms":    pipeline.sla_target_ms,
                "sla_pass_rate":    round(pipeline.sla_pass_rate, 4),
                "sla_breaches":     pipeline.sla_breach_count,
                "failures_by_code": dict(pipeline.failures_by_code),
            },
            "steps": step_data,
        }

    def log_summary(self) -> None:
        """Write the full summary snapshot to the logger (INFO level)."""
        import json
        snapshot = self.summary()
        logger.info("MONITOR SUMMARY\n%s", json.dumps(snapshot, indent=2))

    def health_check(self) -> dict:
        """
        Simple health-check used by Varun's GET /health endpoint.
        Returns status = 'healthy' / 'degraded' / 'unhealthy'.
        """
        p = self._pipeline

        if p.total_runs == 0:
            status = "healthy"          # no runs yet, nothing broken
        elif p.success_rate < 0.5:
            status = "unhealthy"
        elif p.success_rate < 0.9 or p.sla_pass_rate < 0.8:
            status = "degraded"
        else:
            status = "healthy"

        return {
            "status":          status,
            "total_runs":      p.total_runs,
            "success_rate":    round(p.success_rate, 4),
            "sla_pass_rate":   round(p.sla_pass_rate, 4),
            "avg_latency_ms":  round(p.avg_ms, 2),
        }


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json, random

    mon = OrchestratorMonitor()

    # Simulate 10 pipeline runs
    for i in range(10):
        ms = random.uniform(80, 600)
        if random.random() > 0.85:
            mon.record_failure(random.choice(["VALIDATION_ERROR", "MODEL_ERROR"]))
        else:
            mon.record_success(ms)
            for step in OrchestratorMonitor.PIPELINE_STEPS:
                mon.record_step(step, random.uniform(0.01, 0.12))

    print(json.dumps(mon.summary(), indent=2))
    print("\nHealth check:", mon.health_check())
