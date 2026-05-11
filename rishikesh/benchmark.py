"""
benchmark.py
------------
Latency benchmark for the Behavioral Risk Detection Engine.

Runs 1000 predictions using get_risk_score() directly (bypassing ModelService
cache) and reports latency statistics. Asserts that mean latency < 50ms.

Usage (from workspace root):
    python rishikesh/benchmark.py
"""

import random
import time

from model_inference import get_risk_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_PREDICTIONS = 1000
N_FEATURES = 20
MEAN_LATENCY_THRESHOLD_MS = 50.0


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(n: int = N_PREDICTIONS) -> list[float]:
    """
    Run n predictions with random valid Feature_Vectors.
    Returns a sorted list of per-call latencies in milliseconds.
    """
    # Pre-generate all vectors so generation time is not included in latency
    vectors = [
        [random.uniform(0.0, 1.0) for _ in range(N_FEATURES)]
        for _ in range(n)
    ]

    latencies: list[float] = []
    for v in vectors:
        t0 = time.perf_counter()
        get_risk_score(v)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

    latencies.sort()
    return latencies


def print_stats(latencies: list[float]) -> None:
    """Print latency statistics to stdout."""
    n = len(latencies)
    mean_ms = sum(latencies) / n
    min_ms = latencies[0]
    max_ms = latencies[-1]
    p95_ms = latencies[int(n * 0.95) - 1]
    p99_ms = latencies[int(n * 0.99) - 1]

    print(f"\n{'='*50}")
    print(f"  Benchmark Results ({n} predictions)")
    print(f"{'='*50}")
    print(f"  min  : {min_ms:.3f} ms")
    print(f"  mean : {mean_ms:.3f} ms")
    print(f"  p95  : {p95_ms:.3f} ms")
    print(f"  p99  : {p99_ms:.3f} ms")
    print(f"  max  : {max_ms:.3f} ms")
    print(f"{'='*50}")

    threshold_status = "✓ PASS" if mean_ms < MEAN_LATENCY_THRESHOLD_MS else "✗ FAIL"
    print(f"  Mean < {MEAN_LATENCY_THRESHOLD_MS}ms threshold: {threshold_status}")
    print(f"{'='*50}\n")

    assert mean_ms < MEAN_LATENCY_THRESHOLD_MS, (
        f"Mean latency {mean_ms:.3f}ms exceeds {MEAN_LATENCY_THRESHOLD_MS}ms threshold. "
        f"Performance requirement not met."
    )


if __name__ == "__main__":
    print(f"Running {N_PREDICTIONS} predictions...")
    latencies = run_benchmark(N_PREDICTIONS)
    print_stats(latencies)
    print("Benchmark complete. All assertions passed.")
