# Requirements Document

## Introduction

The AI Model Engineer module (Rishikesh's system) is a self-contained behavioral risk detection component within the D Terminal trading platform. It receives exactly 20 pre-scaled behavioral feature values from Mugeesh's upstream feature pipeline, runs them through a trained machine learning model, and returns a structured 6-field risk prediction to Varun's downstream API.

The module is responsible for model training, inference, caching, logging, benchmarking, and testing. It must **never** re-scale or transform the input features, as they arrive already normalized. All new code lives under the `rishikesh/` directory at the workspace root.

---

## Glossary

| Term | Definition |
|---|---|
| **Inference_Engine** | The `model_inference.py` module responsible for loading the trained model and computing risk predictions from a 20-float Feature_Vector. |
| **Model_Service** | The `ModelService` class in `model_service.py` that wraps the Inference_Engine with caching, timing, and structured logging. |
| **Feature_Vector** | A list of exactly 20 floats, already normalized by Mugeesh's upstream pipeline, representing behavioral signals for a single trade event. Must never be re-scaled. |
| **Risk_Prediction** | The 6-field dict returned by every prediction call: `risk_score`, `behavior_type`, `confidence`, `sub_scores`, `alert_message`, `intervention_level`. |
| **Behavior_Type** | One of six mutually exclusive behavioral classifications: `NORMAL`, `REVENGE_TRADING`, `OVERTRADING`, `IMPULSIVE_ENTRY`, `FATIGUE_TRADING`, `TILT`. |
| **Intervention_Level** | One of three action levels derived from `risk_score`: `NONE` (0–39), `WARN` (40–69), `BLOCK` (70–100). |
| **Main_Model** | The primary multi-class RandomForestClassifier saved as `rishikesh/models/main_model.pkl`, trained to classify all 6 Behavior_Types. |
| **Specialist_Model** | One of three binary classifiers (`revenge_model.pkl`, `overtrading_model.pkl`, `fatigue_model.pkl`), each trained to distinguish one Behavior_Type from all others. |
| **Prediction_Cache** | An in-memory dict keyed by `"{user_id}:{trade_id}"` that stores previously computed Risk_Predictions to avoid redundant computation. |
| **Prediction_Log** | The file `rishikesh/predictions.log`, auto-created at runtime, recording one line per new (non-cached) prediction. |
| **Benchmark_Runner** | The `benchmark.py` script that executes 1000 predictions and reports latency statistics. |
| **Session_Trade_Count** | Feature index 3 (0-indexed) in the Feature_Vector, representing the number of trades placed in the current session. Used in the OVERTRADING alert message. |

---

## System Context

```
Mugeesh's Pipeline  →  [20 pre-scaled floats]  →  Rishikesh's Module  →  [Risk_Prediction dict]  →  Varun's API
```

**Critical contract:** The Feature_Vector arrives fully normalized. The Inference_Engine must pass it directly to the model without any transformation.

---

## File Structure

```
rishikesh/
├── models/
│   ├── main_model.pkl            # Multi-class RandomForest (6 classes)
│   ├── revenge_model.pkl         # Binary: REVENGE_TRADING vs. rest
│   ├── overtrading_model.pkl     # Binary: OVERTRADING vs. rest
│   ├── fatigue_model.pkl         # Binary: FATIGUE_TRADING vs. rest
│   └── model_metadata.json       # Training metadata
├── model_inference.py            # get_risk_score() — core inference
├── model_service.py              # ModelService — cache + logging wrapper
├── train_main_model.py           # Training script for main_model.pkl
├── train_specialist_models.py    # Training script for 3 specialist models
├── benchmark.py                  # Latency benchmark (1000 predictions)
├── predictions.log               # Auto-created at runtime
└── tests/
    ├── __init__.py
    ├── test_inference.py         # Property-based tests for get_risk_score
    └── test_service.py           # Property-based tests for ModelService
```

---

## Feature Vector Schema

The 20 features are always provided in this fixed order (0-indexed):

| Index | Feature | Description |
|---|---|---|
| 0 | `win_rate` | Ratio of winning trades |
| 1 | `trade_count_session` | Number of trades in current session |
| 2 | `after_loss_flag` | 1 if previous trade was a loss, else 0 |
| 3 | `rapid_reentry_flag` | 1 if re-entered within 60s of a loss, else 0 |
| 4 | `session_duration_minutes` | Total session duration in minutes |
| 5 | `avg_hold_time_minutes` | Average trade hold time in minutes |
| 6 | `risk_reward_ratio` | Average risk-to-reward ratio |
| 7 | `position_size_ratio` | Position size as fraction of account |
| 8 | `drawdown_pct` | Current drawdown percentage |
| 9 | `trades_per_hour` | Trade frequency per hour |
| 10 | `consecutive_losses` | Number of consecutive losing trades |
| 11 | `hour_of_day` | Hour of day (0–23) |
| 12 | `day_of_week` | Day of week (0=Mon, 6=Sun) |
| 13 | `pnl_last_trade` | P&L of the most recent trade |
| 14 | `pnl_session_total` | Total session P&L |
| 15 | `session_high_pnl` | Highest P&L reached in session |
| 16 | `volatility_index` | Market volatility index |
| 17 | `news_event_flag` | 1 if major news event active, else 0 |
| 18 | `avg_slippage` | Average slippage per trade |
| 19 | `emotional_score` | Composite emotional state score |

> **Note:** `trade_count_session` (index 1) is used to populate `{n}` in the OVERTRADING alert message. Cast to `int` before interpolation.

---

## Requirements

### Requirement 1: Feature Vector Integrity

**User Story:** As Rishikesh (AI Model Engineer), I want the system to strictly validate and preserve the input Feature_Vector, so that model predictions are never corrupted by malformed or re-scaled inputs.

#### Acceptance Criteria

1. WHEN a Feature_Vector of exactly 20 numeric values is provided, THE Inference_Engine SHALL accept it without modification or re-scaling.
2. WHEN a Feature_Vector with fewer or more than 20 elements is provided, THE Inference_Engine SHALL raise a `ValueError` with a message that includes the actual length received and the expected length of 20.
3. WHEN any element of a Feature_Vector is not a numeric type (`int` or `float`), THE Inference_Engine SHALL raise a `ValueError` with a message identifying the offending index and type.
4. THE Inference_Engine SHALL never apply any normalization, standardization, min-max scaling, or any other transformation to the input Feature_Vector before passing it to the model.
5. WHEN a Feature_Vector contains `NaN` or `inf` values, THE Inference_Engine SHALL raise a `ValueError` with a descriptive message.

---

### Requirement 2: Model Loading and Initialization

**User Story:** As Rishikesh, I want the trained model to be loaded once at module import time, so that per-call latency is not inflated by repeated disk I/O.

#### Acceptance Criteria

1. WHEN the `model_inference` module is imported, THE Inference_Engine SHALL load `main_model.pkl` from `rishikesh/models/main_model.pkl` exactly once using `joblib.load`.
2. WHILE the `model_inference` module remains imported in the same Python process, THE Inference_Engine SHALL reuse the same in-memory model object for all subsequent `get_risk_score` calls without reloading from disk.
3. IF `main_model.pkl` does not exist at the expected path at import time, THEN THE Inference_Engine SHALL raise a `FileNotFoundError` with the full expected file path included in the error message.
4. THE model object loaded at import time SHALL be the sole model used for all inference; no other model file SHALL be loaded during `get_risk_score` execution.

---

### Requirement 3: Risk Prediction Output Structure

**User Story:** As Varun (API Engineer), I want every prediction call to return a complete, consistently structured 6-field dict, so that my API can reliably serialize and forward the result.

#### Acceptance Criteria

1. WHEN `get_risk_score` is called with a valid Feature_Vector, THE Inference_Engine SHALL return a `dict` containing **exactly** the following keys and no others: `risk_score`, `behavior_type`, `confidence`, `sub_scores`, `alert_message`, `intervention_level`.
2. THE `risk_score` field SHALL be of Python type `int` with a value in the inclusive range 0 to 100.
3. THE `behavior_type` field SHALL be a `str` equal to exactly one of: `NORMAL`, `REVENGE_TRADING`, `OVERTRADING`, `IMPULSIVE_ENTRY`, `FATIGUE_TRADING`, `TILT`.
4. THE `confidence` field SHALL be a `float` with a value in the inclusive range 0.0 to 1.0.
5. THE `sub_scores` field SHALL be a `dict` containing **exactly** the following 6 keys: `normal_probability`, `revenge_trading_probability`, `overtrading_probability`, `impulsive_entry_probability`, `fatigue_trading_probability`, `tilt_probability`, each mapped to a `float` value in the range 0.0 to 1.0.
6. THE Inference_Engine SHALL ensure the sum of all 6 values in `sub_scores` equals 1.0 within a tolerance of ±0.01.
7. THE `behavior_type` SHALL be the class label corresponding to the highest probability value in `sub_scores`.
8. THE `risk_score` SHALL be computed as `int(highest_class_probability * 100)`, where `highest_class_probability` is the maximum value among all 6 `sub_scores` values.
9. THE `confidence` SHALL be set equal to `highest_class_probability` as a `float`.
10. THE `sub_scores` probabilities SHALL be sourced directly from `model.predict_proba()` output, mapped to their corresponding class labels in the order returned by `model.classes_`.

---

### Requirement 4: Intervention Level Thresholds

**User Story:** As Varun, I want the intervention level to be deterministically derived from the risk score, so that downstream enforcement logic can rely on a single consistent rule.

#### Acceptance Criteria

1. WHEN `risk_score` is in the range 0 to 39 inclusive, THE Inference_Engine SHALL set `intervention_level` to `"NONE"`.
2. WHEN `risk_score` is in the range 40 to 69 inclusive, THE Inference_Engine SHALL set `intervention_level` to `"WARN"`.
3. WHEN `risk_score` is in the range 70 to 100 inclusive, THE Inference_Engine SHALL set `intervention_level` to `"BLOCK"`.
4. FOR ALL valid Feature_Vectors, the `intervention_level` in the returned Risk_Prediction SHALL be consistent with the `risk_score` in the same Risk_Prediction according to the thresholds above — no exceptions.
5. THE threshold boundaries SHALL be treated as inclusive on both ends: `risk_score == 39` → `NONE`, `risk_score == 40` → `WARN`, `risk_score == 69` → `WARN`, `risk_score == 70` → `BLOCK`.

---

### Requirement 5: Alert Messages

**User Story:** As a trader, I want to receive a human-readable alert message with each prediction, so that I understand what behavioral risk has been detected and what action to take.

#### Acceptance Criteria

1. WHEN `behavior_type` is `NORMAL`, THE Inference_Engine SHALL set `alert_message` to exactly: `"Trading behavior looks healthy. Continue as planned."`.
2. WHEN `behavior_type` is `REVENGE_TRADING`, THE Inference_Engine SHALL set `alert_message` to exactly: `"Possible emotional trading detected after a loss. Consider stepping back."`.
3. WHEN `behavior_type` is `OVERTRADING`, THE Inference_Engine SHALL set `alert_message` to a string in the format: `"You have placed {n} trades this session. Overtrading detected — step away."`, where `{n}` is the integer value of `trade_count_session` (Feature_Vector index 1, cast to `int`).
4. WHEN `behavior_type` is `IMPULSIVE_ENTRY`, THE Inference_Engine SHALL set `alert_message` to exactly: `"Trade entry detected without proper setup. Slow down and review your plan."`.
5. WHEN `behavior_type` is `FATIGUE_TRADING`, THE Inference_Engine SHALL set `alert_message` to exactly: `"Trading fatigue detected. You may be trading during off-hours or after a long session."`.
6. WHEN `behavior_type` is `TILT`, THE Inference_Engine SHALL set `alert_message` to exactly: `"Multiple risk signals detected simultaneously. This is a high-risk state. Trade blocked."`.
7. THE `alert_message` SHALL be a non-empty string for every valid Behavior_Type — no null, empty string, or placeholder value is permitted.

---

### Requirement 6: Prediction Caching

**User Story:** As Rishikesh, I want identical `user_id` + `trade_id` pairs to return cached results, so that repeated calls for the same trade do not waste compute resources.

#### Acceptance Criteria

1. WHEN `predict` is called with a `user_id` and `trade_id` that have been previously scored in the same `ModelService` instance, THE Model_Service SHALL return the cached Risk_Prediction without invoking `get_risk_score` again.
2. WHEN `predict` is called with a `user_id` and `trade_id` combination not present in the Prediction_Cache, THE Model_Service SHALL invoke `get_risk_score`, store the result in the Prediction_Cache keyed by `f"{user_id}:{trade_id}"`, and return the result.
3. THE Model_Service SHALL maintain independent cache entries for different `trade_id` values even when the `user_id` is the same (e.g., `"U001:T001"` and `"U001:T002"` are separate entries).
4. THE Model_Service SHALL maintain independent cache entries for different `user_id` values even when the `trade_id` is the same (e.g., `"U001:T001"` and `"U002:T001"` are separate entries).
5. THE cached Risk_Prediction returned on a cache hit SHALL be identical (same dict contents) to the Risk_Prediction originally computed and stored.
6. THE Prediction_Cache SHALL be instance-scoped — a new `ModelService()` instance SHALL start with an empty cache.

---

### Requirement 7: Prediction Logging

**User Story:** As Rishikesh, I want every new prediction to be appended to `predictions.log`, so that I can audit model behavior and diagnose latency issues in production.

#### Acceptance Criteria

1. WHEN a new prediction is computed (not served from cache), THE Model_Service SHALL append exactly one line to `rishikesh/predictions.log` in the format: `[TIMESTAMP] trade_id={trade_id} behavior={behavior_type} risk={risk_score} ms={ms_taken}`, where `TIMESTAMP` is an ISO 8601 datetime string and `ms_taken` is rounded to 2 decimal places.
2. THE Model_Service SHALL measure prediction latency in milliseconds starting immediately before the `get_risk_score` call and ending immediately after it returns.
3. WHEN prediction latency exceeds 100 milliseconds, THE Model_Service SHALL additionally emit a `WARNING` level log message via Python's `logging` module indicating the trade_id and the actual latency.
4. IF `predictions.log` does not exist at the time of the first prediction, THEN THE Model_Service SHALL create it automatically (including any missing parent directories).
5. THE Model_Service SHALL NOT write any log entry to `predictions.log` for predictions served from the Prediction_Cache.
6. THE log file SHALL use append mode so that existing entries are never overwritten across restarts.

---

### Requirement 8: Main Model Training

**User Story:** As Rishikesh, I want a reproducible training script that generates synthetic data and saves a trained multi-class classifier, so that the Inference_Engine always has a valid model to load.

#### Acceptance Criteria

1. WHEN `train_main_model.py` is executed, THE Main_Model training script SHALL generate synthetic training data with balanced representation across all 6 Behavior_Type classes (minimum 100 samples per class).
2. WHEN training completes, THE Main_Model training script SHALL save the trained classifier to `rishikesh/models/main_model.pkl` using `joblib.dump`, creating the `rishikesh/models/` directory if it does not exist.
3. WHEN training completes, THE Main_Model training script SHALL save a `model_metadata.json` file to `rishikesh/models/` containing at minimum: `model_type` (string), `n_features` (integer, must be 20), `classes` (list of 6 Behavior_Type strings in label order), and `trained_at` (ISO 8601 UTC timestamp string).
4. THE Main_Model training script SHALL train a `RandomForestClassifier` with a fixed `random_state` for reproducibility, on exactly 20 input features.
5. THE Main_Model training script SHALL NOT apply any feature scaling (e.g., `StandardScaler`, `MinMaxScaler`) to the training data, preserving the contract that inputs arrive pre-scaled.
6. THE trained model SHALL expose a `predict_proba` method and a `classes_` attribute, as required by the Inference_Engine.
7. WHEN `train_main_model.py` is executed, it SHALL print a confirmation message to stdout indicating the save path and training completion.

---

### Requirement 9: Specialist Model Training

**User Story:** As Rishikesh, I want three binary specialist classifiers trained for revenge, overtrading, and fatigue detection, so that targeted sub-scores can be computed independently.

#### Acceptance Criteria

1. WHEN `train_specialist_models.py` is executed, THE Specialist_Model training script SHALL train a binary classifier for `REVENGE_TRADING` vs. all other classes and save it to `rishikesh/models/revenge_model.pkl`.
2. WHEN `train_specialist_models.py` is executed, THE Specialist_Model training script SHALL train a binary classifier for `OVERTRADING` vs. all other classes and save it to `rishikesh/models/overtrading_model.pkl`.
3. WHEN `train_specialist_models.py` is executed, THE Specialist_Model training script SHALL train a binary classifier for `FATIGUE_TRADING` vs. all other classes and save it to `rishikesh/models/fatigue_model.pkl`.
4. THE Specialist_Model training script SHALL use the same 20-feature input format and the same synthetic data generation approach as the Main_Model training script.
5. THE Specialist_Model training script SHALL NOT apply any feature scaling to the training data.
6. WHEN `train_specialist_models.py` is executed, it SHALL print a confirmation message for each model saved, including the save path.

---

### Requirement 10: Latency Benchmarking

**User Story:** As Rishikesh, I want a benchmark script that validates inference latency at scale, so that I can confirm the system meets the sub-50ms mean latency requirement before deployment.

#### Acceptance Criteria

1. WHEN `benchmark.py` is executed, THE Benchmark_Runner SHALL generate 1000 random valid Feature_Vectors (each containing 20 floats in the range 0.0 to 1.0) and call `get_risk_score` for each.
2. WHEN benchmarking completes, THE Benchmark_Runner SHALL print the following latency statistics in milliseconds: minimum, maximum, mean, 95th percentile (p95), and 99th percentile (p99).
3. WHEN benchmarking completes, THE Benchmark_Runner SHALL assert that mean latency is strictly less than 50 milliseconds, raising an `AssertionError` with the actual mean value if the assertion fails.
4. THE Benchmark_Runner SHALL measure per-call latency using `time.perf_counter` for high-resolution timing.
5. THE Benchmark_Runner SHALL NOT use the `ModelService` cache — each of the 1000 calls SHALL invoke `get_risk_score` directly to measure raw inference latency.

---

### Requirement 11: Inference Property-Based Tests

**User Story:** As Rishikesh, I want property-based tests covering all output invariants of `get_risk_score`, so that I can detect regressions across the full input space rather than only hand-picked examples.

#### Acceptance Criteria

1. THE `test_inference.py` test suite SHALL use the Hypothesis library (`hypothesis.strategies`) to generate arbitrary valid Feature_Vectors for all property tests.
2. FOR ALL valid 20-float Feature_Vectors, the output of `get_risk_score` SHALL be a `dict` containing **exactly** the 6 required keys: `risk_score`, `behavior_type`, `confidence`, `sub_scores`, `alert_message`, `intervention_level` — no more, no fewer.
3. FOR ALL valid 20-float Feature_Vectors, `risk_score` in the output SHALL be a Python `int` in the inclusive range 0 to 100.
4. FOR ALL valid 20-float Feature_Vectors, `confidence` in the output SHALL be a Python `float` in the inclusive range 0.0 to 1.0.
5. FOR ALL valid 20-float Feature_Vectors, `behavior_type` in the output SHALL be one of the 6 valid Behavior_Type strings.
6. FOR ALL valid 20-float Feature_Vectors, `intervention_level` in the output SHALL be consistent with `risk_score` according to the threshold rules in Requirement 4 (verified by the test, not assumed).
7. FOR ALL valid 20-float Feature_Vectors, `sub_scores` SHALL contain exactly 6 keys and the sum of all values SHALL equal 1.0 within ±0.01.
8. FOR ALL Feature_Vectors with a length other than 20 (both shorter and longer), `get_risk_score` SHALL raise a `ValueError`.
9. FOR ALL Feature_Vectors containing a non-numeric element, `get_risk_score` SHALL raise a `ValueError`.
10. THE `alert_message` in the output SHALL be a non-empty string for all valid inputs.
11. THE `behavior_type` in the output SHALL correspond to the key with the highest value in `sub_scores` (verified by the test).

---

### Requirement 12: Service Property-Based Tests

**User Story:** As Rishikesh, I want property-based tests covering the caching and logging behavior of `ModelService`, so that I can verify correctness across arbitrary user and trade ID combinations.

#### Acceptance Criteria

1. FOR ALL valid `user_id` and `trade_id` string pairs, calling `predict` twice on the same `ModelService` instance with the same pair SHALL return identical Risk_Prediction dicts (cache correctness property).
2. FOR ALL pairs of distinct `trade_id` values with the same `user_id`, the results of `predict` SHALL be independently stored — retrieving one SHALL NOT affect the other (no cross-contamination).
3. FOR ALL valid inputs, `predict` SHALL return a `dict` containing all 6 required Risk_Prediction keys.
4. WHEN `predict` is called with a new `user_id` + `trade_id` combination, THE Model_Service SHALL append exactly one new line to `predictions.log`.
5. WHEN `predict` is called with a previously cached `user_id` + `trade_id` combination, THE Model_Service SHALL NOT append any new line to `predictions.log`.
6. FOR ALL valid inputs, the `predict` return value SHALL satisfy all structural invariants defined in Requirements 3, 4, and 5 (risk_score range, behavior_type validity, intervention_level consistency, sub_scores sum).

---

## Correctness Properties Summary

The following properties must hold for all valid inputs and are verified by the property-based test suite:

| # | Property | Verified In |
|---|---|---|
| P1 | Output always contains exactly 6 keys | test_inference.py |
| P2 | `risk_score` is always `int` in [0, 100] | test_inference.py |
| P3 | `confidence` is always `float` in [0.0, 1.0] | test_inference.py |
| P4 | `behavior_type` is always one of 6 valid strings | test_inference.py |
| P5 | `intervention_level` is always consistent with `risk_score` | test_inference.py |
| P6 | `sub_scores` always has 6 keys summing to 1.0 ±0.01 | test_inference.py |
| P7 | `behavior_type` matches the highest `sub_scores` key | test_inference.py |
| P8 | `alert_message` is always a non-empty string | test_inference.py |
| P9 | Invalid length input always raises `ValueError` | test_inference.py |
| P10 | Same `user_id:trade_id` always returns identical result | test_service.py |
| P11 | Different `trade_id`s are independently cached | test_service.py |
| P12 | New predictions always produce a log entry | test_service.py |
| P13 | Cached predictions never produce a log entry | test_service.py |
