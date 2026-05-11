# Implementation Plan: Behavioral Risk Detection Engine

## Overview

Implement the `rishikesh/` module from scratch: directory scaffolding, training scripts, inference engine, service layer, benchmark, and property-based test suites. Each task builds on the previous so that the module is always in a runnable state. Training scripts are executed to produce the `.pkl` model files before any inference code is exercised.

## Tasks

- [x] 1. Scaffold the `rishikesh/` directory structure
  - Create `rishikesh/models/` directory (with a `.gitkeep` so it is tracked)
  - Create `rishikesh/tests/__init__.py` (empty file to make `tests/` a package)
  - Create `rishikesh/__init__.py` (empty, marks the directory as a Python package)
  - _Requirements: File Structure (requirements.md)_

- [x] 2. Implement `train_main_model.py` — main multi-class model
  - [x] 2.1 Write synthetic data generator for 6 Behavior_Type classes
    - Generate ≥100 samples per class (≥600 total) with 20 features each
    - Use class-specific feature distributions so classes are distinguishable
    - Use `random_state=42` throughout for reproducibility
    - _Requirements: 8.1, 8.4_
  - [x] 2.2 Train and save the `RandomForestClassifier`
    - Fit `RandomForestClassifier(random_state=42)` on the synthetic data
    - Do NOT apply any feature scaling (`StandardScaler`, `MinMaxScaler`, etc.)
    - Save to `rishikesh/models/main_model.pkl` via `joblib.dump`; create directory with `os.makedirs(..., exist_ok=True)`
    - Save `rishikesh/models/model_metadata.json` with keys `model_type`, `n_features` (20), `classes`, `trained_at` (ISO 8601 UTC)
    - Print confirmation message to stdout with save path
    - _Requirements: 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_

- [x] 3. Implement `train_specialist_models.py` — three binary classifiers
  - [x] 3.1 Reuse the same synthetic data generation approach as task 2.1
    - Generate the same 20-feature dataset; no feature scaling applied
    - _Requirements: 9.4, 9.5_
  - [x] 3.2 Train and save each binary classifier
    - `REVENGE_TRADING` vs. all others → `rishikesh/models/revenge_model.pkl`
    - `OVERTRADING` vs. all others → `rishikesh/models/overtrading_model.pkl`
    - `FATIGUE_TRADING` vs. all others → `rishikesh/models/fatigue_model.pkl`
    - Each uses `RandomForestClassifier(random_state=42)`
    - Print a confirmation line per saved model including the save path
    - _Requirements: 9.1, 9.2, 9.3, 9.6_

- [x] 4. Run training scripts to generate model files
  - Execute `python rishikesh/train_main_model.py` from the workspace root
  - Execute `python rishikesh/train_specialist_models.py` from the workspace root
  - Verify that `rishikesh/models/main_model.pkl`, `model_metadata.json`, and all three specialist `.pkl` files exist after execution
  - _Requirements: 8.2, 8.3, 9.1, 9.2, 9.3_

- [x] 5. Checkpoint — verify training artifacts
  - Ensure all four `.pkl` files and `model_metadata.json` are present in `rishikesh/models/`
  - Ask the user if questions arise before proceeding to inference code.

- [x] 6. Implement `model_inference.py` — core inference engine
  - [x] 6.1 Add module-level model loading
    - Import `joblib`, `math`, `os`
    - Set `_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "main_model.pkl")`
    - Load `_model = joblib.load(_MODEL_PATH)` at module import time (not inside the function)
    - If the file is absent, let `FileNotFoundError` propagate unchanged
    - _Requirements: 2.1, 2.2, 2.3, 2.4_
  - [x] 6.2 Implement input validation in `get_risk_score`
    - Check `len(features) == 20`; raise `ValueError(f"Expected 20 features, got {len(features)}")` if not
    - For each element check `isinstance(v, (int, float))`; raise `ValueError(f"Feature at index {i} is not numeric: type={type(v)}")` if not
    - For each element check `math.isfinite(v)`; raise `ValueError(f"Feature at index {i} contains NaN or inf: value={v}")` if not
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  - [x] 6.3 Implement inference and output dict construction
    - Call `_model.predict_proba([features])[0]` to get a 6-element probability array
    - Build `sub_scores` by mapping `_model.classes_` labels to their probability keys using `_KEY_MAP`
    - Set `behavior_type` to the class with the highest probability (`proba.argmax()`)
    - Set `risk_score = int(highest_prob * 100)`, `confidence = float(highest_prob)`
    - Apply intervention level thresholds: 0–39 → `"NONE"`, 40–69 → `"WARN"`, 70–100 → `"BLOCK"`
    - Set `alert_message` from the per-behavior-type mapping (OVERTRADING uses `int(features[1])`)
    - Return the 6-key dict
    - _Requirements: 3.1–3.10, 4.1–4.5, 5.1–5.7_
  - [ ]* 6.4 Write property test for output structure (P1)
    - **Property 1: Output has exactly 6 keys**
    - **Validates: Requirements 3.1, 11.2**
  - [ ]* 6.5 Write property test for risk_score range (P2)
    - **Property 2: risk_score is int in [0, 100]**
    - **Validates: Requirements 3.2, 11.3**
  - [ ]* 6.6 Write property test for confidence range (P3)
    - **Property 3: confidence is float in [0.0, 1.0]**
    - **Validates: Requirements 3.4, 11.4**
  - [ ]* 6.7 Write property test for behavior_type validity (P4)
    - **Property 4: behavior_type is one of 6 valid strings**
    - **Validates: Requirements 3.3, 11.5**
  - [ ]* 6.8 Write property test for intervention_level consistency (P5)
    - **Property 5: intervention_level is consistent with risk_score**
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 11.6**
  - [ ]* 6.9 Write property test for sub_scores structure and sum (P6)
    - **Property 6: sub_scores has 6 keys summing to 1.0 ±0.01**
    - **Validates: Requirements 3.5, 3.6, 11.7**
  - [ ]* 6.10 Write property test for behavior_type argmax (P7)
    - **Property 7: behavior_type matches highest sub_scores key**
    - **Validates: Requirements 3.7, 11.11**
  - [ ]* 6.11 Write property test for alert_message non-empty (P8)
    - **Property 8: alert_message is a non-empty string**
    - **Validates: Requirements 5.7, 11.10**
  - [ ]* 6.12 Write property test for invalid-length input (P9)
    - **Property 9: Invalid-length input raises ValueError**
    - **Validates: Requirements 1.2, 11.8**

- [x] 7. Implement `tests/test_inference.py` — full inference test suite
  - [x] 7.1 Set up Hypothesis strategies and imports
    - Import `pytest`, `hypothesis`, `hypothesis.strategies as st`, `math`
    - Define `valid_feature_vector = st.lists(st.floats(allow_nan=False, allow_infinity=False, allow_subnormal=False), min_size=20, max_size=20)`
    - Define `invalid_length_vector = st.one_of(st.lists(..., max_size=19), st.lists(..., min_size=21))`
    - Apply `@settings(max_examples=100)` to all property tests
    - _Requirements: 11.1_
  - [x] 7.2 Write all 9 property-based tests (P1–P9)
    - `test_output_has_exactly_6_keys` — P1
    - `test_risk_score_is_int_in_range` — P2
    - `test_confidence_is_float_in_range` — P3
    - `test_behavior_type_is_valid` — P4
    - `test_intervention_level_consistent` — P5
    - `test_sub_scores_structure_and_sum` — P6
    - `test_behavior_type_matches_argmax` — P7
    - `test_alert_message_nonempty` — P8
    - `test_invalid_length_raises_valueerror` — P9
    - Annotate each with `# Feature: behavioral-risk-detection-engine, Property {N}: {property_text}`
    - _Requirements: 11.2–11.11_
  - [ ]* 7.3 Write example-based / smoke tests
    - `test_nan_input_raises_valueerror` — pass vector with `float('nan')`
    - `test_inf_input_raises_valueerror` — pass vector with `float('inf')`
    - `test_overtrading_alert_contains_trade_count` — verify `{n}` interpolation uses `int(features[1])`
    - `test_exact_alert_messages` — verify fixed-string messages for all non-OVERTRADING behavior types
    - `test_model_loaded_at_import` — verify `model_inference._model` is not `None`
    - _Requirements: 1.5, 2.1, 5.1–5.7_

- [x] 8. Implement `model_service.py` — caching and logging wrapper
  - [x] 8.1 Implement `ModelService.__init__`
    - Initialize `self._cache: dict[str, dict] = {}`
    - Configure Python `logging` to write to `rishikesh/predictions.log` in append mode
    - Use `os.makedirs(log_dir, exist_ok=True)` to ensure the log directory exists
    - _Requirements: 6.6, 7.4, 7.6_
  - [x] 8.2 Implement `ModelService.predict`
    - Build `cache_key = f"{user_id}:{trade_id}"`
    - Return cached result immediately on cache hit (no log write, no `get_risk_score` call)
    - On cache miss: time the `get_risk_score` call with `time.perf_counter`, store result in cache, append log line
    - Log line format: `[{ISO8601_TIMESTAMP}] trade_id={trade_id} behavior={behavior_type} risk={risk_score} ms={ms_taken:.2f}`
    - Emit `logging.warning(...)` if `ms_taken > 100`
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 7.1, 7.2, 7.3, 7.5_
  - [ ]* 8.3 Write property test for cache correctness (P10)
    - **Property 10: Same user_id:trade_id returns identical cached result**
    - **Validates: Requirements 6.1, 6.5, 12.1**
  - [ ]* 8.4 Write property test for cache independence (P11)
    - **Property 11: Different trade_ids are independently cached**
    - **Validates: Requirements 6.3, 6.4, 12.2**
  - [ ]* 8.5 Write property test for new prediction log append (P12)
    - **Property 12: New prediction appends exactly one log line**
    - **Validates: Requirements 7.1, 12.4**
  - [ ]* 8.6 Write property test for cached prediction no log write (P13)
    - **Property 13: Cached prediction does not append to log**
    - **Validates: Requirements 7.5, 12.5**

- [x] 9. Implement `tests/test_service.py` — full service test suite
  - [x] 9.1 Set up Hypothesis strategies and imports
    - Import `pytest`, `hypothesis`, `hypothesis.strategies as st`, `unittest.mock`, `tempfile`, `os`
    - Define `user_id_strategy = st.text(min_size=1)` and `trade_id_strategy = st.text(min_size=1)`
    - Apply `@settings(max_examples=100)` to all property tests
    - Use `tmp_path` or `tempfile.TemporaryDirectory` to isolate log files per test
    - _Requirements: 12.1_
  - [x] 9.2 Write all 4 property-based tests (P10–P13)
    - `test_same_pair_returns_identical_result` — P10
    - `test_different_trade_ids_are_independent` — P11 (use `st.assume(trade_id_1 != trade_id_2)`)
    - `test_new_prediction_appends_log_line` — P12
    - `test_cached_prediction_no_log_write` — P13
    - Annotate each with `# Feature: behavioral-risk-detection-engine, Property {N}: {property_text}`
    - _Requirements: 12.1–12.5_
  - [ ]* 9.3 Write example-based / smoke tests
    - `test_new_instance_has_empty_cache` — create two `ModelService` instances, verify independent caches
    - `test_slow_prediction_emits_warning` — mock `time.perf_counter` to simulate >100 ms, verify `logging.warning` called
    - `test_log_file_created_if_missing` — delete log file, call `predict`, verify file exists
    - _Requirements: 6.6, 7.3, 7.4_

- [x] 10. Checkpoint — run the full test suite
  - Run `python -m pytest rishikesh/tests/ -v` from the workspace root
  - All property tests (P1–P13) and example tests must pass
  - Fix any failures before proceeding to the benchmark
  - Ask the user if questions arise.

- [x] 11. Implement `benchmark.py` — latency benchmark
  - Generate 1000 random Feature_Vectors: `[[random.uniform(0.0, 1.0) for _ in range(20)] for _ in range(1000)]`
  - Time each `get_risk_score` call with `time.perf_counter`; collect latencies in milliseconds
  - Sort latencies; compute and print `min`, `max`, `mean`, `p95` (index 949), `p99` (index 989)
  - Assert `mean_ms < 50`; raise `AssertionError` with actual mean if assertion fails
  - Call `get_risk_score` directly — do NOT use `ModelService`
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [x] 12. Run benchmark and final validation
  - Execute `python rishikesh/benchmark.py` from the workspace root
  - Confirm the assertion passes (mean latency < 50 ms) and latency stats are printed
  - Re-run `python -m pytest rishikesh/tests/ -v` to confirm all tests still pass after benchmark run
  - Ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Training scripts (task 4) must be run before any inference code is imported or tested
- The `rishikesh/` directory is the workspace root for all `python` commands — use `python rishikesh/<script>.py` from the repo root, or `cd rishikesh && python <script>.py`
- `model_inference.py` loads the model at import time; tests that need to control the model path should use `importlib.reload` or mock `joblib.load`
- Log file isolation in service tests is critical — use `tmp_path` or `tempfile` to avoid cross-test contamination
- Property tests use `@settings(max_examples=100)` minimum; increase for deeper coverage
- Each property test is annotated with its property number for traceability back to the design document
