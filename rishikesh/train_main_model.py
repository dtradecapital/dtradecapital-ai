"""
train_main_model.py
-------------------
Generates synthetic training data for 6 behavioral risk classes and trains
a multi-class RandomForestClassifier. Saves the model and metadata to
rishikesh/models/.

Usage (from workspace root):
    python rishikesh/train_main_model.py

CRITICAL: No feature scaling is applied. Inputs are pre-scaled by Mugeesh's
pipeline and must be passed to the model as-is.
"""

import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASSES = [
    "NORMAL",
    "REVENGE_TRADING",
    "OVERTRADING",
    "IMPULSIVE_ENTRY",
    "FATIGUE_TRADING",
    "TILT",
]

N_FEATURES = 20
SAMPLES_PER_CLASS = 200
RANDOM_STATE = 42

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "main_model.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def _generate_class_samples(label: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate n synthetic 20-feature samples for a given behavior class.

    Feature index reference (all values are pre-scaled / raw domain values):
      0  win_rate                  1  trade_count_session
      2  after_loss_flag           3  rapid_reentry_flag
      4  session_duration_minutes  5  avg_hold_time_minutes
      6  risk_reward_ratio         7  position_size_ratio
      8  drawdown_pct              9  trades_per_hour
     10  consecutive_losses       11  hour_of_day
     12  day_of_week              13  pnl_last_trade
     14  pnl_session_total        15  session_high_pnl
     16  volatility_index         17  news_event_flag
     18  avg_slippage             19  emotional_score
    """
    base = rng.uniform(0.0, 1.0, size=(n, N_FEATURES))

    if label == "NORMAL":
        # High win rate, low trade count, no loss flags, positive PnL
        base[:, 0] = rng.uniform(0.55, 0.90, n)   # win_rate high
        base[:, 1] = rng.uniform(1, 8, n)          # trade_count_session low
        base[:, 2] = rng.choice([0], n)             # after_loss_flag off
        base[:, 3] = rng.choice([0], n)             # rapid_reentry_flag off
        base[:, 10] = rng.uniform(0, 1, n)          # consecutive_losses low
        base[:, 13] = rng.uniform(10, 200, n)       # pnl_last_trade positive
        base[:, 19] = rng.uniform(0.6, 1.0, n)     # emotional_score calm

    elif label == "REVENGE_TRADING":
        # After a loss, rapid re-entry, negative PnL, high emotional score
        base[:, 0] = rng.uniform(0.10, 0.40, n)    # win_rate low
        base[:, 2] = rng.choice([1], n)             # after_loss_flag on
        base[:, 3] = rng.choice([1], n)             # rapid_reentry_flag on
        base[:, 10] = rng.uniform(3, 8, n)          # consecutive_losses high
        base[:, 13] = rng.uniform(-300, -50, n)     # pnl_last_trade negative
        base[:, 19] = rng.uniform(0.0, 0.3, n)     # emotional_score stressed

    elif label == "OVERTRADING":
        # Very high trade count, high trades_per_hour, long session
        base[:, 1] = rng.uniform(20, 60, n)         # trade_count_session very high
        base[:, 4] = rng.uniform(300, 600, n)       # session_duration_minutes long
        base[:, 9] = rng.uniform(8, 20, n)          # trades_per_hour high
        base[:, 5] = rng.uniform(0.5, 3.0, n)      # avg_hold_time_minutes short
        base[:, 14] = rng.uniform(-500, 100, n)     # pnl_session_total declining

    elif label == "IMPULSIVE_ENTRY":
        # Poor risk/reward, large position size, no setup
        base[:, 6] = rng.uniform(0.1, 0.5, n)      # risk_reward_ratio poor
        base[:, 7] = rng.uniform(0.7, 1.0, n)      # position_size_ratio large
        base[:, 3] = rng.choice([1], n)             # rapid_reentry_flag on
        base[:, 5] = rng.uniform(0.1, 1.0, n)      # avg_hold_time_minutes very short
        base[:, 19] = rng.uniform(0.0, 0.4, n)     # emotional_score impulsive

    elif label == "FATIGUE_TRADING":
        # Off-hours trading, very long session, high slippage
        base[:, 11] = rng.choice([0, 1, 2, 22, 23], n)  # hour_of_day off-hours
        base[:, 4] = rng.uniform(400, 720, n)            # session_duration_minutes very long
        base[:, 18] = rng.uniform(0.05, 0.15, n)        # avg_slippage high
        base[:, 9] = rng.uniform(0.5, 2.0, n)           # trades_per_hour low (tired)
        base[:, 19] = rng.uniform(0.1, 0.4, n)          # emotional_score fatigued

    elif label == "TILT":
        # Multiple simultaneous risk signals: loss streak + revenge + overtrading
        base[:, 0] = rng.uniform(0.05, 0.25, n)    # win_rate very low
        base[:, 2] = rng.choice([1], n)             # after_loss_flag on
        base[:, 3] = rng.choice([1], n)             # rapid_reentry_flag on
        base[:, 1] = rng.uniform(15, 50, n)         # trade_count_session high
        base[:, 10] = rng.uniform(5, 10, n)         # consecutive_losses very high
        base[:, 8] = rng.uniform(0.15, 0.40, n)    # drawdown_pct severe
        base[:, 13] = rng.uniform(-500, -100, n)    # pnl_last_trade very negative
        base[:, 19] = rng.uniform(0.0, 0.2, n)     # emotional_score extreme stress

    return base


def generate_training_data(
    samples_per_class: int = SAMPLES_PER_CLASS,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) arrays for all 6 classes."""
    rng = np.random.default_rng(random_state)
    X_parts, y_parts = [], []
    for label in CLASSES:
        X_parts.append(_generate_class_samples(label, samples_per_class, rng))
        y_parts.extend([label] * samples_per_class)
    X = np.vstack(X_parts)
    y = np.array(y_parts)
    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_save(
    samples_per_class: int = SAMPLES_PER_CLASS,
    random_state: int = RANDOM_STATE,
) -> None:
    print("Generating synthetic training data...")
    X, y = generate_training_data(samples_per_class, random_state)
    print(f"  Total samples: {len(X)} ({samples_per_class} per class × {len(CLASSES)} classes)")

    print("Training RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X, y)
    print(f"  Classes learned: {list(clf.classes_)}")

    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(clf, MODEL_PATH)
    print(f"  Model saved → {MODEL_PATH}")

    metadata = {
        "model_type": "RandomForestClassifier",
        "n_features": N_FEATURES,
        "classes": list(clf.classes_),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved → {METADATA_PATH}")

    print("Training complete.")


if __name__ == "__main__":
    train_and_save()
