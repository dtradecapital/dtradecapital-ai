"""
train_specialist_models.py
--------------------------
Trains three binary specialist classifiers:
  - revenge_model.pkl    : REVENGE_TRADING vs. all others
  - overtrading_model.pkl: OVERTRADING vs. all others
  - fatigue_model.pkl    : FATIGUE_TRADING vs. all others

Uses the same synthetic data generation approach as train_main_model.py.
No feature scaling is applied.

Usage (from workspace root):
    python rishikesh/train_specialist_models.py
"""

import os

import joblib
from sklearn.ensemble import RandomForestClassifier

from train_main_model import MODELS_DIR, RANDOM_STATE, generate_training_data

# ---------------------------------------------------------------------------
# Specialist model definitions
# ---------------------------------------------------------------------------

SPECIALIST_MODELS = [
    {
        "positive_class": "REVENGE_TRADING",
        "filename": "revenge_model.pkl",
    },
    {
        "positive_class": "OVERTRADING",
        "filename": "overtrading_model.pkl",
    },
    {
        "positive_class": "FATIGUE_TRADING",
        "filename": "fatigue_model.pkl",
    },
]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_specialist(
    positive_class: str,
    filename: str,
    random_state: int = RANDOM_STATE,
) -> None:
    """Train a binary classifier for positive_class vs. all others and save it."""
    X, y = generate_training_data(random_state=random_state)

    # Binarize labels: positive_class → 1, everything else → 0
    y_binary = (y == positive_class).astype(int)

    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X, y_binary)

    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, filename)
    joblib.dump(clf, save_path)
    print(f"  Specialist model saved → {save_path}  (positive class: {positive_class})")


def train_all_specialists(random_state: int = RANDOM_STATE) -> None:
    print("Training specialist binary classifiers...")
    for spec in SPECIALIST_MODELS:
        train_specialist(
            positive_class=spec["positive_class"],
            filename=spec["filename"],
            random_state=random_state,
        )
    print("All specialist models saved.")


if __name__ == "__main__":
    train_all_specialists()
