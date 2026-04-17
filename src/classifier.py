"""
Classifier that predicts Subsurface vs Terrestrial from feature vectors.

Supports three input modes:
  - "embedding"    : Evo-2 embeddings only
  - "annotation"   : Functional annotation features only
  - "hybrid"       : Concatenated embeddings + annotation features

The model itself is a lightweight MLP or logistic regression on top of the
pre-computed features.  All parameters are saved to / loaded from a model dir
so future runs can skip embedding extraction.

Handles the heavy class imbalance (5 Subsurface vs 191 Terrestrial) via
class_weight='balanced' and stratified cross-validation.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_pipeline(model_type: str = "logistic") -> Pipeline:
    """
    Return a sklearn Pipeline with scaler + classifier.

    Args:
        model_type: "logistic" | "mlp"
    """
    if model_type == "logistic":
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            C=0.1,
            solver="lbfgs",
            random_state=42,
        )
    elif model_type == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(256, 64),
            activation="relu",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "logistic",
    cv_folds: int = 5,
) -> Tuple[Pipeline, dict]:
    """
    Train a classifier with stratified k-fold cross-validation.

    Because Subsurface class has very few samples the default is 5-fold; with
    only 5 Subsurface genomes this is leave-one-out territory — adjust cv_folds
    accordingly.

    Returns:
        (fitted_pipeline, cv_metrics_dict)
    """
    # Clamp folds to min(cv_folds, min_class_count)
    min_class = int(np.bincount(y.astype(int)).min())
    cv_folds = min(cv_folds, min_class)
    if cv_folds < 2:
        cv_folds = 2
        print(f"WARNING: only {min_class} samples in minority class; using 2-fold CV")

    pipeline = build_pipeline(model_type)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    scoring = ["balanced_accuracy", "f1", "roc_auc", "average_precision"]
    cv_results = cross_validate(
        pipeline, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1
    )

    metrics = {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
        for k, v in cv_results.items()
    }

    # Fit on full dataset for the saved model
    pipeline.fit(X, y)
    return pipeline, metrics


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(
    pipeline: Pipeline,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (predicted_labels, subsurface_probabilities).
    """
    labels = pipeline.predict(X)
    proba = pipeline.predict_proba(X)[:, 1]  # P(Subsurface)
    return labels, proba


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(
    pipeline: Pipeline,
    model_dir: str | Path,
    metadata: Optional[dict] = None,
) -> None:
    """Save pipeline + metadata to model_dir."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "pipeline.pkl", "wb") as fh:
        pickle.dump(pipeline, fh)
    if metadata:
        with open(model_dir / "metadata.json", "w") as fh:
            json.dump(metadata, fh, indent=2)
    print(f"[classifier] Model saved to {model_dir}")


def load_model(model_dir: str | Path) -> Tuple[Pipeline, dict]:
    """Load pipeline + metadata from model_dir."""
    model_dir = Path(model_dir)
    with open(model_dir / "pipeline.pkl", "rb") as fh:
        pipeline = pickle.load(fh)
    metadata = {}
    meta_file = model_dir / "metadata.json"
    if meta_file.exists():
        with open(meta_file) as fh:
            metadata = json.load(fh)
    return pipeline, metadata


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> None:
    from .data_loader import LABEL_NAMES
    target_names = [LABEL_NAMES[0], LABEL_NAMES[1]]
    print(classification_report(y_true, y_pred, target_names=target_names,
                                zero_division=0))
    try:
        auc = roc_auc_score(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        print(f"ROC-AUC: {auc:.3f}   Average Precision: {ap:.3f}")
    except ValueError:
        pass
