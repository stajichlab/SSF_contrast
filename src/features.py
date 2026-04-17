"""
Feature importance and interpretability for the SSF classifier.

Methods:
  1. Logistic regression coefficients (direct, for embedding + annotation features)
  2. Permutation importance (model-agnostic, robust)
  3. SHAP values (optional, requires `pip install shap`)
  4. UMAP visualization of the embedding space
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # headless rendering on cluster


# ---------------------------------------------------------------------------
# Logistic regression coefficients
# ---------------------------------------------------------------------------

def logistic_coefficients(
    pipeline,
    feature_names: List[str],
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Extract signed feature weights from a fitted LogisticRegression pipeline.

    Positive coefficient → pushes toward Subsurface (label=1).
    Negative coefficient → pushes toward Terrestrial (label=0).
    """
    from sklearn.linear_model import LogisticRegression

    clf = pipeline.named_steps["clf"]
    if not isinstance(clf, LogisticRegression):
        raise TypeError("logistic_coefficients requires a LogisticRegression classifier")

    coef = clf.coef_[0]
    df = pd.DataFrame({"feature": feature_names, "coefficient": coef})
    df["abs_coef"] = df["coefficient"].abs()
    df = df.sort_values("abs_coef", ascending=False).head(top_n).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------

def permutation_importance(
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 30,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Model-agnostic permutation importance using balanced accuracy.

    Works with any pipeline (logistic, MLP, …).
    """
    from sklearn.inspection import permutation_importance as sk_perm_imp
    from sklearn.metrics import balanced_accuracy_score, make_scorer

    scorer = make_scorer(balanced_accuracy_score)
    result = sk_perm_imp(
        pipeline, X, y, n_repeats=n_repeats, random_state=42, scoring=scorer, n_jobs=-1
    )
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    df = df.sort_values("importance_mean", ascending=False).head(top_n).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# SHAP values
# ---------------------------------------------------------------------------

def shap_summary(
    pipeline,
    X: np.ndarray,
    feature_names: List[str],
    output_path: Optional[str | Path] = None,
) -> np.ndarray:
    """
    Compute SHAP values for the classifier.
    Saves a beeswarm plot if output_path is provided.

    Returns shap_values array of shape (n_samples, n_features).
    """
    import shap

    # Transform X through the scaler step
    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]
    X_scaled = scaler.transform(X)

    explainer = shap.LinearExplainer(clf, X_scaled, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_scaled)

    if output_path:
        plt.figure()
        shap.summary_plot(
            shap_values,
            X_scaled,
            feature_names=feature_names,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150)
        plt.close()
        print(f"[features] SHAP summary saved to {output_path}")

    return shap_values


# ---------------------------------------------------------------------------
# UMAP embedding visualization
# ---------------------------------------------------------------------------

def plot_umap(
    X: np.ndarray,
    labels: List[int],
    names: List[str],
    output_path: str | Path,
    n_neighbors: int = 10,
    min_dist: float = 0.3,
    title: str = "Evo-2 genome embeddings (UMAP)",
) -> None:
    """
    Reduce embedding matrix to 2-D with UMAP and save two plots:
      1. output_path              — subsurface points labeled only
      2. output_path stem + _all_labeled suffix — all points labeled
    """
    import umap

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    coords = reducer.fit_transform(X)

    label_arr = np.array(labels)
    colors = {0: "#1f77b4", 1: "#d62728"}  # blue=Terrestrial, red=Subsurface
    label_names = {0: "Terrestrial", 1: "Subsurface"}

    def _scatter(ax):
        for lbl in sorted(set(labels)):
            mask = label_arr == lbl
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=colors[lbl], label=label_names[lbl], alpha=0.7, s=60,
            )
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.legend()

    # --- Plot 1: subsurface labels only ---
    fig, ax = plt.subplots(figsize=(9, 7))
    _scatter(ax)
    for i, (name, lbl) in enumerate(zip(names, labels)):
        if lbl == 1:
            ax.annotate(
                name, (coords[i, 0], coords[i, 1]),
                fontsize=7, xytext=(4, 4), textcoords="offset points",
            )
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    print(f"[features] UMAP plot saved to {output_path}")

    # --- Plot 2: all points labeled ---
    output_path = Path(output_path)
    labeled_path = output_path.with_stem(output_path.stem + "_all_labeled")
    fig, ax = plt.subplots(figsize=(14, 11))
    _scatter(ax)
    for i, (name, lbl) in enumerate(zip(names, labels)):
        ax.annotate(
            name, (coords[i, 0], coords[i, 1]),
            fontsize=5, xytext=(3, 3), textcoords="offset points",
            color=colors[lbl],
        )
    ax.set_title(title + " (all labeled)")
    plt.tight_layout()
    plt.savefig(str(labeled_path), dpi=150)
    plt.close()
    print(f"[features] UMAP plot (all labeled) saved to {labeled_path}")


# ---------------------------------------------------------------------------
# Bar-plot for top features
# ---------------------------------------------------------------------------

def plot_top_features(
    df: pd.DataFrame,
    score_col: str,
    output_path: str | Path,
    title: str = "Top discriminating features",
) -> None:
    """Plot a horizontal bar chart of the top features dataframe."""
    df_plot = df.head(20).copy()
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in df_plot[score_col]]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df_plot["feature"][::-1], df_plot[score_col][::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(score_col)
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    print(f"[features] Feature plot saved to {output_path}")
