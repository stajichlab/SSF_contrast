#!/usr/bin/env python3
"""
SHAP-based explanation of the annotation-feature classifier.

Produces:
  results/shap_beeswarm.png         — per-sample SHAP values, coloured by feature value
  results/shap_bar.png              — mean |SHAP| per feature (global importance)
  results/shap_waterfall_<name>.png — per-genome waterfall for every Subsurface sample
  results/shap_dependence_<feat>.png— SHAP vs raw value for the top-N features
  results/shap_values.csv           — full SHAP matrix
  results/shap_class_means.csv      — mean SHAP per class per feature
  results/logistic_coefficients.csv — LR weights (if logistic model)
  results/permutation_importance.csv

Usage:
  pixi run python explain_annotation.py --model-dir models/annotation --data-dir classify
  pixi run python explain_annotation.py --model-dir models/annotation --top-dependence 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.classifier import load_model
from src.data_loader import discover_genomes, summarize_dataset
from src.annotation_features import build_annotation_matrix


def compute_shap(pipeline, X: np.ndarray):
    """Return (shap_values, X_scaled, explainer_type)."""
    import shap
    from sklearn.linear_model import LogisticRegression

    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]
    X_scaled = scaler.transform(X)

    if isinstance(clf, LogisticRegression):
        explainer = shap.LinearExplainer(
            clf, X_scaled, feature_perturbation="interventional"
        )
        shap_values = explainer.shap_values(X_scaled)
        expected_value = explainer.expected_value
        return shap_values, X_scaled, expected_value, "linear"
    else:
        # Use TreeExplainer-style background sampling for KernelExplainer
        background = shap.sample(X_scaled, min(50, len(X_scaled)), random_state=42)
        explainer = shap.KernelExplainer(
            lambda x: clf.predict_proba(x)[:, 1], background
        )
        shap_values = explainer.shap_values(X_scaled, nsamples=200)
        return shap_values, X_scaled, explainer.expected_value, "kernel"


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_beeswarm(shap_values, X_scaled, feature_names, out_path):
    import shap
    plt.figure(figsize=(10, max(6, len(feature_names) * 0.35)))
    shap.summary_plot(
        shap_values, X_scaled, feature_names=feature_names,
        plot_type="dot", show=False, max_display=len(feature_names),
    )
    plt.title("SHAP beeswarm — positive = Subsurface, negative = Terrestrial")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[explain] Beeswarm plot → {out_path}")


def plot_bar(shap_values, feature_names, out_path):
    import shap
    plt.figure(figsize=(9, max(5, len(feature_names) * 0.32)))
    shap.summary_plot(
        shap_values, feature_names=feature_names,
        plot_type="bar", show=False, max_display=len(feature_names),
    )
    plt.title("Mean |SHAP| — global feature importance")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[explain] Bar plot      → {out_path}")


def plot_waterfall(shap_values, X_scaled, expected_value, feature_names,
                   sample_idx: int, sample_name: str, out_path):
    import shap
    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=expected_value,
        data=X_scaled[sample_idx],
        feature_names=feature_names,
    )
    plt.figure(figsize=(10, max(5, len(feature_names) * 0.28)))
    shap.plots.waterfall(explanation, show=False, max_display=len(feature_names))
    plt.title(f"SHAP waterfall — {sample_name}")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[explain] Waterfall     → {out_path}")


def plot_dependence(shap_values, X_scaled, feature_names, feat_name: str, out_path):
    import shap
    plt.figure(figsize=(7, 5))
    shap.dependence_plot(
        feat_name, shap_values, X_scaled,
        feature_names=feature_names, show=False,
    )
    plt.title(f"SHAP dependence — {feat_name}")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[explain] Dependence    → {out_path}")


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def print_shap_report(shap_values, feature_names, labels, genome_names):
    labels = np.array(labels)
    shap_arr = np.array(shap_values)

    mean_abs = np.abs(shap_arr).mean(axis=0)
    sub_mean = shap_arr[labels == 1].mean(axis=0) if (labels == 1).any() else np.zeros(len(feature_names))
    ter_mean = shap_arr[labels == 0].mean(axis=0) if (labels == 0).any() else np.zeros(len(feature_names))

    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
        "mean_shap_subsurface": sub_mean,
        "mean_shap_terrestrial": ter_mean,
    }).sort_values("mean_abs_shap", ascending=False)

    print("\n" + "=" * 72)
    print("  SHAP FEATURE IMPORTANCE SUMMARY")
    print("  Positive SHAP → pushes toward Subsurface")
    print("  Negative SHAP → pushes toward Terrestrial")
    print("=" * 72)
    print(f"{'Feature':<28} {'|SHAP|':>8}  {'Subsurface':>12}  {'Terrestrial':>12}")
    print("-" * 72)
    for _, row in df.iterrows():
        direction = "↑Sub" if row["mean_shap_subsurface"] > 0 else "↑Ter"
        print(
            f"{row['feature']:<28} {row['mean_abs_shap']:>8.4f}"
            f"  {row['mean_shap_subsurface']:>+12.4f}"
            f"  {row['mean_shap_terrestrial']:>+12.4f}  {direction}"
        )
    print("=" * 72)

    # Per-subsurface-genome breakdown
    sub_indices = np.where(labels == 1)[0]
    if len(sub_indices):
        print("\n  TOP-3 DRIVING FEATURES — SUBSURFACE GENOMES")
        print("-" * 72)
        for idx in sub_indices:
            name = genome_names[idx]
            vals = shap_arr[idx]
            top3 = sorted(zip(feature_names, vals), key=lambda x: abs(x[1]), reverse=True)[:3]
            parts = ",  ".join(f"{f}: {v:+.3f}" for f, v in top3)
            print(f"  {name[:35]:<35}  {parts}")
        print("=" * 72)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Explain annotation classifier with SHAP")
    p.add_argument("--model-dir", default="models/annotation",
                   help="Directory containing pipeline.pkl + metadata.json")
    p.add_argument("--data-dir", default="classify",
                   help="classify/ directory with Subsurface/ and Terrestrial/ subdirs")
    p.add_argument("--results-dir", default="results",
                   help="Output directory for plots and CSVs")
    p.add_argument("--top-dependence", type=int, default=5,
                   help="Number of top features to produce dependence plots for")
    p.add_argument("--n-workers", type=int, default=8,
                   help="Threads for annotation loading")
    return p.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    print(f"[explain] Loading model from {model_dir}")
    pipeline, metadata = load_model(model_dir)
    clf_type = metadata.get("clf_type", "logistic")
    print(f"          classifier type : {clf_type}")
    print(f"          feature mode    : {metadata.get('mode', 'unknown')}")

    # ---- Rebuild annotation features ----
    print(f"\n[explain] Discovering genomes in {args.data_dir} …")
    records = discover_genomes(Path(args.data_dir))
    if not records:
        print(f"ERROR: no genomes found under {args.data_dir}")
        sys.exit(1)
    summarize_dataset(records)

    print("\n[explain] Building annotation feature matrix …")
    X, feature_names, labels = build_annotation_matrix(records, n_workers=args.n_workers)
    genome_names = [r.name for r in records]
    labels = np.array(labels)
    print(f"          {X.shape[0]} samples × {X.shape[1]} features")

    # ---- Logistic coefficients ----
    from sklearn.linear_model import LogisticRegression
    if isinstance(pipeline.named_steps["clf"], LogisticRegression):
        print("\n[explain] Logistic regression coefficients …")
        from src.features import logistic_coefficients, plot_top_features
        coef_df = logistic_coefficients(pipeline, feature_names, top_n=len(feature_names))
        coef_df.to_csv(results_dir / "logistic_coefficients.csv", index=False)
        print(f"\n{'Feature':<28} {'Coefficient':>12}")
        print("-" * 42)
        for _, row in coef_df.iterrows():
            direction = "→ Subsurface" if row["coefficient"] > 0 else "→ Terrestrial"
            print(f"  {row['feature']:<26} {row['coefficient']:>+10.4f}  {direction}")
        plot_top_features(
            coef_df, "coefficient",
            results_dir / "logistic_coefficients.png",
            title="Logistic regression coefficients\n(positive → Subsurface, negative → Terrestrial)",
        )

    # ---- Permutation importance ----
    print("\n[explain] Permutation importance (n_repeats=30) …")
    from src.features import permutation_importance as perm_imp, plot_top_features
    perm_df = perm_imp(pipeline, X, labels, feature_names, n_repeats=30,
                       top_n=len(feature_names))
    perm_df.to_csv(results_dir / "permutation_importance.csv", index=False)
    plot_top_features(
        perm_df, "importance_mean",
        results_dir / "permutation_importance.png",
        title="Permutation importance (balanced accuracy drop)",
    )

    # ---- SHAP values ----
    print("\n[explain] Computing SHAP values …")
    shap_values, X_scaled, expected_value, explainer_type = compute_shap(pipeline, X)
    print(f"          explainer type: {explainer_type}")

    # Save raw SHAP matrix
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.insert(0, "genome", genome_names)
    shap_df.insert(1, "label", ["Subsurface" if l == 1 else "Terrestrial" for l in labels])
    shap_df.to_csv(results_dir / "shap_values.csv", index=False)
    print(f"[explain] SHAP values   → {results_dir / 'shap_values.csv'}")

    # ---- Print text report ----
    summary_df = print_shap_report(shap_values, feature_names, labels, genome_names)
    summary_df.to_csv(results_dir / "shap_class_means.csv", index=False)
    print(f"\n[explain] Class means   → {results_dir / 'shap_class_means.csv'}")

    # ---- Beeswarm + bar plots ----
    plot_beeswarm(shap_values, X_scaled, feature_names, results_dir / "shap_beeswarm.png")
    plot_bar(shap_values, feature_names, results_dir / "shap_bar.png")

    # ---- Waterfall for each Subsurface genome ----
    sub_indices = np.where(labels == 1)[0]
    print(f"\n[explain] Waterfall plots for {len(sub_indices)} Subsurface genome(s) …")
    for idx in sub_indices:
        safe_name = genome_names[idx].replace(" ", "_").replace("/", "-")
        out = results_dir / f"shap_waterfall_{safe_name}.png"
        plot_waterfall(shap_values, X_scaled, expected_value,
                       feature_names, idx, genome_names[idx], out)

    # ---- Dependence plots for top-N features ----
    top_features = summary_df["feature"].head(args.top_dependence).tolist()
    print(f"\n[explain] Dependence plots for top-{args.top_dependence} features …")
    for feat in top_features:
        safe = feat.replace(" ", "_")
        out = results_dir / f"shap_dependence_{safe}.png"
        plot_dependence(shap_values, X_scaled, feature_names, feat, out)

    print(f"\n[explain] Done. All outputs written to {results_dir}/")


if __name__ == "__main__":
    main()
