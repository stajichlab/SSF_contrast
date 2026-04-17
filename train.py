#!/usr/bin/env python3
"""
Training pipeline for the Subsurface vs Terrestrial fungal genome classifier.

Usage examples:

  # Annotation-only (no GPU needed, fast):
  python train.py --mode annotation --data-dir classify --model-dir models/annotation

  # Evo-2 embeddings from CDS transcripts:
  python train.py --mode embedding --seq-type cds --data-dir classify --model-dir models/evo2_cds

  # Hybrid (embedding + annotation):
  python train.py --mode hybrid --seq-type cds --data-dir classify --model-dir models/hybrid

  # Use scaffold sequences instead of CDS:
  python train.py --mode embedding --seq-type scaffolds --data-dir classify --model-dir models/evo2_scaffolds

  # Specify Evo-2 model size (default: evo2_1b_base):
  python train.py --mode embedding --evo2-model evo2_7b_base ...
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow importing from src/ without installing as package
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import discover_genomes, summarize_dataset
from src.annotation_features import build_annotation_matrix
from src.classifier import train, save_model, print_metrics, predict


def parse_args():
    p = argparse.ArgumentParser(description="Train SSF genome classifier")
    p.add_argument(
        "--data-dir", default="classify",
        help="Directory containing Subsurface/ and Terresterial/ subdirs (default: classify)",
    )
    p.add_argument(
        "--model-dir", default="models/default",
        help="Directory to save trained model and artifacts",
    )
    p.add_argument(
        "--mode", choices=["annotation", "embedding", "hybrid"], default="annotation",
        help="Feature mode: annotation-only, embedding-only, or hybrid",
    )
    p.add_argument(
        "--seq-type", choices=["cds", "scaffolds"], default="cds",
        help="Which sequences to embed (only used when mode includes 'embedding')",
    )
    p.add_argument(
        "--evo2-model", default="evo2_1b_base",
        help="Evo-2 model variant (evo2_1b_base | evo2_7b_base | evo2_40b_base)",
    )
    p.add_argument(
        "--clf-type", choices=["logistic", "mlp"], default="logistic",
        help="Classifier architecture",
    )
    p.add_argument(
        "--cv-folds", type=int, default=5,
        help="Cross-validation folds (clamped to minority class size)",
    )
    p.add_argument(
        "--embedding-cache", default="models/embeddings",
        help="Directory to cache .npy embedding files",
    )
    p.add_argument(
        "--overwrite-embeddings", action="store_true",
        help="Re-compute embeddings even if cache files exist",
    )
    p.add_argument(
        "--results-dir", default="results",
        help="Directory for plots and reports",
    )
    p.add_argument(
        "--n-workers", type=int, default=8,
        help="Threads for parallel annotation TSV loading (default: 8)",
    )
    return p.parse_args()


def build_feature_matrix(args, records):
    """Assemble X, y, feature_names based on --mode."""
    labels = [r.label for r in records]
    y = np.array(labels)

    if args.mode == "annotation":
        X, feature_names, _ = build_annotation_matrix(records, n_workers=args.n_workers)
        return X, y, feature_names

    # Load / compute Evo-2 embeddings
    from src.embeddings import embed_and_cache
    embedding_dir = Path(args.embedding_cache) / args.evo2_model / args.seq_type
    emb_dict = embed_and_cache(
        records,
        cache_dir=embedding_dir,
        seq_type=args.seq_type,
        model_name=args.evo2_model,
        overwrite=args.overwrite_embeddings,
    )

    # Keep only records that have embeddings
    valid_records = [r for r in records if r.name in emb_dict]
    if len(valid_records) < len(records):
        missing = len(records) - len(valid_records)
        print(f"WARNING: {missing} records skipped (no embedding computed)")

    emb_matrix = np.vstack([emb_dict[r.name] for r in valid_records])
    valid_labels = np.array([r.label for r in valid_records])
    emb_feature_names = [f"evo2_{i}" for i in range(emb_matrix.shape[1])]

    if args.mode == "embedding":
        return emb_matrix, valid_labels, emb_feature_names

    # Hybrid: concatenate annotation features
    ann_X, ann_names, _ = build_annotation_matrix(valid_records, n_workers=args.n_workers)
    X = np.hstack([emb_matrix, ann_X])
    feature_names = emb_feature_names + ann_names
    return X, valid_labels, feature_names


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Discover data ----
    records = discover_genomes(data_dir)
    if not records:
        print(f"ERROR: No genomes found under {data_dir}")
        sys.exit(1)
    summarize_dataset(records)

    # ---- Build feature matrix ----
    print(f"\n[train] Building features (mode={args.mode}) …")
    X, y, feature_names = build_feature_matrix(args, records)
    print(f"  Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")

    # ---- Train ----
    print(f"\n[train] Training {args.clf_type} classifier …")
    pipeline, cv_metrics = train(X, y, model_type=args.clf_type, cv_folds=args.cv_folds)

    print("\n[train] Cross-validation results:")
    for metric, vals in sorted(cv_metrics.items()):
        if metric.startswith("test_"):
            name = metric[5:]
            print(f"  {name:30s}  {vals['mean']:.3f} ± {vals['std']:.3f}")

    # ---- Save model ----
    metadata = {
        "mode": args.mode,
        "seq_type": args.seq_type,
        "evo2_model": args.evo2_model,
        "clf_type": args.clf_type,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_samples": int(X.shape[0]),
        "cv_metrics": cv_metrics,
        "label_map": {"0": "Terrestrial", "1": "Subsurface"},
    }
    save_model(pipeline, model_dir, metadata)

    # ---- Full-dataset evaluation ----
    print("\n[train] Full-dataset evaluation (optimistic, for diagnostics only):")
    y_pred, y_proba = predict(pipeline, X)
    print_metrics(y, y_pred, y_proba)

    # ---- Feature importance ----
    print("\n[train] Computing feature importance …")
    try:
        from src.features import (
            logistic_coefficients,
            permutation_importance,
            plot_top_features,
        )

        if args.clf_type == "logistic":
            coef_df = logistic_coefficients(pipeline, feature_names)
            coef_df.to_csv(results_dir / "logistic_coefficients.csv", index=False)
            plot_top_features(
                coef_df, "coefficient",
                results_dir / "top_features_logistic.png",
                title="Top features by logistic regression coefficient\n"
                      "(positive → Subsurface, negative → Terrestrial)",
            )

        perm_df = permutation_importance(pipeline, X, y, feature_names)
        perm_df.to_csv(results_dir / "permutation_importance.csv", index=False)
        plot_top_features(
            perm_df, "importance_mean",
            results_dir / "top_features_permutation.png",
            title="Top features by permutation importance",
        )

    except Exception as exc:
        print(f"  WARNING: feature importance failed: {exc}")

    # ---- UMAP (embedding and hybrid modes) ----
    if args.mode in ("embedding", "hybrid"):
        try:
            from src.features import plot_umap
            names_list = [r.name for r in records if r.name in {
                rec.name for rec in records
            }]
            plot_umap(
                X[:, : (X.shape[1] if args.mode == "embedding" else
                         X.shape[1] - len(feature_names) + X.shape[1] // 2)],
                list(y),
                [r.name for r in records][:len(y)],
                results_dir / "umap_embeddings.png",
            )
        except Exception as exc:
            print(f"  WARNING: UMAP failed: {exc}")

    print(f"\nDone. Model: {model_dir}  Results: {results_dir}")


if __name__ == "__main__":
    main()
