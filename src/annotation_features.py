"""
Feature engineering from functional annotations.

Produces a numeric feature vector per genome from its annotations.txt and clusters.txt.
These features can be used standalone or concatenated with Evo-2 embeddings for
a hybrid classifier.

Feature groups:
  - CAZyme counts by class (GH, GT, PL, CE, AA, CBM, …)
  - Secreted / membrane / protease gene counts
  - COG category frequencies
  - antiSMASH cluster type counts
  - BUSCO completeness proxy (count of non-empty BUSCO hits)
  - GO term category counts (biological_process, molecular_function, cellular_component)
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_loader import GenomeRecord


# ---------------------------------------------------------------------------
# CAZyme parsing
# ---------------------------------------------------------------------------

CAZYME_CLASSES = ["GH", "GT", "PL", "CE", "AA", "CBM"]

def _count_cazymes(df: pd.DataFrame) -> dict:
    counts = {cls: 0 for cls in CAZYME_CLASSES}
    counts["total_cazyme"] = 0
    if "CAZyme" not in df.columns:
        return counts
    for val in df["CAZyme"].dropna():
        val = str(val)
        if not val or val in (".", "nan"):
            continue
        counts["total_cazyme"] += 1
        for cls in CAZYME_CLASSES:
            if re.search(rf"\b{cls}\d+", val):
                counts[cls] += 1
    return counts


# ---------------------------------------------------------------------------
# COG categories
# ---------------------------------------------------------------------------

COG_CATEGORIES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def _count_cog(df: pd.DataFrame) -> dict:
    counts = {f"COG_{c}": 0 for c in COG_CATEGORIES}
    if "COG" not in df.columns:
        return counts
    for val in df["COG"].dropna():
        val = str(val)
        if not val or val in (".", "nan"):
            continue
        for c in val:
            key = f"COG_{c}"
            if key in counts:
                counts[key] += 1
    return counts


# ---------------------------------------------------------------------------
# Secretome / special gene classes
# ---------------------------------------------------------------------------

def _count_functional(df: pd.DataFrame) -> dict:
    counts = {"secreted": 0, "membrane": 0, "protease": 0}
    for col, key in [("Secreted", "secreted"), ("Membrane", "membrane"),
                     ("Protease", "protease")]:
        if col in df.columns:
            counts[key] = df[col].apply(
                lambda v: 0 if (pd.isna(v) or str(v).strip() in ("", "0", ".")) else 1
            ).sum()
    return counts


# ---------------------------------------------------------------------------
# BUSCO hits
# ---------------------------------------------------------------------------

def _count_busco(df: pd.DataFrame) -> dict:
    if "BUSCO" not in df.columns:
        return {"busco_hits": 0}
    hits = df["BUSCO"].apply(
        lambda v: 0 if (pd.isna(v) or str(v).strip() in ("", ".")) else 1
    ).sum()
    return {"busco_hits": int(hits)}


# ---------------------------------------------------------------------------
# antiSMASH cluster types
# ---------------------------------------------------------------------------

ANTISMASH_TYPES = [
    "nrps", "pks", "terpene", "indole", "alkaloid", "other",
    "t1pks", "t2pks", "t3pks", "hybrid", "rripp"
]

def _count_clusters(cluster_df: Optional[pd.DataFrame]) -> dict:
    counts = {f"bgc_{t}": 0 for t in ANTISMASH_TYPES}
    counts["bgc_total"] = 0
    if cluster_df is None or "ClusterPred" not in cluster_df.columns:
        return counts
    for val in cluster_df["ClusterPred"].dropna():
        val = str(val).lower()
        if val in (".", "", "flanking", "nan"):
            continue
        counts["bgc_total"] += 1
        for t in ANTISMASH_TYPES:
            if t in val:
                counts[f"bgc_{t}"] += 1
    return counts


# ---------------------------------------------------------------------------
# GO term category counts
# ---------------------------------------------------------------------------

GO_ASPECTS = {
    "biological_process": "go_bp",
    "molecular_function": "go_mf",
    "cellular_component": "go_cc",
}

def _count_go(df: pd.DataFrame) -> dict:
    counts = {v: 0 for v in GO_ASPECTS.values()}
    if "GO Terms" not in df.columns:
        return counts
    for val in df["GO Terms"].dropna():
        val = str(val).lower()
        for aspect, key in GO_ASPECTS.items():
            counts[key] += val.count(aspect)
    return counts


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def annotation_feature_vector(rec: GenomeRecord) -> Tuple[np.ndarray, List[str]]:
    """
    Build annotation-based feature vector for a GenomeRecord.

    Returns:
        (vector: np.ndarray, feature_names: List[str])
    """
    ann_df = rec.load_annotations()
    cluster_df = rec.load_clusters()

    # Always initialise every possible key to 0 so all records produce the same
    # feature names regardless of which annotation files are present.
    feature_dict: dict = {"gene_count": 0}
    feature_dict.update({cls: 0 for cls in CAZYME_CLASSES})
    feature_dict["total_cazyme"] = 0
    feature_dict.update({f"COG_{c}": 0 for c in COG_CATEGORIES})
    feature_dict.update({"secreted": 0, "membrane": 0, "protease": 0})
    feature_dict["busco_hits"] = 0
    feature_dict.update({v: 0 for v in GO_ASPECTS.values()})
    feature_dict.update({f"bgc_{t}": 0 for t in ANTISMASH_TYPES})
    feature_dict["bgc_total"] = 0

    if ann_df is not None:
        feature_dict["gene_count"] = len(ann_df)
        feature_dict.update(_count_cazymes(ann_df))
        feature_dict.update(_count_cog(ann_df))
        feature_dict.update(_count_functional(ann_df))
        feature_dict.update(_count_busco(ann_df))
        feature_dict.update(_count_go(ann_df))

    feature_dict.update(_count_clusters(cluster_df))

    # Normalize raw counts by gene count to get rates
    n_genes = max(feature_dict.get("gene_count", 1), 1)
    normalized = dict(feature_dict)
    for k, v in feature_dict.items():
        if k != "gene_count" and isinstance(v, (int, float)):
            normalized[f"{k}_rate"] = v / n_genes

    names = sorted(normalized.keys())
    vec = np.array([normalized[k] for k in names], dtype=np.float32)
    return vec, names


def build_annotation_matrix(
    records: List[GenomeRecord],
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Build annotation feature matrix for a list of GenomeRecords.

    Returns:
        X: np.ndarray of shape (n_records, n_features)
        feature_names: List[str]
        labels: List[int]
    """
    # First pass: collect all names and per-record dicts
    all_vecs_by_name: list[dict] = []
    all_names_set: set = set()
    labels: list[int] = []

    for rec in records:
        vec, names = annotation_feature_vector(rec)
        row = dict(zip(names, vec))
        all_vecs_by_name.append(row)
        all_names_set.update(names)
        labels.append(rec.label)

    # Second pass: align to sorted union of all names (fills 0 for missing keys)
    all_names = sorted(all_names_set)
    X = np.array(
        [[row.get(n, 0.0) for n in all_names] for row in all_vecs_by_name],
        dtype=np.float32,
    )
    return X, all_names, labels
