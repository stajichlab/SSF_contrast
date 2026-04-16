"""
Feature engineering from annotation_summary TSV files.

Columns used:
  pfam_domains    pipe-separated PF_ID:NAME:EVALUE entries
  signalp_prob    signal-peptide probability (secreted if > 0.5)
  merops_id       MEROPS protease family ID (non-empty = protease)
  tmhmm_pred_hel  number of predicted TM helices (non-empty = membrane protein)
  cazy_family     CAZy family string, e.g. "GH5_12(15-548)"

Feature groups produced:
  - gene_count
  - CAZyme class counts (GH, GT, PL, CE, AA, CBM) + total_cazyme
  - secreted count (signalp_prob > 0.5)
  - membrane count (tmhmm_pred_hel > 0)
  - protease count (merops_id non-empty)
  - pfam_annotated count (any pfam domain present)
  - All counts normalised by gene_count to produce *_rate variants
"""

from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np
import pandas as pd

from .data_loader import GenomeRecord


CAZYME_CLASSES = ["GH", "GT", "PL", "CE", "AA", "CBM"]
SIGNALP_THRESHOLD = 0.5


def _count_cazymes(df: pd.DataFrame) -> dict:
    counts = {cls: 0 for cls in CAZYME_CLASSES}
    counts["total_cazyme"] = 0
    if "cazy_family" not in df.columns:
        return counts
    for val in df["cazy_family"].dropna():
        val = str(val).strip()
        if not val or val == "nan":
            continue
        counts["total_cazyme"] += 1
        for cls in CAZYME_CLASSES:
            if re.search(rf"\b{cls}\d", val):
                counts[cls] += 1
    return counts


def _count_secreted(df: pd.DataFrame) -> dict:
    if "signalp_prob" not in df.columns:
        return {"secreted": 0}
    n = pd.to_numeric(df["signalp_prob"], errors="coerce")
    return {"secreted": int((n > SIGNALP_THRESHOLD).sum())}


def _count_membrane(df: pd.DataFrame) -> dict:
    if "tmhmm_pred_hel" not in df.columns:
        return {"membrane": 0}
    n = pd.to_numeric(df["tmhmm_pred_hel"], errors="coerce").fillna(0)
    return {"membrane": int((n > 0).sum())}


def _count_proteases(df: pd.DataFrame) -> dict:
    if "merops_id" not in df.columns:
        return {"protease": 0}
    has_hit = df["merops_id"].apply(
        lambda v: bool(v and str(v).strip() not in ("", "nan"))
    )
    return {"protease": int(has_hit.sum())}


def _count_pfam(df: pd.DataFrame) -> dict:
    if "pfam_domains" not in df.columns:
        return {"pfam_annotated": 0}
    has_pfam = df["pfam_domains"].apply(
        lambda v: bool(v and str(v).strip() not in ("", "nan"))
    )
    return {"pfam_annotated": int(has_pfam.sum())}


def annotation_feature_vector(rec: GenomeRecord) -> Tuple[np.ndarray, List[str]]:
    """
    Build annotation-based feature vector for a GenomeRecord.

    Returns:
        (vector: np.ndarray, feature_names: List[str])
    """
    ann_df = rec.load_annotation_summary()

    feature_dict: dict = {"gene_count": 0}
    feature_dict.update({cls: 0 for cls in CAZYME_CLASSES})
    feature_dict["total_cazyme"] = 0
    feature_dict.update({"secreted": 0, "membrane": 0, "protease": 0})
    feature_dict["pfam_annotated"] = 0

    if ann_df is not None:
        feature_dict["gene_count"] = len(ann_df)
        feature_dict.update(_count_cazymes(ann_df))
        feature_dict.update(_count_secreted(ann_df))
        feature_dict.update(_count_membrane(ann_df))
        feature_dict.update(_count_proteases(ann_df))
        feature_dict.update(_count_pfam(ann_df))

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
    all_vecs_by_name: list[dict] = []
    all_names_set: set = set()
    labels: list[int] = []

    for rec in records:
        vec, names = annotation_feature_vector(rec)
        row = dict(zip(names, vec))
        all_vecs_by_name.append(row)
        all_names_set.update(names)
        labels.append(rec.label)

    all_names = sorted(all_names_set)
    X = np.array(
        [[row.get(n, 0.0) for n in all_names] for row in all_vecs_by_name],
        dtype=np.float32,
    )
    return X, all_names, labels
