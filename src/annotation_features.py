"""
Feature engineering from annotation_summary TSV files.

Columns used:
  pfam_domains    pipe-separated PF_ACC:PF_NAME:EVALUE entries; one protein may
                  carry multiple domains, and the same accession may appear more
                  than once (multiple hits along the sequence)
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

  PFAM scalar features (per genome):
  - pfam_annotated          proteins with ≥1 domain
  - total_pfam_domains      total domain instances (counting duplicates)
  - unique_pfam_accessions  distinct PF_ACC values in the genome
  - avg_domains_per_annotated_protein
  - multi_domain_proteins   proteins carrying >1 distinct PF_ACC

  PFAM per-family features (only in build_annotation_matrix, two-pass):
  - pfam_<ACC>_count        proteins carrying that domain (genome-level)
  - pfam_<ACC>_rate         above / gene_count

  All non-rate counts also normalised by gene_count to produce *_rate variants.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_loader import GenomeRecord


CAZYME_CLASSES = ["GH", "GT", "PL", "CE", "AA", "CBM"]
SIGNALP_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Internal parsers
# ---------------------------------------------------------------------------

def _parse_pfam(df: pd.DataFrame) -> dict:
    """
    Parse the pfam_domains column.

    Each row may hold several pipe-delimited 'PF_ACC:PF_NAME:EVALUE' tokens.
    The same accession can appear multiple times within one protein (multiple
    hit regions along the sequence) — we count all instances for domain-count
    stats but deduplicate per protein when counting 'proteins carrying domain X'.

    Returns a dict with:
      pfam_annotated, total_pfam_domains, unique_pfam_accessions,
      avg_domains_per_annotated_protein, multi_domain_proteins,
      protein_domain_sets  — list[set[str]], one entry per annotated protein
      domain_instance_counts — Counter of PF_ACC → total instances
    """
    if "pfam_domains" not in df.columns:
        return {
            "pfam_annotated": 0,
            "total_pfam_domains": 0,
            "unique_pfam_accessions": 0,
            "avg_domains_per_annotated_protein": 0.0,
            "multi_domain_proteins": 0,
            "protein_domain_sets": [],
            "domain_instance_counts": Counter(),
        }

    protein_domain_sets: list[set] = []
    domain_instance_counts: Counter = Counter()
    total_domains = 0

    for val in df["pfam_domains"].dropna():
        val = str(val).strip()
        if not val or val == "nan":
            continue

        seen_in_protein: set[str] = set()
        instances_in_protein = 0
        for token in val.split("|"):
            token = token.strip()
            if not token:
                continue
            acc = token.split(":")[0].strip()
            if not acc:
                continue
            seen_in_protein.add(acc)
            domain_instance_counts[acc] += 1
            instances_in_protein += 1

        if seen_in_protein:
            protein_domain_sets.append(seen_in_protein)
            total_domains += instances_in_protein

    pfam_annotated = len(protein_domain_sets)
    unique_acc = len(domain_instance_counts)
    avg_per_protein = total_domains / pfam_annotated if pfam_annotated else 0.0
    multi_domain = sum(1 for s in protein_domain_sets if len(s) > 1)

    return {
        "pfam_annotated": pfam_annotated,
        "total_pfam_domains": total_domains,
        "unique_pfam_accessions": unique_acc,
        "avg_domains_per_annotated_protein": avg_per_protein,
        "multi_domain_proteins": multi_domain,
        "protein_domain_sets": protein_domain_sets,
        "domain_instance_counts": domain_instance_counts,
    }


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


# ---------------------------------------------------------------------------
# Public: single-genome feature vector (scalar features only)
# ---------------------------------------------------------------------------

def annotation_feature_vector(rec: GenomeRecord) -> Tuple[np.ndarray, List[str]]:
    """
    Build scalar annotation feature vector for one GenomeRecord.

    Per-PFAM-family counts are NOT included here (they require a cross-genome
    vocabulary); use build_annotation_matrix for the full feature set.

    Returns (vector: np.ndarray, feature_names: List[str]).
    """
    ann_df = rec.load_annotation_summary()

    feature_dict: dict = {"gene_count": 0}
    feature_dict.update({cls: 0 for cls in CAZYME_CLASSES})
    feature_dict["total_cazyme"] = 0
    feature_dict.update({"secreted": 0, "membrane": 0, "protease": 0})
    feature_dict.update({
        "pfam_annotated": 0,
        "total_pfam_domains": 0,
        "unique_pfam_accessions": 0,
        "avg_domains_per_annotated_protein": 0.0,
        "multi_domain_proteins": 0,
    })

    if ann_df is not None:
        feature_dict["gene_count"] = len(ann_df)
        feature_dict.update(_count_cazymes(ann_df))
        feature_dict.update(_count_secreted(ann_df))
        feature_dict.update(_count_membrane(ann_df))
        feature_dict.update(_count_proteases(ann_df))

        pfam_stats = _parse_pfam(ann_df)
        for k in ("pfam_annotated", "total_pfam_domains", "unique_pfam_accessions",
                  "avg_domains_per_annotated_protein", "multi_domain_proteins"):
            feature_dict[k] = pfam_stats[k]

    n_genes = max(feature_dict.get("gene_count", 1), 1)
    normalized = dict(feature_dict)
    for k, v in feature_dict.items():
        if k not in ("gene_count", "avg_domains_per_annotated_protein") and isinstance(v, (int, float)):
            normalized[f"{k}_rate"] = v / n_genes

    names = sorted(normalized.keys())
    vec = np.array([normalized[k] for k in names], dtype=np.float32)
    return vec, names


# ---------------------------------------------------------------------------
# Public: full matrix with per-PFAM-family features
# ---------------------------------------------------------------------------

def build_annotation_matrix(
    records: List[GenomeRecord],
    n_workers: int = 8,
    min_genome_freq: int = 2,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Build annotation feature matrix for a list of GenomeRecords.

    Two-pass approach:
      Pass 1 — parse every genome; collect scalar features + per-protein
               domain sets for vocabulary construction.
      Pass 2 — for each PFAM accession appearing in ≥ min_genome_freq genomes,
               add pfam_<ACC>_count (proteins carrying that domain) and
               pfam_<ACC>_rate (/ gene_count).

    Args:
        records:          list of GenomeRecord
        n_workers:        threads for parallel TSV loading
        min_genome_freq:  minimum number of genomes a domain must appear in to
                          become a feature (filters out ultra-rare domains)

    Returns:
        X              np.ndarray (n_records, n_features)
        feature_names  List[str]
        labels         List[int]
    """

    # ---- Pass 1: parse all genomes in parallel -------------------------
    parsed: list[Optional[dict]] = [None] * len(records)

    def _process(idx: int, rec: GenomeRecord):
        ann_df = rec.load_annotation_summary()
        n_genes = 0
        scalar: dict = {}
        pfam_result: dict = {}

        if ann_df is not None:
            n_genes = len(ann_df)
            scalar.update(_count_cazymes(ann_df))
            scalar.update(_count_secreted(ann_df))
            scalar.update(_count_membrane(ann_df))
            scalar.update(_count_proteases(ann_df))
            pfam_result = _parse_pfam(ann_df)
        else:
            scalar = {cls: 0 for cls in CAZYME_CLASSES}
            scalar["total_cazyme"] = 0
            scalar.update({"secreted": 0, "membrane": 0, "protease": 0})
            pfam_result = {
                "pfam_annotated": 0,
                "total_pfam_domains": 0,
                "unique_pfam_accessions": 0,
                "avg_domains_per_annotated_protein": 0.0,
                "multi_domain_proteins": 0,
                "protein_domain_sets": [],
                "domain_instance_counts": Counter(),
            }

        return idx, n_genes, scalar, pfam_result

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process, i, rec): i for i, rec in enumerate(records)}
        for fut in as_completed(futures):
            idx, n_genes, scalar, pfam_result = fut.result()
            parsed[idx] = {"n_genes": n_genes, "scalar": scalar, "pfam": pfam_result}

    # ---- Build PFAM vocabulary -----------------------------------------
    # Count how many genomes each accession appears in (genome-level presence)
    genome_freq: Counter = Counter()
    for entry in parsed:
        for acc in entry["pfam"]["domain_instance_counts"]:
            genome_freq[acc] += 1

    vocab: List[str] = sorted(
        acc for acc, freq in genome_freq.items() if freq >= min_genome_freq
    )
    print(f"[annotation] PFAM vocabulary: {len(genome_freq)} total accessions, "
          f"{len(vocab)} retained (≥{min_genome_freq} genomes)")

    # ---- Pass 2: assemble feature rows ---------------------------------
    scalar_keys = (
        ["gene_count"]
        + [cls for cls in CAZYME_CLASSES]
        + ["total_cazyme", "secreted", "membrane", "protease",
           "pfam_annotated", "total_pfam_domains", "unique_pfam_accessions",
           "avg_domains_per_annotated_protein", "multi_domain_proteins"]
    )

    rows: list[dict] = []
    for entry in parsed:
        n_genes = entry["n_genes"]
        scalar = entry["scalar"]
        pfam = entry["pfam"]

        row: dict = {"gene_count": n_genes}
        row.update(scalar)
        for k in ("pfam_annotated", "total_pfam_domains", "unique_pfam_accessions",
                  "avg_domains_per_annotated_protein", "multi_domain_proteins"):
            row[k] = pfam[k]

        n_norm = max(n_genes, 1)

        # Rate variants for all non-ratio scalars
        for k, v in list(row.items()):
            if k not in ("gene_count", "avg_domains_per_annotated_protein") and isinstance(v, (int, float)):
                row[f"{k}_rate"] = v / n_norm

        # Per-PFAM-family: proteins carrying the domain (deduplicated per protein)
        protein_sets = pfam["protein_domain_sets"]
        for acc in vocab:
            count = sum(1 for s in protein_sets if acc in s)
            row[f"pfam_{acc}_count"] = count
            row[f"pfam_{acc}_rate"] = count / n_norm

        rows.append(row)

    all_names = sorted(rows[0].keys())
    labels = [rec.label for rec in records]
    X = np.array(
        [[row.get(n, 0.0) for n in all_names] for row in rows],
        dtype=np.float32,
    )
    return X, all_names, labels
