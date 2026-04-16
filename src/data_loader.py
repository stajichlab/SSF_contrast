"""
Data loading utilities for SSF_contrast classifier.

Expects directory layout:
  <data_dir>/
    Subsurface/   *.scaffolds.fa, *.cds-transcripts.fa, *.annotations.txt, *.clusters.txt
    Terresterial/ *.scaffolds.fa, *.cds-transcripts.fa

Labels: 0 = Terrestrial, 1 = Subsurface
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from Bio import SeqIO


LABEL_MAP = {"Terresterial": 0, "Subsurface": 1}
LABEL_NAMES = {0: "Terrestrial", 1: "Subsurface"}


@dataclass
class GenomeRecord:
    name: str          # Species/strain identifier derived from filename stem
    label: int         # 0=Terrestrial, 1=Subsurface
    label_name: str
    scaffolds_path: Optional[Path] = None
    cds_path: Optional[Path] = None
    annotations_path: Optional[Path] = None
    clusters_path: Optional[Path] = None

    def load_scaffolds(self) -> List[str]:
        """Return list of scaffold sequences as plain strings."""
        if self.scaffolds_path is None or not self.scaffolds_path.exists():
            return []
        return [str(r.seq) for r in SeqIO.parse(self.scaffolds_path, "fasta")]

    def load_cds(self) -> List[str]:
        """Return list of CDS/mRNA transcript sequences."""
        if self.cds_path is None or not self.cds_path.exists():
            return []
        return [str(r.seq) for r in SeqIO.parse(self.cds_path, "fasta")]

    def load_annotations(self) -> Optional[pd.DataFrame]:
        """Return annotations DataFrame (tab-separated, first row is header)."""
        if self.annotations_path is None or not self.annotations_path.exists():
            return None
        return pd.read_csv(self.annotations_path, sep="\t", low_memory=False)

    def load_clusters(self) -> Optional[pd.DataFrame]:
        """Return biosynthetic gene cluster table (tab-separated, skip comment lines)."""
        if self.clusters_path is None or not self.clusters_path.exists():
            return None
        rows = []
        header = None
        with open(self.clusters_path) as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith("#Cluster_"):
                    continue
                if line.startswith("#GeneID"):
                    header = line.lstrip("#").split("\t")
                    continue
                if header is not None and line:
                    rows.append(line.split("\t"))
        if header is None or not rows:
            return None
        return pd.DataFrame(rows, columns=header)


def _stem(path: Path) -> str:
    """Extract species/strain name from filename by stripping known suffixes."""
    name = path.name
    for suffix in (
        ".scaffolds.fa",
        ".cds-transcripts.fa",
        ".mrna-transcripts.fa",
        ".annotations.txt",
        ".clusters.txt",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def discover_genomes(data_dir: str | Path) -> List[GenomeRecord]:
    """
    Walk <data_dir>/Subsurface and <data_dir>/Terresterial and return one
    GenomeRecord per genome.
    """
    data_dir = Path(data_dir)
    records: Dict[Tuple[int, str], GenomeRecord] = {}

    for folder, label in LABEL_MAP.items():
        subdir = data_dir / folder
        if not subdir.exists():
            continue
        for fpath in sorted(subdir.iterdir()):
            if fpath.suffix not in (".fa", ".txt", ".tsv"):
                continue
            name = _stem(fpath)
            key = (label, name)
            if key not in records:
                records[key] = GenomeRecord(
                    name=name, label=label, label_name=LABEL_NAMES[label]
                )
            rec = records[key]
            fname = fpath.name
            if fname.endswith(".scaffolds.fa"):
                rec.scaffolds_path = fpath
            elif fname.endswith((".cds-transcripts.fa", ".mrna-transcripts.fa")):
                rec.cds_path = fpath
            elif fname.endswith(".annotations.txt"):
                rec.annotations_path = fpath
            elif fname.endswith(".clusters.txt"):
                rec.clusters_path = fpath

    return list(records.values())


def summarize_dataset(records: List[GenomeRecord]) -> None:
    subs = sum(1 for r in records if r.label == 1)
    terr = sum(1 for r in records if r.label == 0)
    print(f"Dataset: {len(records)} genomes  (Subsurface={subs}, Terrestrial={terr})")
    print(
        f"  Class imbalance ratio: 1:{terr / max(subs, 1):.1f}  "
        "(use class_weight='balanced' in classifiers)"
    )
