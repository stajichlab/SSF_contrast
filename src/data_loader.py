"""
Data loading utilities for SSF_contrast classifier.

Expects directory layout:
  <data_dir>/
    Subsurface/
      cds/                  *.cds-transcripts.fa
      dna/                  *.scaffolds.fa
      annotation_summary/   *.annotation_summary.tsv
    Terrestrial/
      cds/                  *.cds-transcripts.fa
      dna/                  *.scaffolds.fa
      annotation_summary/   *.annotation_summary.tsv

Labels: 0 = Terrestrial, 1 = Subsurface
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from Bio import SeqIO


LABEL_MAP = {"Terrestrial": 0, "Subsurface": 1}
LABEL_NAMES = {0: "Terrestrial", 1: "Subsurface"}

# Maps the type subdirectory name to the GenomeRecord field it populates
_SUBDIR_ROLE = {
    "cds":                "cds_path",
    "dna":                "scaffolds_path",
    "annotation_summary": "annotation_summary_path",
}


@dataclass
class GenomeRecord:
    name: str          # Species/strain identifier derived from filename stem
    label: int         # 0=Terrestrial, 1=Subsurface
    label_name: str
    scaffolds_path: Optional[Path] = None
    cds_path: Optional[Path] = None
    annotation_summary_path: Optional[Path] = None

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

    def load_annotation_summary(self) -> Optional[pd.DataFrame]:
        """Return annotation_summary DataFrame with pfam, signalp, merops, tmhmm, cazy columns."""
        if self.annotation_summary_path is None or not self.annotation_summary_path.exists():
            return None
        return pd.read_csv(self.annotation_summary_path, sep="\t", low_memory=False, compression="infer")


def _stem(path: Path) -> str:
    """Extract species/strain name from filename by stripping known suffixes."""
    name = path.name
    for suffix in (
        ".scaffolds.fa",
        ".cds-transcripts.fa",
        ".mrna-transcripts.fa",
        ".annotation_summary.tsv.gz",
        ".annotation_summary.tsv",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def discover_genomes(data_dir: str | Path) -> List[GenomeRecord]:
    """
    Walk <data_dir>/Subsurface and <data_dir>/Terrestrial, descending into
    type-specific subdirectories (cds/, dna/, annotation/, BGC/), and return
    one GenomeRecord per genome.
    """
    data_dir = Path(data_dir)
    records: Dict[Tuple[int, str], GenomeRecord] = {}

    for niche, label in LABEL_MAP.items():
        niche_dir = data_dir / niche
        if not niche_dir.exists():
            continue

        for subdir_name, field_name in _SUBDIR_ROLE.items():
            subdir = niche_dir / subdir_name
            if not subdir.exists():
                continue
            for fpath in sorted(subdir.iterdir()):
                if fpath.suffix not in (".fa", ".txt", ".tsv", ".gz"):
                    continue
                name = _stem(fpath)
                key = (label, name)
                if key not in records:
                    records[key] = GenomeRecord(
                        name=name, label=label, label_name=LABEL_NAMES[label]
                    )
                setattr(records[key], field_name, fpath)

    return list(records.values())


def summarize_dataset(records: List[GenomeRecord]) -> None:
    subs = sum(1 for r in records if r.label == 1)
    terr = sum(1 for r in records if r.label == 0)
    print(f"Dataset: {len(records)} genomes  (Subsurface={subs}, Terrestrial={terr})")
    print(
        f"  Class imbalance ratio: 1:{terr / max(subs, 1):.1f}  "
        "(use class_weight='balanced' in classifiers)"
    )
