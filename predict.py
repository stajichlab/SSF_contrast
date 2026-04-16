#!/usr/bin/env python3
"""
Classify new genomes (whole genomes or metagenome short-read assemblies)
using a trained SSF classifier model.

Usage examples:

  # Classify a single FASTA file:
  python predict.py --input unknown.scaffolds.fa --model-dir models/default

  # Classify all *.fa files in a directory:
  python predict.py --input-dir /path/to/new_genomes/ --model-dir models/default

  # Classify short reads (metagenome fragment bins):
  python predict.py --input metagenome_bin.fa --model-dir models/default --short-reads

  # Output CSV with scores:
  python predict.py --input-dir /path/to/genomes/ --model-dir models/default --out results.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.classifier import load_model, predict


def parse_args():
    p = argparse.ArgumentParser(description="Classify genomes as Subsurface or Terrestrial")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Single FASTA file to classify")
    group.add_argument("--input-dir", help="Directory of FASTA files to classify")

    p.add_argument(
        "--model-dir", required=True,
        help="Directory containing trained model (pipeline.pkl + metadata.json)",
    )
    p.add_argument(
        "--seq-type", choices=["cds", "scaffolds", "auto"], default="auto",
        help="Sequence type. 'auto' detects from filename suffix",
    )
    p.add_argument(
        "--short-reads", action="store_true",
        help="Input sequences are short reads / metagenome bins (skip chunking overhead)",
    )
    p.add_argument("--out", default=None, help="CSV output path (default: print to stdout)")
    p.add_argument(
        "--evo2-model", default=None,
        help="Override Evo-2 model (default: use model saved in metadata)",
    )
    return p.parse_args()


def _detect_seq_type(path: Path) -> str:
    name = path.name
    if "cds" in name or "mrna" in name or "transcript" in name:
        return "cds"
    return "scaffolds"


def embed_input(fasta_path: Path, seq_type: str, evo2_model: str, short_reads: bool) -> np.ndarray:
    from Bio import SeqIO
    from src.embeddings import embed_genome_from_scaffolds, embed_genome_from_cds, embed_sequence

    seqs = [str(r.seq) for r in SeqIO.parse(fasta_path, "fasta")]
    if not seqs:
        raise ValueError(f"No sequences found in {fasta_path}")

    if short_reads:
        # Embed each read independently, then average
        from src.embeddings import _load_evo2
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer = _load_evo2(evo2_model, device)
        from src.embeddings import _embed_single
        vecs = [_embed_single(s[:4096], model, tokenizer, device) for s in seqs[:1000]]
        return np.mean(vecs, axis=0)

    if seq_type == "cds":
        return embed_genome_from_cds(seqs, model_name=evo2_model)
    else:
        return embed_genome_from_scaffolds(seqs, model_name=evo2_model)


def build_annotation_vec_for_path(fasta_path: Path, metadata: dict) -> np.ndarray:
    """
    If an annotations.txt file exists alongside the FASTA, extract annotation features.
    Returns zero vector if not found.
    """
    ann_path = fasta_path.with_suffix("").with_suffix(".annotations.txt")
    clusters_path = fasta_path.with_suffix("").with_suffix(".clusters.txt")

    # Try variations like sample.scaffolds.fa → sample.annotations.txt
    stem = fasta_path.name
    for suffix in (".scaffolds.fa", ".cds-transcripts.fa", ".mrna-transcripts.fa", ".fa"):
        if stem.endswith(suffix):
            base = fasta_path.parent / stem[: -len(suffix)]
            ann_path = base.with_suffix(".annotations.txt")
            clusters_path = base.with_suffix(".clusters.txt")
            break

    n_ann_features = sum(
        1 for n in metadata.get("feature_names", []) if not n.startswith("evo2_")
    )
    if n_ann_features == 0:
        return np.array([], dtype=np.float32)

    from src.data_loader import GenomeRecord
    from src.annotation_features import annotation_feature_vector

    rec = GenomeRecord(
        name=fasta_path.stem,
        label=0,
        label_name="unknown",
        annotations_path=ann_path if ann_path.exists() else None,
        clusters_path=clusters_path if clusters_path.exists() else None,
    )
    vec, _ = annotation_feature_vector(rec)
    return vec


def classify_file(fasta_path: Path, pipeline, metadata: dict, args) -> dict:
    mode = metadata.get("mode", "embedding")
    evo2_model = args.evo2_model or metadata.get("evo2_model", "evo2_1b_base")
    seq_type = args.seq_type
    if seq_type == "auto":
        seq_type = _detect_seq_type(fasta_path)

    label_map = metadata.get("label_map", {"0": "Terrestrial", "1": "Subsurface"})

    feature_parts = []

    if mode in ("embedding", "hybrid"):
        print(f"  Embedding {fasta_path.name} …")
        emb = embed_input(fasta_path, seq_type, evo2_model, args.short_reads)
        feature_parts.append(emb)

    if mode in ("annotation", "hybrid"):
        ann_vec = build_annotation_vec_for_path(fasta_path, metadata)
        feature_parts.append(ann_vec)

    if not feature_parts:
        raise ValueError("No features computed")

    X = np.hstack(feature_parts).reshape(1, -1)
    labels, proba = predict(pipeline, X)

    label_int = int(labels[0])
    p_sub = float(proba[0])
    return {
        "file": str(fasta_path),
        "predicted_class": label_map.get(str(label_int), str(label_int)),
        "p_subsurface": round(p_sub, 4),
        "p_terrestrial": round(1 - p_sub, 4),
        "confidence": round(max(p_sub, 1 - p_sub), 4),
    }


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)

    pipeline, metadata = load_model(model_dir)
    print(f"[predict] Loaded model from {model_dir}")
    print(f"          mode={metadata.get('mode')}  "
          f"evo2={metadata.get('evo2_model')}  "
          f"clf={metadata.get('clf_type')}")

    fasta_files: list[Path] = []
    if args.input:
        fasta_files = [Path(args.input)]
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        fasta_files = sorted(
            p for p in input_dir.iterdir()
            if p.suffix in (".fa", ".fasta", ".fna", ".ffn")
        )

    if not fasta_files:
        print("ERROR: No FASTA files found.")
        sys.exit(1)

    rows = []
    for fasta_path in fasta_files:
        try:
            result = classify_file(fasta_path, pipeline, metadata, args)
            rows.append(result)
            print(
                f"  {result['file']}"
                f"  → {result['predicted_class']}"
                f"  (P_sub={result['p_subsurface']:.3f})"
            )
        except Exception as exc:
            print(f"  ERROR classifying {fasta_path}: {exc}")
            rows.append({"file": str(fasta_path), "error": str(exc)})

    df = pd.DataFrame(rows)
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nResults saved to {args.out}")
    else:
        print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
