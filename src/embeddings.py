"""
Evo-2 embedding extraction for genomic sequences.

Evo-2 (Arc Institute) handles sequences up to 1 Mb but GPU memory limits practical
chunk sizes.  For whole genomes we tile the scaffolds into overlapping windows,
extract per-token embeddings, mean-pool across the window, then mean-pool across all
windows to produce a single fixed-length genome embedding.

For short reads (metagenome fragments ≤ a few kb) the whole sequence is embedded
in one shot.

Install Evo-2:
    pip install evo2
    # Model weights download automatically on first use (~7 GB for 1B, ~28 GB for 7B)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Evo-2 lazy import — allows the rest of the package to import without GPU
# ---------------------------------------------------------------------------
_evo2_model = None
_evo2_tokenizer = None


def _load_evo2(model_name: str = "evo2_1b_base", device: Optional[str] = None):
    """Load Evo-2 model and tokenizer, caching globally."""
    global _evo2_model, _evo2_tokenizer
    if _evo2_model is None:
        from evo2 import Evo2  # noqa: F401  (requires `pip install evo2`)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[embeddings] Loading Evo-2 model '{model_name}' on {device} …")
        _evo2_model = Evo2(model_name)
        _evo2_model.model.eval()
        _evo2_tokenizer = _evo2_model.tokenizer
        print("[embeddings] Model ready.")
    return _evo2_model, _evo2_tokenizer


# ---------------------------------------------------------------------------
# Core embedding helpers
# ---------------------------------------------------------------------------

def _embed_single(sequence: str, model, tokenizer, device: str) -> np.ndarray:
    """
    Embed a single DNA sequence string.
    Returns a 1-D numpy array of shape (hidden_dim,).
    """
    sequence = sequence.upper().replace("N", "A")  # simple N-masking
    token_ids = tokenizer.tokenize(sequence)
    input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    # Evo2.forward() uses hooks; layer_names picks which layer to capture.
    # "norm" is the final RMSNorm before the unembedding head.
    _, emb_dict = model(input_ids, return_embeddings=True, layer_names=["norm"])
    # emb_dict["norm"] shape: (batch, seq_len, hidden_dim) — mean-pool over seq_len
    vec = emb_dict["norm"][0].mean(dim=0).float().cpu().numpy()
    return vec


def embed_sequence(
    sequence: str,
    chunk_size: int = 8192,
    stride: int = 4096,
    model_name: str = "evo2_1b_base",
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Embed a (potentially long) DNA sequence by tiling into overlapping windows.
    Returns mean-pooled embedding of shape (hidden_dim,).

    Args:
        sequence:   DNA string (A/C/G/T/N).
        chunk_size: Window length in bp.
        stride:     Step size; use chunk_size//2 for 50 % overlap.
        model_name: Evo-2 model variant.
        device:     Torch device string; auto-detected if None.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = _load_evo2(model_name, device)

    seq = sequence.upper()
    n = len(seq)

    if n <= chunk_size:
        return _embed_single(seq, model, tokenizer, device)

    chunk_vecs = []
    for start in range(0, n, stride):
        chunk = seq[start : start + chunk_size]
        if len(chunk) < 64:  # skip tiny trailing chunks
            break
        chunk_vecs.append(_embed_single(chunk, model, tokenizer, device))

    return np.mean(chunk_vecs, axis=0)


def embed_genome_from_scaffolds(
    scaffolds: List[str],
    max_scaffolds: int = 50,
    chunk_size: int = 8192,
    stride: int = 4096,
    model_name: str = "evo2_1b_base",
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Embed a genome represented as a list of scaffold strings.
    The top-N longest scaffolds are used (for speed); their embeddings are averaged.

    Args:
        scaffolds:      List of scaffold sequences.
        max_scaffolds:  Maximum number of scaffolds to process.
        chunk_size:     Embedding window size in bp.
        stride:         Stride between windows.
    Returns:
        1-D numpy array of shape (hidden_dim,).
    """
    if not scaffolds:
        raise ValueError("No scaffolds provided")

    # Sort by length descending, take top N
    scaffolds_sorted = sorted(scaffolds, key=len, reverse=True)[:max_scaffolds]

    scaffold_vecs = []
    for i, seq in enumerate(scaffolds_sorted):
        print(f"  scaffold {i + 1}/{len(scaffolds_sorted)} (len={len(seq):,})")
        vec = embed_sequence(seq, chunk_size=chunk_size, stride=stride,
                             model_name=model_name, device=device)
        scaffold_vecs.append(vec)

    return np.mean(scaffold_vecs, axis=0)


def embed_genome_from_cds(
    cds_seqs: List[str],
    max_transcripts: int = 500,
    model_name: str = "evo2_1b_base",
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Embed a genome using its CDS transcripts (shorter, more tractable than scaffolds).
    Transcripts are embedded individually; mean-pool across transcripts.
    """
    if not cds_seqs:
        raise ValueError("No CDS sequences provided")

    seqs = sorted(cds_seqs, key=len, reverse=True)[:max_transcripts]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = _load_evo2(model_name, device)

    vecs = []
    for i, seq in enumerate(seqs):
        if (i + 1) % 50 == 0:
            print(f"  CDS {i + 1}/{len(seqs)}")
        vecs.append(_embed_single(seq, model, tokenizer, device))

    return np.mean(vecs, axis=0)


# ---------------------------------------------------------------------------
# Batch embedding with persistence
# ---------------------------------------------------------------------------

def embed_and_cache(
    records,
    cache_dir: str | Path,
    seq_type: str = "cds",   # "cds" | "scaffolds"
    model_name: str = "evo2_1b_base",
    device: Optional[str] = None,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    """
    Compute and cache Evo-2 embeddings for a list of GenomeRecord objects.

    Args:
        records:    List of GenomeRecord from data_loader.discover_genomes().
        cache_dir:  Directory for .npy embedding files.
        seq_type:   Which sequence type to embed ("cds" or "scaffolds").
        overwrite:  Re-compute even if cache file exists.

    Returns:
        Dict mapping record.name → embedding array.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    embeddings: dict[str, np.ndarray] = {}

    for rec in records:
        cache_file = cache_dir / f"{rec.name}.{seq_type}.npy"

        if cache_file.exists() and not overwrite:
            embeddings[rec.name] = np.load(cache_file)
            continue

        print(f"[embed] {rec.label_name} | {rec.name}")
        try:
            if seq_type == "cds":
                seqs = rec.load_cds()
                if not seqs:
                    print(f"  WARNING: no CDS sequences for {rec.name}, skipping")
                    continue
                vec = embed_genome_from_cds(seqs, model_name=model_name, device=device)
            elif seq_type == "scaffolds":
                seqs = rec.load_scaffolds()
                if not seqs:
                    print(f"  WARNING: no scaffolds for {rec.name}, skipping")
                    continue
                vec = embed_genome_from_scaffolds(
                    seqs, model_name=model_name, device=device
                )
            else:
                raise ValueError(f"Unknown seq_type: {seq_type}")

            np.save(cache_file, vec)
            embeddings[rec.name] = vec

        except Exception as exc:
            print(f"  ERROR embedding {rec.name}: {exc}")

    return embeddings
