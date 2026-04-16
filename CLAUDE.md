# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Build an [Evo-2](https://github.com/ARC-AGI/evo2)-based classifier that distinguishes fungal genomes from two ecological niches:
- **Subsurface** (cave/deep-earth fungi) — 5 genomes in `classify/Subsurface/`
- **Terrestrial** (surface-soil fungi) — 191 genomes in `classify/Terresterial/`

The classifier must support both whole-genome inputs and short metagenome reads.  Trained parameters persist to `models/` so future runs skip embedding extraction.

## Environment setup

```bash
pixi install          # uses pixi.toml; install pixi from https://pixi.sh if needed
pixi shell            # or prefix commands with: pixi run python ...
```

Evo-2 model weights download automatically on first use (~7 GB for `evo2_1b_base`, GPU strongly recommended).

## Common commands

### Train — annotation features only (no GPU needed, fast baseline)
```bash
pixi run python train.py --mode annotation --data-dir classify --model-dir models/annotation
```

### Train — Evo-2 embeddings from CDS transcripts
```bash
pixi run python train.py --mode embedding --seq-type cds --data-dir classify \
    --model-dir models/evo2_cds --embedding-cache models/embeddings
```

### Train — hybrid (embeddings + functional annotations)
```bash
pixi run python train.py --mode hybrid --seq-type cds --data-dir classify \
    --model-dir models/hybrid
```

### Classify a new genome (whole genome)
```bash
pixi run python predict.py --input unknown.scaffolds.fa --model-dir models/hybrid
```

### Classify metagenome short reads
```bash
pixi run python predict.py --input bin.fa --model-dir models/hybrid --short-reads
```

### Classify all FASTA files in a directory
```bash
pixi run python predict.py --input-dir /path/to/new_genomes/ --model-dir models/hybrid --out results.csv
```

## Architecture

```
src/
  data_loader.py          Discovers genomes; GenomeRecord dataclass with lazy loaders
  embeddings.py           Evo-2 embedding extraction; tiles long scaffolds into windows;
                          caches per-genome .npy files in models/embeddings/
  annotation_features.py  CAZyme counts, COG frequencies, secretome, BGC types →
                          numeric feature vector from annotations.txt + clusters.txt
  classifier.py           sklearn Pipeline (StandardScaler + LogisticRegression/MLP);
                          save/load via pickle; handles class imbalance with balanced weights
  features.py             Feature importance: logistic coefficients, permutation importance,
                          SHAP values, UMAP visualization

train.py                  End-to-end training with cross-validation; saves model + plots
predict.py                Inference on new FASTA files (whole genome or short reads)
models/                   Saved pipelines (pipeline.pkl) and metadata (metadata.json)
results/                  Plots and CSVs produced by train.py
classify/
  Subsurface/             5 cave fungi: *.scaffolds.fa, *.cds-transcripts.fa,
                          *.annotations.txt, *.clusters.txt
  Terresterial/           191 surface fungi: *.scaffolds.fa, *.cds-transcripts.fa
```

## Key design decisions

- **Class imbalance (5 vs 191)**: all classifiers use `class_weight='balanced'`; CV folds are clamped to minority class size; evaluation uses balanced accuracy, F1, and ROC-AUC.
- **Three feature modes**: `annotation` (no GPU, fast), `embedding` (Evo-2 only), `hybrid` (both concatenated).
- **Embedding caching**: computed embeddings are stored as `.npy` files keyed by `{genome_name}.{seq_type}.npy`; reuse across training runs with `--embedding-cache`.
- **Genome tiling**: scaffolds are chunked into overlapping `chunk_size`-bp windows (default 8192 bp, 50% overlap); per-window embeddings are mean-pooled to a single genome vector.
- **Annotation features**: CAZyme class counts (GH/GT/PL/CE/AA/CBM), COG categories, secreted/membrane/protease rates, antiSMASH BGC types, BUSCO hit density, GO term category counts — all normalized by gene count to produce per-genome rates.
- **Evo-2 API**: package `evo2` exposes `Evo2(model_name)` with a `.model` attribute and `.tokenizer`; forward pass with `return_embeddings=True` returns `(logits, embeddings)`.

## Data file types

| Suffix | Content |
|---|---|
| `.scaffolds.fa` | Genome assembly (nuclear scaffolds) |
| `.cds-transcripts.fa` / `.mrna-transcripts.fa` | CDS/mRNA sequences (primary sequences for embedding) |
| `.annotations.txt` | TSV: GeneID, Product, PFAM, COG, GO Terms, CAZyme, antiSMASH, Secreted, … |
| `.clusters.txt` | antiSMASH biosynthetic gene cluster table (secondary metabolites) |
| `.fcs_gx-taxonomy.tsv` | FCS-GX contamination taxonomy (some Terrestrial only, not used in classifier) |

Note: `classify/Terresterial/` is a typo in the directory name — do not rename it.
