# SSF Contrast — Subsurface vs Terrestrial Fungal Genome Classifier

Classifies fungal genomes as **Subsurface** (cave/deep-earth) or **Terrestrial**
(surface-soil) using [Evo-2](https://github.com/ARC-AGI/evo2) genomic language model
embeddings and/or functional annotation features.  Trained models persist to disk so
future runs — including short-read metagenome data — can skip the embedding step.

## Data

Genome files live under `classify/`:

| Directory | Genomes | File types |
|---|---|---|
| `classify/Subsurface/` | 5 cave fungi | `.scaffolds.fa`, `.cds-transcripts.fa`, `.annotations.txt`, `.clusters.txt` |
| `classify/Terresterial/` | 193 surface fungi | `.scaffolds.fa`, `.cds-transcripts.fa` |

> The `Terresterial/` spelling is intentional in the directory name — do not rename it.

## Installation

This project uses [pixi](https://pixi.sh) for environment management.

### Install pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### Install project dependencies

```bash
cd SSF_contrast
pixi install
```

All subsequent commands should be prefixed with `pixi run` to use the managed
environment, or run inside a shell activated with `pixi shell`.

## Usage

### 1. Annotation-only baseline (no GPU required)

Builds a classifier from CAZyme profiles, COG categories, secretome content, and
biosynthetic gene cluster types. Only the 5 Subsurface genomes have full annotation
files; Terrestrial genomes receive zeros for annotation features.

```bash
pixi run python train.py \
    --mode annotation \
    --data-dir classify \
    --model-dir models/annotation
```

### 2. Evo-2 embeddings (GPU recommended)

Evo-2 model weights (~7 GB for the 1B variant) download automatically on first use.

```bash
pixi run python train.py \
    --mode embedding \
    --seq-type cds \
    --data-dir classify \
    --model-dir models/evo2_cds \
    --embedding-cache models/embeddings
```

Use `--seq-type scaffolds` to embed whole-genome scaffold sequences instead of CDS
transcripts (slower, higher GPU memory).

### 3. Hybrid mode (embeddings + annotations)

```bash
pixi run python train.py \
    --mode hybrid \
    --seq-type cds \
    --data-dir classify \
    --model-dir models/hybrid \
    --embedding-cache models/embeddings
```

### 4. Classify new genomes

```bash
# Single FASTA
pixi run python predict.py \
    --input unknown_genome.scaffolds.fa \
    --model-dir models/hybrid

# Directory of FASTA files, save results to CSV
pixi run python predict.py \
    --input-dir /path/to/new_genomes/ \
    --model-dir models/hybrid \
    --out results/predictions.csv
```

### 5. Classify metagenome short reads

```bash
pixi run python predict.py \
    --input metagenome_bin.fa \
    --model-dir models/hybrid \
    --short-reads
```

## Training options

| Flag | Default | Description |
|---|---|---|
| `--mode` | `annotation` | `annotation` / `embedding` / `hybrid` |
| `--seq-type` | `cds` | Sequences to embed: `cds` or `scaffolds` |
| `--evo2-model` | `evo2_1b_base` | Evo-2 variant (`evo2_1b_base`, `evo2_7b_base`, `evo2_40b_base`) |
| `--clf-type` | `logistic` | Classifier head: `logistic` or `mlp` |
| `--cv-folds` | `5` | Cross-validation folds (auto-clamped to minority class size) |
| `--embedding-cache` | `models/embeddings` | Directory for cached `.npy` embedding files |
| `--overwrite-embeddings` | off | Re-compute embeddings even if cache exists |
| `--results-dir` | `results` | Output directory for plots and CSVs |

## Outputs

After training, `models/<name>/` contains:

- `pipeline.pkl` — fitted sklearn pipeline (scaler + classifier)
- `metadata.json` — feature names, hyperparameters, CV metrics, label map

`results/` contains (where applicable):

- `logistic_coefficients.csv` / `top_features_logistic.png`
- `permutation_importance.csv` / `top_features_permutation.png`
- `umap_embeddings.png`

## Architecture

```
src/
  data_loader.py          GenomeRecord dataclass; discovers genomes from classify/
  embeddings.py           Evo-2 tiling, pooling, and .npy caching
  annotation_features.py  105-feature functional annotation vectors
  classifier.py           sklearn pipeline; balanced class weights; save/load
  features.py             Coefficients, permutation importance, SHAP, UMAP
train.py                  Training pipeline
predict.py                Inference on new FASTA files
```

## Notes on class imbalance

With 5 Subsurface and 193 Terrestrial genomes (1:38 ratio), all classifiers use
`class_weight='balanced'`.  Cross-validation is stratified and folds are clamped to
the minority class count.  Prefer **balanced accuracy**, **F1 (Subsurface class)**, and
**ROC-AUC** over raw accuracy when interpreting results.
