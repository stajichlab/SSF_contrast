# Changelog

## 2026-04-15 — Initial codebase

### Added

**`src/data_loader.py`**
- `GenomeRecord` dataclass: holds paths to all file types per genome with lazy loaders
  (`load_scaffolds()`, `load_cds()`, `load_annotations()`, `load_clusters()`)
- `discover_genomes(data_dir)`: walks `Subsurface/` and `Terresterial/` subdirectories,
  matches files by suffix, and returns one `GenomeRecord` per genome
- `summarize_dataset()`: prints class counts and imbalance ratio

**`src/annotation_features.py`**
- Extracts a 105-element numeric feature vector per genome from `.annotations.txt`
  and `.clusters.txt`; genomes without annotation files receive zeros for all fields
- Feature groups: CAZyme class counts (GH/GT/PL/CE/AA/CBM), COG category frequencies,
  secreted/membrane/protease gene rates, antiSMASH BGC type counts, BUSCO hit density,
  GO term category counts — all raw counts plus gene-count-normalized rates
- `build_annotation_matrix()`: two-pass construction that aligns all records to the union
  of feature names, preventing shape mismatches when some genomes lack annotation files

**`src/embeddings.py`**
- `embed_sequence()`: tiles a long DNA string into overlapping windows (default 8 192 bp,
  50 % stride) and mean-pools Evo-2 hidden states across windows
- `embed_genome_from_scaffolds()`: selects the top-N longest scaffolds, embeds each,
  and averages to a single genome vector
- `embed_genome_from_cds()`: embeds up to 500 CDS transcripts and averages; faster than
  scaffold mode for large genomes
- `embed_and_cache()`: batch-embeds a list of `GenomeRecord` objects and writes per-genome
  `.npy` files; skips records already cached

**`src/classifier.py`**
- `build_pipeline()`: returns a `StandardScaler → LogisticRegression` or
  `StandardScaler → MLPClassifier` sklearn pipeline
- `train()`: stratified k-fold cross-validation with `class_weight='balanced'`;
  CV folds are automatically clamped to minority-class size; reports balanced accuracy,
  F1, ROC-AUC, and average precision
- `save_model()` / `load_model()`: pickle the fitted pipeline alongside a JSON metadata
  file that records feature names, hyperparameters, and CV metrics

**`src/features.py`**
- `logistic_coefficients()`: extracts signed feature weights (positive → Subsurface)
- `permutation_importance()`: model-agnostic importance via balanced-accuracy drop
- `shap_summary()`: SHAP beeswarm plot for logistic regression (requires `shap`)
- `plot_umap()`: 2-D UMAP projection of the embedding matrix coloured by niche label,
  with Subsurface genomes individually labelled
- `plot_top_features()`: horizontal bar chart of the top-N discriminating features

**`train.py`**
- End-to-end training script; supports `--mode annotation | embedding | hybrid`
- Saves trained pipeline + metadata to `--model-dir`, writes importance CSVs and plots
  to `--results-dir`, caches embeddings to `--embedding-cache`

**`predict.py`**
- Classifies new FASTA files (whole genomes or metagenome bins) using a saved model
- `--short-reads` flag skips chunking and embeds each read independently
- Accepts a single file (`--input`) or a directory (`--input-dir`); outputs a CSV

**`environment.yml`** — conda environment spec (superseded by `pixi.toml`)

**`pixi.toml`** — pixi workspace definition (pre-existing; dependencies verified against
the code)
