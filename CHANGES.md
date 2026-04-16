# Changelog

## 2026-04-16 — Bug fixes: Evo-2 tokenizer and embedding API

### Fixed

- `src/embeddings.py`: `CharLevelTokenizer` is not callable like a HuggingFace tokenizer;
  replaced `tokenizer(sequence, return_tensors="pt")["input_ids"]` with
  `torch.tensor([tokenizer.tokenize(sequence)], dtype=torch.long)`
- `src/embeddings.py`: `StripedHyena.forward()` does not accept `return_embeddings`;
  the correct entry point is `Evo2.__call__()` which requires `layer_names` alongside
  `return_embeddings=True` and returns a `dict` of tensors keyed by layer name;
  updated `_embed_single` to call `model(input_ids, return_embeddings=True, layer_names=["norm"])`
  and read `emb_dict["norm"]` (final RMSNorm output, shape `(batch, seq_len, hidden_dim)`)
- `src/embeddings.py`: removed manual `_evo2_model.model.to(device)` call in `_load_evo2`;
  Evo2/vortex handles device placement internally and the manual move broke multi-GPU configs

## 2026-04-15 — Reorganised data layout

### Changed

- Data directory renamed from `classfiy/` to `classify/`; niche folder `Terresterial/`
  corrected to `Terrestrial/`
- Files within each niche are now split into type-specific subdirectories:
  `cds/`, `dna/`, `annotation/`, `BGC/`
- `src/data_loader.py`: `discover_genomes()` rewrote to walk the new subdir layout via
  `_SUBDIR_ROLE` mapping (`cds/` → `cds_path`, `dna/` → `scaffolds_path`,
  `annotation/` → `annotations_path`, `BGC/` → `clusters_path`); removed flat-file
  suffix matching
- `LABEL_MAP` updated from `{"Terresterial": 0, …}` to `{"Terrestrial": 0, …}`
- `train.py` default `--data-dir` updated to `classify`
- `README.md` and `CLAUDE.md` updated to reflect new layout

## 2026-04-15 — GPU/CUDA installation notes

### Added

- `README.md`: documented that `transformer-engine[pytorch]` must be installed
  separately after `pixi install` using `pip3 install --no-build-isolation` so the
  build links against the host CUDA libraries; annotated which usage modes require this

## 2026-04-15 — Bug fix: annotation feature matrix shape mismatch

### Fixed

- `src/annotation_features.py`: `build_annotation_matrix()` raised
  `ValueError: all input array dimensions must match` because Terrestrial genomes
  (no `.annotations.txt`) produced ~25 features while annotated Subsurface genomes
  produced 102, making `np.vstack()` fail
- Fix 1: `annotation_feature_vector()` now initialises every possible feature key to 0
  at the top of the function regardless of which files are present, so all records
  always return the same complete key set
- Fix 2: `build_annotation_matrix()` switched to a two-pass approach — first collect
  per-record dicts, then align to the sorted union of all names with zero-fill for
  missing keys — as a structural safeguard against any future key-set divergence

### Added

- `README.md` — user-facing guide (installation, usage, options table, architecture)
- `CHANGES.md` — this file

## 2026-04-15 — Initial codebase

### Added

**`src/data_loader.py`**
- `GenomeRecord` dataclass: holds paths to all file types per genome with lazy loaders
  (`load_scaffolds()`, `load_cds()`, `load_annotations()`, `load_clusters()`)
- `discover_genomes(data_dir)`: walks niche subdirectories and type-specific subdirs
  (`cds/`, `dna/`, `annotation/`, `BGC/`), returning one `GenomeRecord` per genome
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
