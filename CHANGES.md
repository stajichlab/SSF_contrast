# Changelog

## 2026-04-16 — SHAP explanation script and richer PFAM annotation features

### Added

- `explain_annotation.py`: standalone script that loads a trained annotation model,
  rebuilds the feature matrix, and produces a full SHAP-based interpretability report:
  - `results/shap_beeswarm.png` — per-sample SHAP values coloured by feature value
  - `results/shap_bar.png` — mean |SHAP| global feature ranking
  - `results/shap_waterfall_<name>.png` — per-genome waterfall for each Subsurface sample
  - `results/shap_dependence_<feat>.png` — SHAP vs raw value for top-N features
  - `results/shap_values.csv` — full SHAP matrix (genome × feature)
  - `results/shap_class_means.csv` — mean SHAP per class per feature
  - `results/logistic_coefficients.csv/.png` — LR weights (if logistic model)
  - `results/permutation_importance.csv/.png`
  - Formatted text table printed to stdout: mean |SHAP|, class-directional sign, and
    top-3 driving features per Subsurface genome
  - Supports logistic regression (LinearExplainer) and MLP (KernelExplainer)
  - Usage: `pixi run python explain_annotation.py --model-dir models/annotation`

- `src/annotation_features._parse_pfam()`: replaces the old `_count_pfam()` with a
  full parser for the pipe-delimited `PF_ACC:PF_NAME:EVALUE` format; correctly handles
  multiple domains per protein and repeated accessions (multiple hit regions); returns
  five new scalar features per genome:
  - `total_pfam_domains` — total domain instances (counting duplicates)
  - `unique_pfam_accessions` — distinct PF_ACC values in the genome
  - `avg_domains_per_annotated_protein` — mean domain load per annotated protein
  - `multi_domain_proteins` — proteins carrying >1 distinct domain family
  - `pfam_annotated` retained (proteins with ≥1 domain)

- `src/annotation_features.build_annotation_matrix()`: extended to a two-pass build
  that adds per-PFAM-family features across all genomes:
  - Pass 1: parse every genome in parallel; collect scalar stats and per-protein domain sets
  - Vocabulary step: retain accessions appearing in ≥`min_genome_freq` genomes (default 2);
    6 366 accessions found in the current dataset, 5 976 retained
  - Pass 2: for each vocabulary accession, add `pfam_<ACC>_count` (proteins carrying that
    domain, deduplicated per protein) and `pfam_<ACC>_rate` (/ gene_count)
  - Full feature matrix grows from 23 scalars to ~11 982 features (30 scalar + 11 952
    per-family); `min_genome_freq` parameter controls vocabulary size

### Changed

- `src/annotation_features.annotation_feature_vector()`: updated to use `_parse_pfam()`
  and include the four new PFAM scalar features; per-family features are not included
  here (they require a cross-genome vocabulary — use `build_annotation_matrix()`)

## 2026-04-16 — Multithreading and bug fix

### Added

- `src/annotation_features.build_annotation_matrix()`: parallel TSV loading via
  `ThreadPoolExecutor`; genome reads are I/O-bound and independent, so wall time
  drops roughly proportionally to thread count up to ~8 threads; controlled by the
  new `n_workers` parameter (default 8)
- `train.py`: `--n-workers N` flag (default 8) sets the thread count for annotation
  loading; passed through to both annotation-only and hybrid modes
- `src/classifier.cross_validate()`: `n_jobs=-1` so CV folds run in parallel across
  all available CPUs via sklearn's joblib backend
- `src/features.permutation_importance()`: `n_jobs=-1` so the 30 permutation repeats
  run in parallel

### Fixed

- `src/classifier.print_metrics()`: `from data_loader import LABEL_NAMES` raised
  `ModuleNotFoundError`; corrected to relative import `from .data_loader import LABEL_NAMES`

## 2026-04-16 — Annotation format migration to annotation_summary TSV

### Changed

- Data layout: `annotation/` and `BGC/` subdirectories replaced by
  `annotation_summary/` containing `*.annotation_summary.tsv[.gz]` files with columns
  `protein_id`, `pfam_domains`, `signalp_{start,end,prob}`, `merops_{id,pct_id,evalue}`,
  `tmhmm_{pred_hel,exp_aa,topology}`, `cazy_{family,EC,substrate}`
- `src/data_loader.py`: `GenomeRecord` fields `annotations_path` and `clusters_path`
  replaced by `annotation_summary_path`; `load_annotations()` / `load_clusters()`
  replaced by `load_annotation_summary()`; `_SUBDIR_ROLE` updated accordingly;
  `_stem()` strips `.annotation_summary.tsv` and `.annotation_summary.tsv.gz`;
  `discover_genomes()` accepts `.gz` suffix; `pd.read_csv` uses `compression="infer"`
  so gzip files are transparently decompressed
- `src/annotation_features.py`: fully rewritten for the new TSV format; feature set
  reduced from 105 COG/BUSCO/GO/BGC features to 23 focused features — CAZyme class
  counts (GH/GT/PL/CE/AA/CBM), secreted (SignalP prob > 0.5), membrane (TMHMM
  helices > 0), protease (MEROPS hit), PFAM-annotated gene count, plus gene-count-
  normalised rates for each

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
