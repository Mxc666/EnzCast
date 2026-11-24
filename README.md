# EnzCast & IsoKin

Official method repository for **KinBench**, **EnzCast**, and **IsoKin**: sequence-clustered enzyme kinetic benchmarking, task-adaptive multimodal probabilistic kinetic prediction, and isoform-resolved stoichiometric Bayesian kinetic inference on transcriptomic cohorts.

- **Web server**: https://enzcast.med.sustech.edu.cn
- **GitHub repository (code & minimal examples)**: https://github.com/Mxc666/EnzCast
- **Hugging Face deposit (checkpoints, KinBench splits, structures & precomputed graphs, Recon3D priors)**: https://huggingface.co/Mxc666/EnzCast

## GitHub vs. Hugging Face Asset Mapping

To keep the code repository lightweight while providing complete training, prediction, and cohort-fitting assets, files are distributed across GitHub and Hugging Face and share a single parent directory layout:

```text
EnzCast-Release/                     # Common workspace root (--project-root)
├── Code/                            # [GitHub] Source code, CLI entrypoints, tests, and tiny smoke examples
│   ├── enzcast/                     # EnzCast architecture, dataset loaders, graph featurizer, train/predict/eval CLIs
│   │   ├── model.py                 # EnzCastModel (ESM-2 LoRA + VN-EGNN + GIN-Transformer + Dual-Pathway Heads)
│   │   ├── dataset.py               # Dataset loaders, organism/EC vocabularies, Gaussian denormalization
│   │   ├── featurize.py             # C-alpha GearNet protein graph & RDKit molecular graph builders + CLI
│   │   ├── preprocess_recon3d_embeddings.py
│   │   ├── train.py                 # Multi-task and single-task training CLI (Beta-NLL + EMA + consistency)
│   │   ├── predict.py               # Task-adaptive & single-checkpoint inference CLI
│   │   ├── evaluate_kinbench.py     # KinBench point & probabilistic uncertainty evaluation CLI
│   │   └── assets/esm2_t33_650M_UR50D/ # Lightweight ESM-2 architecture & tokenizer config (MIT License)
│   ├── kinbench/                    # KinBench sequence-clustered split builder & verifier
│   │   ├── build_splits.py          # MMseqs2 (50% identity, 80% coverage) cluster split builder
│   │   └── verify_splits.py         # Schema, record count, and cluster-disjointness verifier
│   ├── isokin/                      # Installable enzcast-isokin package (Recon3D + GPR + Pyro SVI)
│   │   ├── pyproject.toml
│   │   ├── enzcast_isokin/          # Stoichiometric network builder, GPR AST/soft-min, prior assembly, SVI
│   │   ├── docs/                    # Tutorials, workflow templates, and parameter reference
│   │   └── tests/                   # Unit and integration test suite
│   └── examples/                    # Self-contained human enzyme sample CSV and PDB, sample .h5ad, and verification script
├── Model/                           # [Hugging Face] Official EnzCast weights, vocabularies, and calibration parameters
│   ├── benchmark/                   # Held-out KinBench evaluation suite (3 checkpoints + metadata + val scaling)
│   ├── fulldata_priors/             # New-pair prediction & downstream prior suite (3 checkpoints + metadata)
│   └── esm2_backbone_config/        # ESM-2 architecture & tokenizer files
└── Dataset/                         # [Hugging Face] KinBench benchmark splits/structures/graphs & IsoKin Recon3D priors
    ├── kinbench/                    # Official train/val/test CSVs, cluster map, 3D PDB archive, and .pt graph tarballs
    └── isokin/                      # Recon3D stoichiometry, GPR rules, enzyme-substrate scaffold, and EnzCast priors
```

## Installation & Asset Extraction

### 1. Create Environment & Install `enzcast-isokin`

```bash
mkdir -p EnzCast-Release
cd EnzCast-Release
git clone https://github.com/Mxc666/EnzCast.git Code
cd Code
conda env create -f environment.yml
conda activate enzcast-isokin
python -m pip install -e isokin
```

### 2. Download & Unpack Hugging Face Assets

Place `Model/` and `Dataset/` from `https://huggingface.co/Mxc666/EnzCast` alongside `Code/`, unpack the KinBench archives from `Dataset/kinbench`, and then return to `Code/` before running training, evaluation, or prediction commands:

```bash
# From Code/, fetch the model and dataset directories into the shared parent.
python -m pip install huggingface_hub
hf download Mxc666/EnzCast --include "Model/*" "Dataset/*" --local-dir ..
cd ../Dataset/kinbench

# Extract precomputed PyTorch Geometric graph caches (used directly by train.py / evaluate_kinbench.py / predict.py)
mkdir -p features/protein_graphs features/substrate_graphs
tar -xf features/protein_graphs.tar -C features/protein_graphs
tar -xf features/substrate_graphs.tar -C features/substrate_graphs

# Optional: extract standardized 3D protein coordinate files to structures/pdb/{UniProt_ID}.pdb
mkdir -p structures/pdb
tar -xzf structures/protein_structures_pdb.tar.gz --strip-components=1 -C structures/pdb

# Return to Code/ before executing the workflow commands below
cd ../../Code
```

## KinBench Structure & Graph Feature Mapping

Each row in `kinbench_{train,val,test}.csv` (and any user prediction CSV) resolves sequence, 3D enzyme structure, and substrate structure as follows:

| CSV Column | Coordinate / Chemical Input | Precomputed PyTorch Geometric Feature File | Description |
|---|---|---|---|
| `UniProt_ID` | `structures/pdb/{UniProt_ID}.pdb` | `features/protein_graphs/{UniProt_ID}.pt` | KinBench looks up both the PDB file and precomputed C-alpha graph by `UniProt_ID` (`25,119` unique accessions, each capped at `<= 1,024` C-alpha residues). Structural provenance is recorded in the `structure_origin` column of `kinbench_structure_manifest.csv` (`AlphaFold_DB_v4_cropped`: `16,594`; `ESMFold_predicted_cropped`: `8,525`). Because multiple records sharing a `UniProt_ID` can carry variant, isoform, or truncated sequences, the structure C-alpha count (`ca_residue_count`) does not need to equal `Model_Input_Sequence` length (`8,715` KinBench records differ in length), and EnzCast pools sequence and 3D graph representations independently prior to multimodal fusion. |
| `SMILES` | Canonical SMILES string in CSV | `features/substrate_graphs/{MD5(SMILES)}.pt` | 42-D atom / 6-D bond RDKit molecular graph indexed by the lowercase hex MD5 digest of `SMILES` (`hashlib.md5(smiles.encode("utf-8")).hexdigest()`), paired with a 2,048-bit radius-3 Morgan fingerprint. |
| `Model_Input_Sequence` | Amino acid string in CSV | Tokenized on the fly | Record-specific input sequence (`<= 1,024` aa) passed to the ESM-2 + LoRA sequence encoder. |

### Generating Features for New Enzymes and Substrates

For novel enzyme–substrate pairs not in KinBench:
1. Obtain a monomeric `.pdb` structure for the enzyme (`<= 1,024` C-alpha residues; exact residue-count equality with `Model_Input_Sequence` is not required) from AlphaFold DB (`https://alphafold.ebi.ac.uk/files/AF-{UniProt_ID}-F1-model_v4.pdb`), RCSB PDB (`https://files.rcsb.org/download/{PDB_ID}.pdb`), or structure prediction (ESMFold / AlphaFold2), saved as `{UniProt_ID}.pdb`.
2. Either pass `--pdb-dir /path/to/pdbs` directly to `enzcast/predict.py` (which builds protein C-alpha GearNet graphs and RDKit substrate graphs on the fly), or precompute reusable `.pt` graph caches in batch:

```bash
python enzcast/featurize.py proteins \
  --pdb-dir examples/structures \
  --output-dir /tmp/custom_protein_graphs

python enzcast/featurize.py substrates \
  --input-csv examples/kinbench_sample.csv \
  --smiles-col SMILES \
  --output-dir /tmp/custom_substrate_graphs
```

---

## Workflow 1: Train and Evaluate EnzCast on KinBench

> **Checkpoint Policy**:
> - `Model/benchmark/` (`--suite benchmark`) is trained strictly on `kinbench_train.csv` and MUST be used for held-out evaluation on `kinbench_val.csv` and `kinbench_test.csv`.
> - `Model/fulldata_priors/` (`--suite fulldata`) is trained on merged KinBench data and MUST be used for predicting new enzyme–substrate pairs and generating downstream priors for IsoKin.

### Train EnzCast

Full KinBench training (downloads base `facebook/esm2_t33_650M_UR50D` weights on first run, or accepts a local ESM-2 directory):

```bash
# Multi-task model (jointly trains km, kcat, kcat_km, ki with Beta-NLL + EMA + thermodynamic consistency)
python enzcast/train.py \
  --mode multitask \
  --data-mode benchmark \
  --split-dir ../Dataset/kinbench \
  --protein-graph-dir ../Dataset/kinbench/features/protein_graphs \
  --substrate-graph-dir ../Dataset/kinbench/features/substrate_graphs \
  --output-dir ./trained_models/benchmark_multitask

# Dedicated single-task Km or Ki models
python enzcast/train.py \
  --mode singletask --task km --data-mode benchmark \
  --split-dir ../Dataset/kinbench \
  --protein-graph-dir ../Dataset/kinbench/features/protein_graphs \
  --substrate-graph-dir ../Dataset/kinbench/features/substrate_graphs \
  --output-dir ./trained_models/benchmark_km
```

Fast offline smoke test on bundled sample graphs (runs 2 optimizer steps and writes reloadable `.pt` + `_metadata.json`):

```bash
python enzcast/train.py \
  --mode multitask \
  --train-csv examples/kinbench_sample.csv \
  --val-csv examples/kinbench_sample.csv \
  --protein-graph-dir examples/features/protein_graphs \
  --substrate-graph-dir examples/features/substrate_graphs \
  --esm-model enzcast/assets/esm2_t33_650M_UR50D \
  --pretrain-epochs 0 --stage1-epochs 1 --max-steps 2 --grad-accum-steps 1 --batch-size 4 \
  --output-dir /tmp/enzcast_smoke_ckpt
```

### Verify KinBench Splits & Evaluate Benchmark Checkpoints

```bash
python kinbench/verify_splits.py --kinbench-dir ../Dataset/kinbench

python enzcast/evaluate_kinbench.py \
  --split-csv ../Dataset/kinbench/kinbench_test.csv \
  --protein-graph-dir ../Dataset/kinbench/features/protein_graphs \
  --substrate-graph-dir ../Dataset/kinbench/features/substrate_graphs \
  --model-dir ../Model/benchmark \
  --variance-scaling-json ../Model/benchmark/variance_scaling_val.json \
  --output-dir /tmp/kinbench_eval
```

---

## Workflow 2: Predict Kinetic Parameters for New Enzyme–Substrate Pairs

Use `--suite fulldata` (`Model/fulldata_priors/`) for new biochemical queries. Task-adaptive routing automatically dispatches `km` to `enzcast_fulldata_km.pt`, `ki` to `enzcast_fulldata_ki.pt`, and `kcat` + `kcat_km` to `enzcast_fulldata_multitask.pt`:

```bash
# Option A: Using precomputed .pt graph directories
python enzcast/predict.py \
  --suite fulldata \
  --input-csv examples/kinbench_sample.csv \
  --protein-graph-dir examples/features/protein_graphs \
  --substrate-graph-dir examples/features/substrate_graphs \
  --output-csv /tmp/enzcast_fulldata_predictions.csv

# Option B: On-the-fly featurization directly from raw .pdb structures + SMILES
python enzcast/predict.py \
  --suite fulldata \
  --input-csv examples/kinbench_sample.csv \
  --pdb-dir examples/structures \
  --output-csv /tmp/enzcast_from_pdb_predictions.csv
```

To run inference with a newly trained single checkpoint from `train.py`:

```bash
python enzcast/predict.py \
  --input-csv examples/kinbench_sample.csv \
  --checkpoint /tmp/enzcast_smoke_ckpt/enzcast_multitask.pt \
  --metadata /tmp/enzcast_smoke_ckpt/enzcast_multitask_metadata.json \
  --model-key multitask \
  --protein-graph-dir examples/features/protein_graphs \
  --substrate-graph-dir examples/features/substrate_graphs \
  --output-csv /tmp/enzcast_reloaded_predictions.csv
```

---

## Workflow 3: Fit IsoKin Kinetic Posteriors on User Transcriptomic Cohorts

IsoKin couples EnzCast priors (`Dataset/isokin/priors/`) with the Recon3D stoichiometric network (`Dataset/isokin/recon3d_model.json`), recursive GPR isoform/subunit-complex decomposition (`Dataset/isokin/recon3d_gpr_rules.json`: soft-min across `AND` complex subunits, linear-scale abundance-weighted aggregation across `OR` isozyme branches), and soft steady-state flux balance (`S · v ≈ 0`) via Pyro low-rank multivariate normal SVI.

```bash
# 1. Validate cohort AnnData (.h5ad) schema and GPR gene overlap
enzcast-isokin validate-h5ad \
  --input examples/sample_cohort.h5ad \
  --patient-col subject_id \
  --celltype-col cell_state

# 2. Run stoichiometric + GPR-resolved Bayesian kinetic fitting on a cohort
enzcast-isokin infer \
  --input examples/sample_cohort.h5ad \
  --patient-col subject_id \
  --celltype-col cell_state \
  --project-root .. \
  --observed-genes-only \
  --max-iterations 200 \
  --posterior-samples 200 \
  --out /tmp/isokin_cohort_output

# 3. Summarize posterior outputs
enzcast-isokin summarize --input /tmp/isokin_cohort_output
```

For full-cohort production inference, omit `--max-iterations` and `--posterior-samples` to use the default schedule (`20,000` SVI iterations, `2,000` posterior draws) and omit `--observed-genes-only` when running on transcriptome-wide `.h5ad` cohorts.

---

## Minimal Verification Suite

The bundled human purine nucleoside phosphorylase example (UniProt P00491) covers all four kinetic targets. The verification script compares eight rows with saved held-out predictions, then tests cohort fitting with the bundled example cohort.

```bash
python examples/run_minimal_verification.py --device cpu
pytest isokin/tests
```

## License

Code is released under the MIT License (see `LICENSE`). External databases and structural resources remain governed by their respective upstream terms (see `../Dataset/kinbench/DATA_DICTIONARY.md`).
