# EnzCast

[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**EnzCast** is a multi-modal deep learning framework for predicting enzyme kinetic parameters from protein sequences, three-dimensional structures, substrate molecules, and experimental conditions. It supports prediction of four fundamental kinetic constants — Km, kcat, kcat/Km, and Ki — all expressed in log10 space. EnzCast further integrates isoform-resolved Bayesian inference from single-cell transcriptomics to produce patient-specific effective kinetic estimates.

This repository accompanies the paper:

> **EnzCast: patient-specific enzyme kinetics through multi-modal deep learning and isoform-resolved Bayesian inference from single-cell transcriptomics**
> *Nature Methods* (2026). DOI: [10.XXXX/XXXXXXXX](https://doi.org/10.XXXX/XXXXXXXX) *(placeholder)*

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Data](#data)
6. [Model Checkpoints](#model-checkpoints)
7. [Usage](#usage)
8. [Performance](#performance)
9. [License](#license)
10. [Citation](#citation)

---

## Overview

Accurate prediction of enzyme kinetic parameters is a long-standing challenge in computational biochemistry, with direct implications for metabolic modelling, drug discovery, and personalized medicine. EnzCast addresses this challenge by combining:

- A protein language model (ESM-2, 650M parameters) with parameter-efficient LoRA fine-tuning to capture sequence-level enzymatic features.
- A hierarchical Virtual Node Equivariant Graph Neural Network (VN-EGNN) to encode three-dimensional protein structure.
- A graph neural network combined with a Transformer encoder for substrate molecular graphs.
- A condition encoder that integrates experimental pH, temperature, organism, EC number, enzyme type, and Morgan fingerprint descriptors.
- A gated fusion module that combines all modalities.
- Dual-pathway prediction heads with heteroscedastic uncertainty (Beta-NLL loss) and Exponential Moving Average (EMA) weight averaging.

A task-adaptive training strategy is employed: kcat and kcat/Km are trained jointly via multi-task learning (MTL) because they share the catalytic machinery, while Km and Ki are trained as separate single-task experts to avoid negative transfer.

---

## Architecture

```
Input
 ├── Protein sequence           --> ESM-2 (650M, LoRA) --> residue + global embeddings
 ├── Protein 3D structure       --> VN-EGNN (hierarchical) --> structure embedding
 ├── Substrate SMILES           --> GIN + Transformer (multi-pooling) --> substrate embedding
 └── Conditions (pH, T, org, EC, type, Morgan FP) --> Condition Encoder --> condition embedding
                                                             |
                                            Gated Bilinear Fusion Module
                                                             |
                               +-----------------------------+-----------------------------+
                               |                                                           |
                      Binding Pathway                                          Catalytic Pathway
                  (Km head / Ki head)                               (kcat head / kcat_Km head)
                               |                                                           |
                        log10(Km), log10(Ki)                       log10(kcat), log10(kcat/Km)
                        + predictive uncertainty                    + predictive uncertainty
```

**Key design choices:**

| Component | Detail |
|---|---|
| Protein language model | ESM-2 (650M), layers 10/21/32 extracted, LoRA fine-tuning (r=8, alpha=16, dropout 0.1) |
| Structure encoder | Hierarchical VN-EGNN with local + global residue features |
| Substrate encoder | GIN layers + Transformer, multi-pooling (mean, max, first, penultimate) |
| Condition encoder | pH, temperature, species embedding, EC number (Recon3D priors), Morgan FP (2048-bit, radius 3), enzyme type |
| Fusion | Gated Fusion Module (MLP + sigmoid gate) |
| Loss | Beta-NLL (heteroscedastic), auxiliary consistency loss |
| Regularisation | EMA decay = 0.999, gradient checkpointing, LoRA dropout |
| Hidden dimension | 256 (backbone), 128 (condition) |
| Sequence max length | 1024 (active-site centred crop for longer sequences) |

---

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- PyTorch Geometric >= 2.7
- `transformers` (Hugging Face)
- `peft` (LoRA support)
- `rdkit`
- `scikit-learn`
- `pandas`
- `numpy`
- `tqdm`

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Mxc666/EnzCast.git
cd EnzCast

# Create and activate a conda environment (recommended)
conda create -n enzcast python=3.10
conda activate enzcast

# Install PyTorch (adjust the CUDA version as needed)
pip install torch>=2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch_geometric

# Install remaining dependencies
pip install transformers peft rdkit scikit-learn pandas numpy tqdm
```

### ESM-2 Protein Language Model

EnzCast relies on the ESM-2 (650M) model. The first time you run training or inference, the `transformers` library will download it automatically from the Hugging Face Hub:

```
facebook/esm2_t33_650M_UR50D
```

To pre-download manually:

```python
from transformers import EsmModel, EsmTokenizer
EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
```

---

## Data

### KinBench

KinBench is the benchmark dataset constructed for this work, aggregating and standardising kinetic records from BRENDA and SABIO-RK. It covers:

| Statistic | Value |
|---|---|
| Total records | 359,652 |
| Unique enzymes | 25,118 |
| Unique substrates | 23,547 |
| Tasks | Km, kcat, kcat/Km, Ki |

Splits are generated using UniRef50 sequence-identity clustering to ensure no sequence leakage between train, validation, and test sets.

| Split | Records |
|---|---|
| Train | 285,892 |
| Validation | 36,770 |
| Test | 36,990 |

The benchmark splits are located in:

```
data/KinBench/
├── train_split.csv
├── val_split.csv
└── test_split.csv
```

Each CSV file contains the following columns:

| Column | Description |
|---|---|
| `UniProt_ID` | UniProt accession of the enzyme |
| `Sequence` | Full amino acid sequence |
| `SMILES` | Canonical SMILES string of the substrate |
| `Task` | One of `km`, `kcat`, `kcat_km`, `ki` |
| `Log10_Value` | Experimentally measured kinetic constant in log10 units |
| `pH` | Assay pH (NaN if unavailable) |
| `Temperature` | Assay temperature in °C (NaN if unavailable) |
| `Organism` | Source organism name |
| `EC_Number` | Enzyme Commission number |

### Preprocessing inputs

Before training, protein 3D structure graphs and substrate molecular graphs must be pre-computed and placed in the appropriate directories under `project_root`:

- **Protein structure graphs**: `Data/Structure_Data/Graph_Features-v2/{UniProt_ID}.pt` (GearNet-style graphs from AlphaFold or ESMFold structures)
- **Substrate molecular graphs**: `Data/Structure_Data/Substrate_Graphs-v2/{MD5(SMILES)}.pt` (molecular graphs with 42-dimensional atom features)

Graph preprocessing scripts are included in the full project repository. Please refer to the paper for details on the graph construction pipeline.

---

## Model Checkpoints

Pre-trained model weights are hosted on Hugging Face Hub due to file size (~2.5 GB per checkpoint):

**Repository:** [https://huggingface.co/Mxc666/EnzCast](https://huggingface.co/Mxc666/EnzCast) *(to be released upon publication)*

```bash
# Download all checkpoints using the Hugging Face CLI
pip install huggingface_hub
huggingface-cli download Mxc666/EnzCast --local-dir models/
```

The `models/` directory should contain the following files after download:

| File | Purpose |
|---|---|
| `enzcast_multitask_ema.pt` | EMA weights for the MTL model (kcat + kcat/Km) |
| `enzcast_singletask_km_ema.pt` | EMA weights for the single-task Km model |
| `enzcast_singletask_ki_ema.pt` | EMA weights for the single-task Ki model |
| `multitask_metadata.pkl` | Training statistics (mean/std) for the MTL model |
| `singletask_km_metadata.pkl` | Training statistics for the Km model |
| `singletask_ki_metadata.pkl` | Training statistics for the Ki model |

The metadata `.pkl` files store per-task normalisation statistics (mean and standard deviation in log10 space) that are required to de-normalise model outputs back to physical units. **Always use the metadata file paired with its corresponding checkpoint.**

---

## Usage

### Training

Ensure that graph files are pre-computed and that `data/KinBench/` contains the benchmark splits before training.

**Multi-task training (kcat + kcat/Km)**

```bash
python src/train_multitask.py --project_root /path/to/EnzCast
```

**Single-task training (Km)**

```bash
python src/train_singletask_km.py --task km --project_root /path/to/EnzCast
```

**Single-task training (Ki)**

```bash
python src/train_singletask_ki.py --task ki --project_root /path/to/EnzCast
```

Key training arguments (all scripts):

| Argument | Default | Description |
|---|---|---|
| `--project_root` | `.` | Path to the repository root |
| `--epochs` | varies | Total training epochs |
| `--batch_size` | auto | Overrides the GPU-auto-detected batch size |
| `--lr_backbone` | 8e-5 | Learning rate for the ESM-2 backbone |
| `--lr_head` | 6e-4 | Learning rate for prediction heads |
| `--seed` | 42 | Random seed for reproducibility |

GPU memory and precision are detected automatically:
- A100/A800: batch size 24, bfloat16
- RTX 4090: batch size 6, bfloat16
- V100: batch size 4, float16

---

## Performance

Benchmark performance on the KinBench test split (UniRef50 clustering-based). Results are reported as R² (coefficient of determination) averaged over multiple independent training seeds.

| Task | R² (mean ± std) |
|---|---|
| Km | 0.404 ± 0.001 |
| kcat | 0.234 ± 0.020 |
| kcat/Km | 0.083 ± 0.007 |
| Ki | 0.468 ± 0.013 |

All values are computed on log10-transformed kinetic constants. The relatively lower R² for kcat/Km reflects the intrinsic difficulty of predicting catalytic efficiency, which compounds uncertainties in both kcat and Km.

EnzCast achieves state-of-the-art performance against baselines including DLKcat, UniKP, TurNuP, and CatPred across all four kinetic parameter tasks.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

If you use EnzCast or KinBench in your work, please cite:

```bibtex
@article{EnzCast2026,
  title     = {EnzCast: patient-specific enzyme kinetics through multi-modal deep learning
               and isoform-resolved Bayesian inference from single-cell transcriptomics},
  author    = {Mu, Xuechen and Yang, Yan and Wang, Qingyu and others},
  journal   = {Nature Methods},
  year      = {2026},
  doi       = {10.XXXX/XXXXXXXX},
  url       = {https://doi.org/10.XXXX/XXXXXXXX}
}
```

---

## Contact

For questions regarding the code or the paper, please open an issue on GitHub or contact the corresponding author via the information provided in the paper.
