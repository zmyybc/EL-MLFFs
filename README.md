# EL-MLFFs: Ensemble Learning for Machine Learning Force Fields

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains the official implementation of **EL-MLFFs**, an ensemble learning framework that combines multiple machine learning force field (MLFF) base models through learnable meta-models for improved accuracy and uncertainty quantification.

## Overview

EL-MLFFs supports two ensemble architectures:
- **Conservative ensemble**: Energy-based formulation with guaranteed energy-force consistency via automatic differentiation
- **Direct ensemble**: Direct force-fitting meta-model for faster inference

Base models include: Deep Potential (DP), Neuroevolution Potential (NEP), Moment Tensor Potential (MTP), Smooth Overlap of Atomic Positions (SOAP), SchNet, PaiNN, and MACE.

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/zmyybc/EL-MLFFs.git
cd EL-MLFFs
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
# or for conservative meta training:
pip install -r requirements-conservative-meta.txt
```

### 3. Download checkpoints and dataset

```bash
python download_artifacts.py
```

This downloads:
- 7 pre-trained base model checkpoints (~43 MB)
- 1 conservative meta-model checkpoint (~49 MB)
- Training and test datasets (~63 MB)

After extraction, artifacts are placed under `el-mlffs/checkpoints/` and `el-mlffs/data/`.

### 4. Run inference

```bash
python eval_oc20_conserv_meta_subset.py --checkpoint el-mlffs/checkpoints/meta_models/conservative_meta_current_bases_8gpu.pth
```

## Training

### Train base models

```bash
bash run_base_models_8gpu_50ksteps.sh
```

### Train conservative meta-model

```bash
python train_conservative_meta_with_bases.py \
  --base-model-checkpoints-dir el-mlffs/checkpoints/base_models_a100_8gpu_50ksteps \
  --data-file el-mlffs/data/train.extxyz \
  --architecture conservative
```

### Train direct meta-model

```bash
python train_direct_meta_with_bases.py \
  --base-model-checkpoints-dir el-mlffs/checkpoints/base_models_a100_8gpu_50ksteps \
  --data-file el-mlffs/data/train.extxyz \
  --architecture direct
```

## Repository Structure

```
EL-MLFFs/
├── el-mlffs/                       # Core package
│   ├── torch_base_models.py        # Base model implementations (DP, NEP, MTP, SOAP, SchNet, PaiNN, MACE)
│   ├── torch_ensemble_models.py    # Ensemble meta-models (ConservativeEnergyMixer, DirectForceFittingEnsemble)
│   ├── torch_data.py               # Dataset loaders and utilities
│   ├── train_distributed.py        # Distributed training utilities
│   └── ...
├── train_conservative_meta_with_bases.py   # Conservative meta-model training
├── train_direct_meta_with_bases.py         # Direct meta-model training
├── eval_*.py                       # Evaluation scripts (AVG, LWA, LNC, LSM baselines)
├── benchmark_speed.py              # Inference speed benchmark
├── download_artifacts.py           # Download checkpoints and datasets
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds upon the following atomic simulation packages and machine learning force field frameworks:

- **Deep Potential (DP)** — [DeepMD-kit](https://github.com/deepmodeling/deepmd-kit)
- **Neuroevolution Potential (NEP)** — [GPUMD](https://github.com/brucefan1983/GPUMD)
- **Moment Tensor Potential (MTP)** — [mlip](https://gitlab.com/ashapeev/mlip-3)
- **Smooth Overlap of Atomic Positions (SOAP)** — [QUIP/GAP](https://github.com/libAtoms/QUIP) and [dscribe](https://github.com/SINGROUP/dscribe)
- **SchNet** — [SchNetPack](https://github.com/atomistic-machine-learning/schnetpack)
- **Polarizable Atom Interaction Neural Network (PaiNN)** — [PaiNN](https://github.com/atomistic-machine-learning/painn)
- **MACE** — [MACE](https://github.com/ACEsuit/mace)

Datasets used in this work include OC20 S2EF 2M, self-computed peptide structures (Psi4 ωB97X/6-31G(d)), and methanol configurations.
