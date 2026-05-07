# OC20 Base-Model Multi-GPU Scripts

This directory contains 7 multi-GPU training scripts for compact base models on OC20 S2EF `2M`.

Files:
- `1.sh`: `dp`
- `2.sh`: `nep`
- `3.sh`: `mtp`
- `4.sh`: `soap`
- `5.sh`: `painn`
- `6.sh`: `schnet`
- `7.sh`: `mace`

Behavior:
- configures the proxy
- creates/activates a `mamba` environment
- installs the Python dependencies needed by this repo
- preprocesses OC20 data into EL-MLFFs LMDB shards if the processed splits are missing
- autotunes a per-GPU local batch size
- launches distributed training with `torchrun`

Expected data location:
- `data/oc20/extracted/2M`
- `data/oc20/extracted/val_id`
- `data/oc20/extracted/val_ood_ads`
- `data/oc20/extracted/val_ood_cat`
- `data/oc20/extracted/val_ood_both`

Useful overrides:
```bash
GPU_IDS=0,1,2,3 NPROC_PER_NODE=4 bash 1.sh
TARGET_TOTAL_STEPS=40000 bash 6.sh
STORE_EDGES=1 DATA_PREP_WORKERS=16 bash 7.sh
```
