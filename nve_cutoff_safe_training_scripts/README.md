# NVE Cutoff-Safe Retraining Scripts

Purpose: retrain only the models needed for the NVE line after the cutoff-safe architecture patch.
This does not overwrite previous non-NVE experiment outputs.

Tasks:

- `1_dp.sh`
- `2_nep.sh`
- `3_mtp.sh`
- `4_soap.sh`
- `5_painn.sh`
- `6_schnet.sh`
- `7_mace.sh`
- `8_meta_conservative.sh`

Default outputs:

- Base checkpoints: `el-mlffs/checkpoints/base_models_nve_cutoff_safe/{model}_torch.pth`
- Meta checkpoint: `el-mlffs/checkpoints/meta_models/nve_cutoff_safe/conservative_meta_7model_nve_cutoff_safe.pth`
- Logs: `el-mlffs/logs/base_models_nve_cutoff_safe/` and `el-mlffs/logs/meta_models_nve_cutoff_safe/`

Default schedule:

- Base models: `BASE_TARGET_TOTAL_STEPS=50000`
- Conservative meta: `META_TARGET_TOTAL_STEPS=20000`
- Base LR: `BASE_LR=5e-4`, `BASE_MIN_LR=1e-6`
- Meta LR: `META_LR=5e-4`, `META_MIN_LR=1e-6`
- Force weight: `50.0`
- Cutoff: `5.0`

Run examples:

```bash
cd $PWD
bash nve_cutoff_safe_training_scripts/1_dp.sh
bash nve_cutoff_safe_training_scripts/2_nep.sh
bash nve_cutoff_safe_training_scripts/3_mtp.sh
bash nve_cutoff_safe_training_scripts/4_soap.sh
bash nve_cutoff_safe_training_scripts/5_painn.sh
bash nve_cutoff_safe_training_scripts/6_schnet.sh
bash nve_cutoff_safe_training_scripts/7_mace.sh
bash nve_cutoff_safe_training_scripts/8_meta_conservative.sh
```

Sequential all-in-one:

```bash
cd $PWD
bash nve_cutoff_safe_training_scripts/run_all_sequential.sh
```

Multi-GPU meta example:

```bash
GPU_IDS=0,1,2,3 NPROC_PER_NODE=4 bash nve_cutoff_safe_training_scripts/8_meta_conservative.sh
```

The scripts intentionally install only `requirements-conservative-meta.txt` and do not install `torch-cluster`, `torch-sparse`, or `torch-scatter`.

Optional 7-GPU base-model parallel run:

```bash
cd $PWD
bash nve_cutoff_safe_training_scripts/run_bases_parallel_7gpu.sh
bash nve_cutoff_safe_training_scripts/8_meta_conservative.sh
```
