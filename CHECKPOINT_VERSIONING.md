# Checkpoint & Code Version Management

> Auto-generated on 2026-05-05.  Managing connected/detach, NVE variants, and checkpoint lineage.

---

## 1. Meta-Model Variants (Connected vs Detach)

The key flag is `differentiate_force_features` in `ConservativeEnergyMixer` (`el-mlffs/torch_ensemble_models.py`).

| Variant | Flag | Force Accuracy | Energy Conservation | Speed | NVE Stability |
|:---|:---|:---|:---|:---|:---|
| **Detach (Fast)** | `False` (default) | Best | ❌ NOT strictly conservative | Fast (~223 ms) | Temperature drift |
| **Connected (Strict)** | `True` | Slightly worse | ✅ Strictly conservative | Slow (~310 ms) | Temperature drift (PES shape) |

### Existing Checkpoints

| Checkpoint | Path | Connected? | #Base | Base Models | Data | Force MAE | Status |
|:---|:---|:---:|:---:|:---|:---|---:|:---|
| `conservative_meta_current_bases_8gpu` | `meta_models/` | ❌ Detach | 7 | all | old OC20-like | **0.0074** | ✅ Primary |
| `conservative_meta_4model_no_dp_painn_mace_l2` | `peptide_dft_wb97x_5k_meta_models/` | ❌ Detach | 4 | nep,mtp,soap,schnet | peptide DFT | 0.131 | ✅ |
| `conservative_meta_4model_no_dp_painn_mace_l2_connected` | `peptide_dft_wb97x_5k_meta_models/` | ✅ **Connected** | 4 | nep,mtp,soap,schnet | peptide DFT | 0.129 | ✅ |
| `conservative_meta_5model_no_dp_painn` | `peptide_dft_wb97x_5k_meta_models/` | ❌ Detach | 5 | nep,mtp,soap,schnet,mace | peptide DFT | 0.104 | ✅ |
| 48 combo checkpoints | `meta_models/conservative_combo/` | ❌ Detach | 1–6 | various | old OC20-like | 0.010–0.020 | ✅ |

**Missing:**
- `conservative_meta_current_bases_8gpu_connected.pth` — training script exists (`old_data_connected_training_scripts/1_meta_conservative_7model_connected_8gpu.sh`) but **checkpoint NOT found on disk**.

---

## 2. NVE Code Variants

NVE runner: `el-mlffs/run_conservative_meta_nve.py`

| Variant | `differentiate_force_features` | `fixed_neighbor_graph` | CLI Flag | Use Case |
|:---|:---|:---|:---|:---|
| **Strict NVE** | `True` | `True` | (default) | Paper NVE claims |
| **Fast NVE** | `False` | `True`/`False` | `--fast-detached-force-features` | Quick testing |

**⚠️ "Smooth vs Unsmoothed" does NOT exist for NVE.**  
The word "smooth" in the codebase (e.g. `SmoothPeriodicSchNet`, `SmoothMACELite`) refers to **base-model cutoff envelope functions**, not NVE variants.

### NVE Cutoff-Safe Retraining Pipeline (NOT YET RUN)

Directory: `nve_cutoff_safe_training_scripts/`
- 8 scripts planned (1–7 base + 1 meta)
- Target outputs: `base_models_nve_cutoff_safe/` + `meta_models/nve_cutoff_safe/`
- **Status:** scripts exist, target directories are **EMPTY**

---

## 3. Base Model Checkpoints

| Directory | #Models | Models | Data | Notes |
|:---|:---:|:---|:---|:---|
| `base_models_a100_8gpu_50ksteps/` | 7 | dp,nep,mtp,soap,painn,schnet,mace | old OC20-like | **Primary** for combos |
| `base_models_a100_e150_cosine/` | 3 | dp,painn,schnet | old OC20-like | Cosine LR schedule |
| `base_models_a100/` | 2 | painn,schnet | old OC20-like | Early experiments |
| `base_models_seq_smoke/` | 1 | schnet | old OC20-like | Smoke test |
| `peptide_dft_wb97x_5k_base_models/` | 7 | dp,nep,mtp,soap,painn,schnet,mace | peptide DFT ωB97X | Peptide experiments |

---

## 4. Naming Convention (Recommended)

Current naming is inconsistent. Adopt this scheme for future checkpoints:

```
{data}_{architecture}_{n}model_{models}_{connected}_{hyperparams}.pth

Examples:
  oc20_conservative_7model_all_detach.pth
  oc20_conservative_7model_all_connected.pth
  peptide_conservative_4model_nep_mtp_soap_schnet_connected.pth
  peptide_conservative_5model_nep_mtp_soap_schnet_mace_detach.pth
```

Required metadata fields (already partially implemented in `checkpoint["config"]`):
- `differentiate_force_features`
- `base_model_names`
- `data_source`
- `weight_decay`
- `learning_rate`
- `epochs` / `steps`

---

## 5. Delivery Bundle Staleness

`ybc_delivery_bundle/el-mlffs/` is a **stale snapshot** of the main code. It lacks:
- `differentiate_force_features` CLI support
- Connected-path training scripts
- NVE cutoff-safe scripts

**Action:** Do not use delivery bundle code for connected/NVE experiments. Use main repo only.

---

## 6. Full Manifest

See `CKPT_MANIFEST.csv` (52 entries) for machine-readable listing of all meta checkpoints.

```bash
# Quick stats
$ python -c "import csv; rows=list(csv.DictReader(open('CKPT_MANIFEST.csv'))); print(f'Total: {len(rows)}, Connected: {sum(1 for r in rows if r[\"connected\"]==\"True\")}, Detach: {sum(1 for r in rows if r[\"connected\"]==\"False\")}')"
# Total: 52, Connected: 1, Detach: 51
```
