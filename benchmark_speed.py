from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

ROOT_DIR = Path(__file__).resolve().parent
MODULE_DIR = ROOT_DIR / "el-mlffs"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from torch_base_models import BASE_MODEL_REGISTRY
from torch_data import ExtXYZDataset
from torch_ensemble_models import BaseModelStack, ConservativeEnergyMixer, DirectForceFittingEnsemble
from torch_workflow import unique_atomic_numbers


BASE_MODELS = ("dp", "nep", "mtp", "soap", "painn", "schnet", "mace")
DEFAULT_CHECKPOINT_DIR = ROOT_DIR / "el-mlffs" / "checkpoints" / "base_models_a100_8gpu_50ksteps"


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_base_models(
    model_names: tuple[str, ...],
    checkpoint_dir: Path,
    cutoff: float,
    device: torch.device,
    all_z: list[int],
) -> dict[str, nn.Module]:
    models: dict[str, nn.Module] = {}
    for model_name in model_names:
        checkpoint_path = checkpoint_dir / f"{model_name}_torch.pth"
        model_cls = BASE_MODEL_REGISTRY[model_name]
        if model_name in {"dp", "nep", "mtp", "soap"}:
            model = model_cls(z_list=all_z, cutoff=cutoff).to(device)
        else:
            model = model_cls(cutoff=cutoff).to(device)
        if checkpoint_path.exists():
            payload = torch.load(checkpoint_path, map_location=device)
            state_dict = payload.get("state_dict", payload)
            model.load_state_dict(state_dict, strict=False)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        models[model_name] = model
    return models


def get_batch(
    data_file: str,
    cutoff: float,
    device: torch.device,
    batch_size: int = 8,
) -> object:
    dataset = ExtXYZDataset(data_file, cutoff=cutoff)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    return batch.to(device)


def measure_latency(
    fn: Callable,
    warmup: int = 10,
    repeats: int = 100,
    batch_size: int = 1,
) -> dict[str, float]:
    """Measure inference latency in milliseconds."""
    device = torch.cuda.current_device() if torch.cuda.is_available() else None

    # Warmup
    for _ in range(warmup):
        fn()
    if device is not None:
        torch.cuda.synchronize(device)

    # Timing with CUDA events if available
    if device is not None:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(repeats):
            start_event.record()
            fn()
            end_event.record()
            torch.cuda.synchronize(device)
            times.append(start_event.elapsed_time(end_event))
    else:
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    import numpy as np

    arr = np.array(times)
    return {
        "latency_ms_mean": float(arr.mean()),
        "latency_ms_median": float(np.median(arr)),
        "latency_ms_std": float(arr.std(ddof=0)),
        "latency_ms_p99": float(np.percentile(arr, 99)),
        "throughput_graphs_per_sec": 1000.0 * batch_size / float(arr.mean()),
    }


def measure_memory(fn: Callable) -> dict[str, float]:
    """Measure peak GPU memory in MB."""
    if not torch.cuda.is_available():
        return {"peak_memory_mb": 0.0, "allocated_mb": 0.0}

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    fn()
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    return {"peak_memory_mb": peak_mb, "allocated_mb": allocated_mb}


def benchmark_base_models(
    checkpoint_dir: Path,
    batch: object,
    device: torch.device,
    all_z: list[int],
    cutoff: float,
    batch_size: int = 1,
) -> list[dict[str, object]]:
    results = []
    for model_name in BASE_MODELS:
        print(f"Benchmarking base model: {model_name}")
        models = load_base_models((model_name,), checkpoint_dir, cutoff, device, all_z)
        model = models[model_name]
        param_count = count_parameters(model)

        def fn():
            b = batch.clone()
            _ = model(b, compute_forces=True, create_graph=False)

        latency = measure_latency(fn, batch_size=batch_size)
        memory = measure_memory(fn)
        results.append(
            {
                "model_category": "base_model",
                "model_name": model_name,
                "ensemble_size": 1,
                "param_count": param_count,
                **latency,
                **memory,
            }
        )
        del model, models
        torch.cuda.empty_cache()
    return results


def benchmark_avg_baseline(
    checkpoint_dir: Path,
    batch: object,
    device: torch.device,
    all_z: list[int],
    cutoff: float,
    batch_size: int = 1,
) -> list[dict[str, object]]:
    results = []
    for size in [1, 3, 7]:
        if size == 1:
            combo = ("mace",)
        elif size == 3:
            combo = ("painn", "schnet", "mace")
        else:
            combo = BASE_MODELS

        print(f"Benchmarking AVG baseline: {combo}")
        models = load_base_models(combo, checkpoint_dir, cutoff, device, all_z)

        def fn():
            b = batch.clone()
            forces_list = []
            for m in models.values():
                out = m(b, compute_forces=True, create_graph=False)
                forces_list.append(out["forces"])
            _ = torch.stack(forces_list, dim=0).mean(dim=0)

        latency = measure_latency(fn, batch_size=batch_size)
        memory = measure_memory(fn)
        results.append(
            {
                "model_category": "avg_baseline",
                "model_name": f"AVG-{size}",
                "ensemble_size": size,
                "param_count": sum(count_parameters(m) for m in models.values()),
                **latency,
                **memory,
            }
        )
        del models
        torch.cuda.empty_cache()
    return results


def benchmark_lnc(
    checkpoint_dir: Path,
    batch: object,
    device: torch.device,
    all_z: list[int],
    cutoff: float,
    batch_size: int = 1,
) -> list[dict[str, object]]:
    from eval_lnc_baseline import LNCBaseline

    results = []
    for size in [1, 3, 7]:
        if size == 1:
            combo = ("mace",)
        elif size == 3:
            combo = ("painn", "schnet", "mace")
        else:
            combo = BASE_MODELS

        print(f"Benchmarking LNC: {combo}")
        models = load_base_models(combo, checkpoint_dir, cutoff, device, all_z)
        lnc = LNCBaseline(num_models=len(combo), hidden_dims=(256, 256, 128)).to(device)
        lnc.eval()
        for param in lnc.parameters():
            param.requires_grad_(False)

        def fn():
            b = batch.clone()
            forces_list = []
            for m in models.values():
                out = m(b, compute_forces=True, create_graph=False)
                forces_list.append(out["forces"])
            base_forces = torch.stack(forces_list, dim=0)
            _ = lnc(b.z, base_forces)

        latency = measure_latency(fn, batch_size=batch_size)
        memory = measure_memory(fn)
        results.append(
            {
                "model_category": "lnc_baseline",
                "model_name": f"LNC-{size}",
                "ensemble_size": size,
                "param_count": sum(count_parameters(m) for m in models.values()) + count_parameters(lnc),
                **latency,
                **memory,
            }
        )
        del models, lnc
        torch.cuda.empty_cache()
    return results


def benchmark_ensemble(
    checkpoint_dir: Path,
    batch: object,
    device: torch.device,
    all_z: list[int],
    cutoff: float,
    architecture: str,
    batch_size: int = 1,
    differentiate_force_features: bool = False,
) -> list[dict[str, object]]:
    results = []
    for size in [1, 3, 7]:
        if size == 1:
            combo = ("mace",)
        elif size == 3:
            combo = ("painn", "schnet", "mace")
        else:
            combo = BASE_MODELS

        print(f"Benchmarking Ensemble-{architecture.upper()}: {combo}")
        models = load_base_models(combo, checkpoint_dir, cutoff, device, all_z)
        stack = BaseModelStack(models)
        stack.freeze()

        if architecture == "direct":
            ensemble_model = DirectForceFittingEnsemble(
                base_models=stack,
                atom_emb_dim=16,
                hidden_scalar_channels=64,
                hidden_vector_channels=32,
                num_layers=3,
                num_basis=32,
                cutoff=cutoff,
            ).to(device)
        else:
            ensemble_model = ConservativeEnergyMixer(
                base_models=stack,
                atom_emb_dim=32,
                hidden_scalar_channels=64,
                hidden_vector_channels=32,
                num_layers=3,
                num_basis=32,
                cutoff=cutoff,
                differentiate_force_features=differentiate_force_features,
            ).to(device)

        ensemble_model.eval()
        for param in ensemble_model.parameters():
            param.requires_grad_(False)

        # Full forward: base + meta
        def fn_full():
            b = batch.clone()
            _ = ensemble_model(b)

        latency_full = measure_latency(fn_full, batch_size=batch_size)
        memory_full = measure_memory(fn_full)
        results.append(
            {
                "model_category": f"ensemble_{architecture}_full",
                "model_name": f"Ensemble-{architecture.upper()}-{size}",
                "ensemble_size": size,
                "param_count": count_parameters(ensemble_model),
                **latency_full,
                **memory_full,
            }
        )

        # Meta-only forward: assume base predictions precomputed
        base_preds = stack(batch.clone(), create_graph=False)

        def fn_meta():
            b = batch.clone()
            _ = ensemble_model(b, base_predictions=base_preds)

        latency_meta = measure_latency(fn_meta, batch_size=batch_size)
        memory_meta = measure_memory(fn_meta)
        results.append(
            {
                "model_category": f"ensemble_{architecture}_meta_only",
                "model_name": f"Ensemble-{architecture.upper()}-{size}-meta",
                "ensemble_size": size,
                "param_count": count_parameters(ensemble_model.encoder) + count_parameters(ensemble_model.force_head if hasattr(ensemble_model, "force_head") else ensemble_model.atomic_correction),
                **latency_meta,
                **memory_meta,
            }
        )

        del ensemble_model, stack, models, base_preds
        torch.cuda.empty_cache()
    return results


def benchmark_lsm_direct(
    batch: object,
    device: torch.device,
    batch_size: int = 1,
) -> list[dict[str, object]]:
    from train_lsm_direct import LSMDirectModel

    print("Benchmarking LSM-Direct")
    model = LSMDirectModel(
        num_atom_types=100,
        atom_emb_dim=64,
        hidden_dim=512,
        num_heads=8,
        num_layers=10,
        dropout=0.0,
        jk_mode="cat",
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    def fn():
        b = batch.clone()
        _ = model(b, compute_forces=True, create_graph=False)

    latency = measure_latency(fn, batch_size=batch_size)
    memory = measure_memory(fn)
    return [
        {
            "model_category": "lsm_direct",
            "model_name": "LSM-Direct",
            "ensemble_size": 1,
            "param_count": count_parameters(model),
            **latency,
            **memory,
        }
    ]


def benchmark_lsm_conserv(
    batch: object,
    device: torch.device,
    batch_size: int = 1,
) -> list[dict[str, object]]:
    from train_lsm_conserv import LSMConservativeEnergyMixer

    print("Benchmarking LSM-Conserv")
    model = LSMConservativeEnergyMixer(
        atom_emb_dim=64,
        hidden_scalar_channels=256,
        hidden_vector_channels=128,
        num_layers=6,
        num_basis=64,
        cutoff=5.0,
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    def fn():
        b = batch.clone()
        _ = model(b)

    latency = measure_latency(fn, batch_size=batch_size)
    memory = measure_memory(fn)
    return [
        {
            "model_category": "lsm_conserv",
            "model_name": "LSM-Conserv",
            "ensemble_size": 1,
            "param_count": count_parameters(model),
            **latency,
            **memory,
        }
    ]


def save_csv(results: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "model_category",
        "model_name",
        "ensemble_size",
        "param_count",
        "latency_ms_mean",
        "latency_ms_median",
        "latency_ms_std",
        "latency_ms_p99",
        "throughput_graphs_per_sec",
        "peak_memory_mb",
        "allocated_mb",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in keys})
    print(f"Saved speed benchmark results to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark inference speed and memory for all model architectures. No training required.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--data-file", default="el-mlffs/data/test.extxyz")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-csv", type=Path, default=ROOT_DIR / "reports" / "speed_benchmark.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA device ID to use (default: 0)")
    parser.add_argument("--skip-base", action="store_true", help="Skip base model benchmarks")
    parser.add_argument("--skip-avg", action="store_true", help="Skip AVG baseline")
    parser.add_argument("--skip-lnc", action="store_true", help="Skip LNC baseline")
    parser.add_argument("--skip-ensemble-direct", action="store_true", help="Skip Direct ensemble")
    parser.add_argument("--skip-ensemble-conserv", action="store_true", help="Skip Conservative ensemble")
    parser.add_argument("--skip-lsm", action="store_true", help="Skip LSM models")
    parser.add_argument("--differentiate-force-features", action="store_true", help="Enable connected conservative path (no detach on base force features)")
    args = parser.parse_args()

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device(args.device)
    batch = get_batch(args.data_file, args.cutoff, device, batch_size=args.batch_size)
    all_z = sorted({int(z) for data in ExtXYZDataset(args.data_file, cutoff=args.cutoff) for z in data.z})

    print(f"Benchmarking on device: {device}")
    print(f"Batch size: {args.batch_size}, num_graphs: {batch.num_graphs}, num_atoms: {batch.z.size(0)}")

    all_results: list[dict[str, object]] = []

    if not args.skip_base:
        all_results.extend(benchmark_base_models(args.checkpoint_dir, batch, device, all_z, args.cutoff, args.batch_size))

    if not args.skip_avg:
        all_results.extend(benchmark_avg_baseline(args.checkpoint_dir, batch, device, all_z, args.cutoff, args.batch_size))

    if not args.skip_lnc:
        all_results.extend(benchmark_lnc(args.checkpoint_dir, batch, device, all_z, args.cutoff, args.batch_size))

    if not args.skip_ensemble_direct:
        all_results.extend(benchmark_ensemble(args.checkpoint_dir, batch, device, all_z, args.cutoff, "direct", args.batch_size))

    if not args.skip_ensemble_conserv:
        all_results.extend(benchmark_ensemble(args.checkpoint_dir, batch, device, all_z, args.cutoff, "conservative", args.batch_size, differentiate_force_features=args.differentiate_force_features))

    if not args.skip_lsm:
        all_results.extend(benchmark_lsm_direct(batch, device, args.batch_size))
        all_results.extend(benchmark_lsm_conserv(batch, device, args.batch_size))

    save_csv(all_results, args.output_csv)

    # Print summary table
    print("\n" + "=" * 100)
    print(f"{'Model':<40} {'Params':>12} {'Latency(ms)':>14} {'Throughput':>14} {'PeakMem(MB)':>12}")
    print("=" * 100)
    for row in all_results:
        print(
            f"{row['model_name']:<40} "
            f"{row['param_count']:>12,} "
            f"{row['latency_ms_median']:>14.3f} "
            f"{row['throughput_graphs_per_sec']:>14.2f} "
            f"{row['peak_memory_mb']:>12.1f}"
        )
    print("=" * 100)


if __name__ == "__main__":
    main()
