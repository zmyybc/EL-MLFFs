from __future__ import annotations

import random
from typing import Iterable

import torch
from torch.utils.data import Subset


def random_split_dataset(
    dataset,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)


def force_magnitude_score(data) -> float:
    forces = getattr(data, "forces", None)
    if forces is None:
        raise AttributeError("Dataset item is missing `forces`, cannot build OOD split.")
    return torch.max(torch.abs(forces)).item()


def build_ood_force_split(
    dataset,
    low_train_fraction: float = 0.65,
    high_train_fraction: float = 0.35,
    seed: int = 42,
) -> tuple[Subset, Subset, dict[str, float]]:
    scores = [(idx, force_magnitude_score(dataset[idx])) for idx in range(len(dataset))]
    scores.sort(key=lambda item: item[1])

    mid_idx = len(scores) // 2
    low_pool = scores[:mid_idx]
    high_pool = scores[mid_idx:]

    rng = random.Random(seed)
    rng.shuffle(low_pool)
    rng.shuffle(high_pool)

    low_train_cut = int(len(low_pool) * low_train_fraction)
    high_train_cut = int(len(high_pool) * high_train_fraction)

    train_indices = [idx for idx, _ in low_pool[:low_train_cut] + high_pool[:high_train_cut]]
    test_indices = [idx for idx, _ in low_pool[low_train_cut:] + high_pool[high_train_cut:]]

    metadata = {
        "mid_force": scores[mid_idx][1] if scores else 0.0,
        "num_train": float(len(train_indices)),
        "num_test": float(len(test_indices)),
        "num_low_train": float(low_train_cut),
        "num_low_test": float(len(low_pool) - low_train_cut),
        "num_high_train": float(high_train_cut),
        "num_high_test": float(len(high_pool) - high_train_cut),
    }
    return Subset(dataset, train_indices), Subset(dataset, test_indices), metadata


def unique_atomic_numbers(dataset: Iterable) -> list[int]:
    atomic_numbers = getattr(dataset, "atomic_numbers", None)
    if atomic_numbers:
        return sorted({int(z) for z in atomic_numbers})

    z_values: set[int] = set()
    for data in dataset:
        for z in data.z:
            z_values.add(int(z.item()))
    return sorted(z_values)
