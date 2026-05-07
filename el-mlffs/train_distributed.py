from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader


@dataclass
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1

    if enabled:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return DistributedContext(
        enabled=enabled,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def cleanup_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def wrap_model(
    model: torch.nn.Module,
    context: DistributedContext,
    find_unused_parameters: bool = False,
) -> torch.nn.Module:
    if not context.enabled:
        return model
    if context.device.type == "cuda":
        return DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            find_unused_parameters=find_unused_parameters,
        )
    return DistributedDataParallel(model, find_unused_parameters=find_unused_parameters)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def build_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    context: DistributedContext,
    num_workers: int = 0,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = None
    if context.enabled:
        sampler = DistributedSampler(dataset, num_replicas=context.world_size, rank=context.rank, shuffle=shuffle)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=context.device.type == "cuda",
    )
    return loader, sampler


def reduce_average(total: float, count: float, context: DistributedContext) -> float:
    if count == 0:
        return 0.0
    stats = torch.tensor([total, count], device=context.device, dtype=torch.float64)
    if context.enabled:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return (stats[0] / stats[1]).item()
