from typing import Any

import torch
from torch import nn, optim
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP


class DistributedInfo:
    def __init__(
        self,
        rank: int,
        local_rank: int,
        world_size: int,
        local_world_size: int
    ) -> None:
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.local_world_size = local_world_size
        if torch.cuda.device_count() == local_world_size:
            device_index = self.local_rank
        elif torch.cuda.device_count() == 1:
            device_index = 0
        else:
            raise RuntimeError(
                f"expected either {local_world_size} or 1 GPUs available, "
                f"but got {torch.cuda.device_count()} GPUs instead"
            )
        self.device = torch.device(device_index)

    @property
    def is_distributed(self) -> bool:
        return not self.is_single_gpu

    @property
    def is_single_gpu(self) -> bool:
        return self.world_size == 1

    @property
    def is_local_main_process(self) -> bool:
        return self.local_rank == 0

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def __repr__(self) -> str:
        return f"DistributedDevice(rank={self.rank}, local_rank={self.local_rank}, " \
            f"world_size={self.world_size}, local_world_size={self.local_world_size}, " \
            f"device={self.device})"


def unwrap_model(model: nn.Module | FSDP | DDP) -> nn.Module:
    unwrapped = model
    while isinstance(unwrapped, DDP):
        unwrapped = unwrapped.module
    return unwrapped


def get_optimizer_state_dict(
    model: nn.Module | FSDP | DDP,
    optimizer: optim.Optimizer
) -> dict[str, Any]:
    if isinstance(model, FSDP):
        return FSDP.optim_state_dict(
            model,
            optimizer
        )
    else:
        return optimizer.state_dict()
