import torch
from rich.console import Console

from .base import ClientBase


class FedAvgClient(ClientBase):
    def __init__(
        self,
        backbone: torch.nn.Module,
        dataset: str,
        processed_data_dir: str,
        batch_size: int,
        valset_ratio: float,
        testset_ratio: float,
        local_epochs: int,
        local_lr: float,
        logger: Console,
        gpu: int,
    ):
        super(FedAvgClient, self).__init__(
            backbone,
            dataset,
            processed_data_dir,
            batch_size,
            valset_ratio,
            testset_ratio,
            local_epochs,
            local_lr,
            logger,
            gpu,
        )
