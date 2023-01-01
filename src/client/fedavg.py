import torch
from rich.console import Console

from .base import ClientBase


class FedAvgClient(ClientBase):
    def __init__(
        self,
        backbone: torch.nn.Module,
        logger: Console,
        args
    ):
        super(FedAvgClient, self).__init__(
            backbone,
            logger,
            args
        )
        self.args = self.arg_dict 
