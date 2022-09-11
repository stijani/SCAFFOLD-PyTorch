import os
import pickle
import random
from argparse import Namespace
from collections import OrderedDict

import torch
from path import Path
from rich.console import Console
from rich.progress import track
from tqdm import tqdm

_CURRENT_DIR = Path(__file__).parent.abspath()

import sys

sys.path.append(_CURRENT_DIR.parent)

from config.models import LeNet5
from config.util import (
    DATA_DIR,
    LOG_DIR,
    PROJECT_DIR,
    TEMP_DIR,
    clone_parameters,
    fix_random_seed,
)

sys.path.append(PROJECT_DIR)
sys.path.append(DATA_DIR)
from client.base import ClientBase
from data.utils.util import get_client_id_indices


class ServerBase:
    def __init__(self, args: Namespace, algo: str):
        self.algo = algo
        self.args = args
        # default log file format
        self.log_name = "{}_{}_{}_{}.html".format(
            self.algo,
            self.args.dataset,
            self.args.global_epochs,
            self.args.local_epochs,
        )
        self.device = torch.device(
            "cuda" if self.args.gpu and torch.cuda.is_available() else "cpu"
        )
        fix_random_seed(self.args.seed)
        self.backbone = LeNet5
        self.logger = Console(record=True, log_path=False, log_time=False,)
        self.client_id_indices, self.client_num_in_total = get_client_id_indices(
            self.args.dataset
        )
        self.temp_dir = TEMP_DIR / self.algo
        if not os.path.isdir(self.temp_dir):
            os.makedirs(self.temp_dir)

        _dummy_model = self.backbone(self.args.dataset).to(self.device)
        passed_epoch = 0
        self.global_params_dict: OrderedDict[str : torch.Tensor] = None
        if os.listdir(self.temp_dir) != []:
            if os.path.exists(self.temp_dir / "global_model.pt"):
                self.global_params_dict = torch.load(self.temp_dir / "global_model.pt")
                self.logger.log("Find existed global model...")

            if os.path.exists(self.temp_dir / "epoch.pkl"):
                with open(self.temp_dir / "epoch.pkl", "rb") as f:
                    passed_epoch = pickle.load(f)
                self.logger.log(f"Have run {passed_epoch} epochs already.",)
        else:
            self.global_params_dict = OrderedDict(
                _dummy_model.state_dict(keep_vars=True)
            )

        self.global_epochs = self.args.global_epochs - passed_epoch
        self.logger.log("Backbone:", _dummy_model)

        self.trainer: ClientBase = None
        self.training_acc = [[] for _ in range(self.global_epochs)]

    def train(self):
        self.logger.log("=" * 30, "TRAINING", "=" * 30, style="bold green")
        progress_bar = (
            track(
                range(self.global_epochs),
                "[bold green]Training...",
                console=self.logger,
            )
            if not self.args.log
            else tqdm(range(self.global_epochs), "Training...")
        )
        for E in progress_bar:

            if E % self.args.verbose_gap == 0:
                self.logger.log("=" * 30, f"ROUND: {E}", "=" * 30)

            selected_clients = random.sample(
                self.client_id_indices, self.args.client_num_per_round
            )
            res_cache = []
            for client_id in selected_clients:
                client_local_params = clone_parameters(self.global_params_dict)
                res, stats = self.trainer.train(
                    client_id=client_id,
                    model_params=client_local_params,
                    verbose=(E % self.args.verbose_gap) == 0,
                )

                res_cache.append(res)
                self.training_acc[E].append(stats["acc_before"])
            self.aggregate(res_cache)

            if E % self.args.save_period == 0:
                torch.save(
                    self.global_params_dict, self.temp_dir / "global_model.pt",
                )
                with open(self.temp_dir / "epoch.pkl", "wb") as f:
                    pickle.dump(E, f)

    @torch.no_grad()
    def aggregate(self, res_cache):
        updated_params_cache = list(zip(*res_cache))[0]
        weights_cache = list(zip(*res_cache))[1]
        weight_sum = sum(weights_cache)
        weights = torch.tensor(weights_cache, device=self.device) / weight_sum

        aggregated_params = []

        for params in zip(*updated_params_cache):
            aggregated_params.append(
                torch.sum(weights * torch.stack(params, dim=-1), dim=-1)
            )

        self.global_params_dict = OrderedDict(
            zip(self.global_params_dict.keys(), aggregated_params)
        )

    def test(self) -> None:
        self.logger.log("=" * 30, "TESTING", "=" * 30, style="bold blue")
        all_loss = []
        all_acc = []
        for client_id in track(
            self.client_id_indices,
            "[bold blue]Testing...",
            console=self.logger,
            disable=self.args.log,
        ):
            client_local_params = clone_parameters(self.global_params_dict)
            stats = self.trainer.test(
                client_id=client_id, model_params=client_local_params,
            )

            self.logger.log(
                f"client [{client_id}]  [red]loss: {stats['loss']:.4f}    [magenta]accuracy: {stats['acc']:.2f}%"
            )
            all_loss.append(stats["loss"])
            all_acc.append(stats["acc"])

        self.logger.log("=" * 20, "RESULTS", "=" * 20, style="bold green")
        self.logger.log(
            "loss: {:.4f}    accuracy: {:.2f}%".format(
                sum(all_loss) / len(all_loss), sum(all_acc) / len(all_acc),
            )
        )

        # check convergence
        epoch_to_50 = 10000000
        epoch_to_60 = 10000000
        epoch_to_70 = 10000000
        epoch_to_80 = 10000000
        for E, acc_list in enumerate(self.training_acc):
            avg_acc = sum(acc_list) / len(acc_list)
            if avg_acc >= 80 and epoch_to_80 > E:
                self.logger.log(
                    "{} achieved 80% accuracy({:.2f}%) at epoch: {}".format(
                        self.algo, avg_acc, E
                    )
                )
                epoch_to_80 = E

            elif avg_acc >= 70 and epoch_to_70 > E and epoch_to_70 >= epoch_to_80:
                self.logger.log(
                    "{} achieved 70% accuracy({:.2f}%) at epoch: {}".format(
                        self.algo, avg_acc, E
                    )
                )
                epoch_to_70 = E

            elif avg_acc >= 60 and epoch_to_60 > E and epoch_to_60 >= epoch_to_70:
                self.logger.log(
                    "{} achieved 60% accuracy({:.2f}%) at epoch: {}".format(
                        self.algo, avg_acc, E
                    )
                )
                epoch_to_60 = E

            elif avg_acc >= 50 and epoch_to_50 > E and epoch_to_50 >= epoch_to_60:
                self.logger.log(
                    "{} achieved 50% accuracy({:.2f}%) at epoch: {}".format(
                        self.algo, avg_acc, E
                    )
                )
                epoch_to_50 = E

    def run(self):
        self.logger.log("Arguments:", dict(self.args._get_kwargs()))
        self.train()
        self.test()
        if self.args.log:
            if not os.path.isdir(LOG_DIR):
                os.mkdir(LOG_DIR)
            self.logger.save_html(LOG_DIR / self.log_name)

        # delete all temporary files
        if os.listdir(self.temp_dir) != []:
            os.system(f"rm -rf {self.temp_dir}")
