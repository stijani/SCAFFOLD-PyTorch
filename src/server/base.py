import os
import pickle
import random
from argparse import Namespace
from collections import OrderedDict

import torch
# from path import Path
from rich.console import Console
from rich.progress import track
from tqdm import tqdm

#_CURRENT_DIR = Path(__file__).parent.abspath()

import sys

#sys.path.append(_CURRENT_DIR.parent)
sys.path.append("./src")
sys.path.append("./data")

from config.models import LeNet5
from config.util import (
    # DATA_DIR,
    LOG_DIR,
    OUTPUT_DIR,
    TEMP_DIR,
    clone_parameters,
    fix_random_seed,
)

#sys.path.append(OUTPUT_DIR)
#sys.path.append(DATA_DIR)
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
        self.temp_dir = os.path.join(TEMP_DIR, self.algo)
        if not os.path.isdir(self.temp_dir):
            os.makedirs(self.temp_dir)

        _dummy_model = self.backbone(self.args.dataset).to(self.device)
        passed_epoch = 0
        self.global_params_dict: OrderedDict[str : torch.Tensor] = None
        if os.listdir(self.temp_dir) != [] and self.args.save_period > 0:
            # if os.path.exists(self.temp_dir / "global_model.pt"):
            if os.path.exists(os.path.join(self.temp_dir, "global_model.pt")):
                self.global_params_dict = torch.load(os.path.join(self.temp_dir, "global_model.pt"))
                self.logger.log("Find existed global model...")

            if os.path.exists(os.path.join(self.temp_dir, "epoch.pkl")):
                with open(os.path.join(self.temp_dir, "epoch.pkl"), "rb") as f:
                    passed_epoch = pickle.load(f)
                self.logger.log(f"Have run {passed_epoch} epochs already.",)
        else:
            self.global_params_dict = OrderedDict(
                _dummy_model.state_dict(keep_vars=True)
            )

        self.global_epochs = self.args.global_epochs - passed_epoch
        self.logger.log("Backbone:", _dummy_model)

        self.trainer: ClientBase = None # trainer is of type CLientBase, e.g FedAvgClient
        self.training_acc = [[] for _ in range(self.global_epochs)]

    def train(self):
        """
           1. performs local training on each client's data, 
           2. aggregates clients' model weigths and 
           3. saves the aggregated weigths periodically
        """
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
                    # self.global_params_dict, self.temp_dir / "global_model.pt",
                    self.global_params_dict, os.path.join(self.temp_dir, "global_model.pt"),
                )
                # with open(self.temp_dir / "epoch.pkl", "wb") as f:
                with open(os.path.join(self.temp_dir, "epoch.pkl"), "wb") as f:
                    pickle.dump(E, f) # save the comms round number too

    @torch.no_grad()
    def aggregate(self, res_cache):
        updated_params_cache = list(zip(*res_cache))[0] # list of updated weight dict from each client
        weights_cache = list(zip(*res_cache))[1] # list of the number of data samples each client held (shouldn't have been named weights) 
        weight_sum = sum(weights_cache) # total number of samples across clients
        # weights = torch.tensor(weights_cache, device=self.device) / weight_sum
        weights = torch.tensor(weights_cache, device=self.device).float() / weight_sum # get the proportion of each client weights to be applied in aggragation e.g 0.1 for client 1

        aggregated_params = []

        for params in zip(*updated_params_cache): # get the same layer weights for each client
            aggregated_params.append(
                torch.sum(weights * torch.stack(params, dim=-1), dim=-1) # in parallel apply the weigth fract for wach client and sum the values for this layer
            )

        self.global_params_dict = OrderedDict(
            zip(self.global_params_dict.keys(), aggregated_params)
        )

    # def test(self) -> None:
    #     """
    #        1. Tests the curent global model on all clients
    #           validation or test dataset.
    #        2. prints out each client loss/acc as well as the average across all clients
    #        TODO: Create similar function to test the global model on the 
    #              global test dataset, call it `test_global()`
    #     """
    #     self.logger.log("=" * 30, "TESTING", "=" * 30, style="bold blue")
    #     all_loss = []
    #     all_acc = []
    #     for client_id in track(
    #         self.client_id_indices,
    #         "[bold blue]Testing...",
    #         console=self.logger,
    #         disable=self.args.log,
    #     ):
    #         client_local_params = clone_parameters(self.global_params_dict) # setting the client's model state to the current global model state
    #         stats = self.trainer.test(
    #             client_id=client_id, model_params=client_local_params,
    #         )

    #         self.logger.log(
    #             f"client [{client_id}]  [red]loss: {stats['loss']:.4f}    [magenta]accuracy: {stats['acc']:.2f}%"
    #         )
    #         all_loss.append(stats["loss"])
    #         all_acc.append(stats["acc"])

    #     self.logger.log("=" * 20, "RESULTS", "=" * 20, style="bold green")
    #     self.logger.log(
    #         "loss: {:.4f}    accuracy: {:.2f}%".format(
    #             sum(all_loss) / len(all_loss), sum(all_acc) / len(all_acc),
    #         )
    #     )

    #     # check convergence
    #     epoch_to_50 = 10000000
    #     epoch_to_60 = 10000000
    #     epoch_to_70 = 10000000
    #     epoch_to_80 = 10000000
    #     for E, acc_list in enumerate(self.training_acc):
    #         avg_acc = sum(acc_list) / len(acc_list)
    #         if avg_acc >= 80 and epoch_to_80 > E:
    #             self.logger.log(
    #                 "{} achieved 80% accuracy({:.2f}%) at epoch: {}".format(
    #                     self.algo, avg_acc, E
    #                 )
    #             )
    #             epoch_to_80 = E

    #         elif avg_acc >= 70 and epoch_to_70 > E and epoch_to_70 >= epoch_to_80:
    #             self.logger.log(
    #                 "{} achieved 70% accuracy({:.2f}%) at epoch: {}".format(
    #                     self.algo, avg_acc, E
    #                 )
    #             )
    #             epoch_to_70 = E

    #         elif avg_acc >= 60 and epoch_to_60 > E and epoch_to_60 >= epoch_to_70:
    #             self.logger.log(
    #                 "{} achieved 60% accuracy({:.2f}%) at epoch: {}".format(
    #                     self.algo, avg_acc, E
    #                 )
    #             )
    #             epoch_to_60 = E

    #         elif avg_acc >= 50 and epoch_to_50 > E and epoch_to_50 >= epoch_to_60:
    #             self.logger.log(
    #                 "{} achieved 50% accuracy({:.2f}%) at epoch: {}".format(
    #                     self.algo, avg_acc, E
    #                 )
    #             )
    #             epoch_to_50 = E

    def test(self, use_valset: bool=True) -> None:
        """
           1. Tests the curent global model on all clients
              validation or test dataset.
           2. prints out each client loss/acc as well as the average across all clients
           TODO: Create similar function to test the global model on the 
                 global test dataset, call it `test_global()`
        """
        self.logger.log("=" * 30, "TESTING", "=" * 30, style="bold blue")
        all_loss = []
        all_acc = []
        for client_id in track(
            self.client_id_indices,
            "[bold blue]Testing...",
            console=self.logger,
            disable=self.args.log,
        ):
            client_local_params = clone_parameters(self.global_params_dict) # setting the client's model state to the current global model state
            
            # to get the specific datasets for this client, we need to change the 
            # state of the client _id of the trainer to the curent clients id
            self.trainer.client_id = client_id
            self.trainer.get_client_local_dataset()
            data = self.trainer.valset if use_valset else selftrainer.testset

            stats = self.trainer.test(
                data, model_params=client_local_params,
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


    # def test_global(self):
    #     """TODO: write doc string
    #     """
    #     self.logger.log("=" * 30, "TESTING", "=" * 30, style="bold blue")
    #     client_local_params = clone_parameters(self.global_params_dict)
    #     stats = self.trainer.test(
    #         client_id=client_id, model_params=client_local_params,
    #     )

    def run(self):
        self.logger.log("Arguments:", dict(self.args._get_kwargs()))
        self.train()
        self.test()
        #self.test_global()
        if self.args.log:
            if not os.path.isdir(LOG_DIR):
                os.mkdir(LOG_DIR)
            self.logger.save_html(LOG_DIR / self.log_name)

        # delete all temporary files
        if os.listdir(self.temp_dir) != []:
            os.system(f"rm -rf {self.temp_dir}")
