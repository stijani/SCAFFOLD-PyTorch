import pickle
import random

import torch
from rich.progress import track
from tqdm import tqdm

from base import ServerBase
from client.scaffold import SCAFFOLDClient
#from config.util import clone_parameters, get_args
from config.util import clone_parameters
from config.options import CONFIG_CIFAR10, CONFIG_MNIST
import argparse
import os
from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy
import csv
from rich.console import Console
import numpy as np
import json


ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True, help="config object to use")
ap.add_argument("--device", required=True, type=int, help="gpu device to use")
ap.add_argument("--exp_name", required=True, type=str, help="summarised name for this experiment")
cmd_args = vars(ap.parse_args())

# configs
configs = {
    "cifar10": CONFIG_CIFAR10,
    "mnist": CONFIG_MNIST
}

config = configs[cmd_args["config"]]
config["gpu"] = cmd_args["device"]
config["exp_name"] = cmd_args["exp_name"]


class SCAFFOLDServer(ServerBase):
    def __init__(self, args):
        super(SCAFFOLDServer, self).__init__(args, "SCAFFOLD")
        self.trainer = SCAFFOLDClient(
            backbone=self.backbone,
            logger=self.logger,
            args=self.args
        )
        self.c_global = [
            torch.zeros_like(param).to(self.device)
            #for param in self.backbone(self.args["dataset"]).parameters()
            for param in self.backbone.parameters()
        ]
        self.global_lr = self.args["global_lr"]
        self.training_acc = [[] for _ in range(self.global_epochs)]

    def train(self):
        self.logger.log("=" * 30, "TRAINING", "=" * 30, style="bold green")
        progress_bar = (
            track(
                range(self.global_epochs),
                "[bold green]Training...",
                console=self.logger,
            )
            if not self.args["log"]
            else tqdm(range(self.global_epochs), "Training...")
        )
        self.logger.log("Arguments:", self.args)
        params = json.dumps(self.args)
        train_acc, train_loss = [params], [params]

        for E in progress_bar:
            if E % self.args["verbose_gap"] == 0:
                self.logger.log("=" * 30, f"ROUND: {E}", "=" * 30)

            selected_clients = random.sample(
                self.client_id_indices, self.args["client_num_per_round"]
            )
            res_cache = []
            for client_id in selected_clients:
                client_local_params = clone_parameters(self.global_params_dict)
                #print(client_local_params)
                res, stats = self.trainer.train(
                    client_id=client_id,
                    model_params=client_local_params,
                    c_global=self.c_global,
                    #verbose=(E % self.args["verbose_gap"]) == 0,
                    verbose=False,
                )

                res_cache.append(res)
                self.training_acc[E].append(stats["acc_before"])
            self.aggregate(res_cache)

            # this epoch's training has completed, let's test it with the global test data
            if E % self.args["global_test_period"] == 0:
                acc_, loss_ = self.test_global(E) # test current global model on the global test dataset
                train_acc.append(acc_)
                train_loss.append(loss_)

            # decay the lr if applicable at this step
            if self.trainer.args["lr_schedule_rate"]:
                self.trainer.scheduler.step()
        # create the metric directories
        metric_dir_acc = os.path.join(self.args["metric_file_dir"], "acc")
        metric_dir_loss = os.path.join(self.args["metric_file_dir"], "loss")
        os.makedirs(metric_dir_acc, exist_ok = True)
        os.makedirs(metric_dir_loss, exist_ok = True)
        # export results
        with open(os.path.join(metric_dir_acc, self.args["metric_filename"]), 'a') as f, open(os.path.join(metric_dir_loss, self.args["metric_filename"]), 'a') as g:
            writer1 = csv.writer(f)
            writer2 = csv.writer(g)
            writer1.writerow(train_acc)
            writer2.writerow(train_loss)

            # if E % self.args["save_period"] == 0 and self.args["save_period"] > 0:
            #     torch.save(
            #         self.global_params_dict, self.temp_dir / "global_model.pt",
            #     )
            #     with open(self.temp_dir / "epoch.pkl", "wb") as f:
            #         pickle.dump(E, f)

    def aggregate(self, res_cache):
        y_delta_cache = list(zip(*res_cache))[0]
        c_delta_cache = list(zip(*res_cache))[1]
        trainable_parameter = filter(
            lambda param: param.requires_grad, self.global_params_dict.values()
        )

        # update global model
        avg_weight = torch.tensor(
            [
                1 / self.args["client_num_per_round"]
                for _ in range(self.args["client_num_per_round"])
            ],
            device=self.device,
        )
        for param, y_del in zip(trainable_parameter, zip(*y_delta_cache)):
            x_del = torch.sum(avg_weight * torch.stack(y_del, dim=-1), dim=-1)
            param.data += self.global_lr * x_del

        # update global control
        for c_g, c_del in zip(self.c_global, zip(*c_delta_cache)):
            c_del = torch.sum(avg_weight * torch.stack(c_del, dim=-1), dim=-1)
            c_g.data += (
                self.args["client_num_per_round"] / len(self.client_id_indices)
            ) * c_del


if __name__ == "__main__":
    # hyperparameter tuning
    if config["tunable_params_vs_values"]:
        exp_name = config["exp_name"]
        for tunable_param, values in config["tunable_params_vs_values"].items():
            for value in values:
                config[tunable_param] = value
                config["exp_name"] = f"{exp_name}_{tunable_param}: {value}"
                server = SCAFFOLDServer(config)
                server.train()

    # one-off training
    else:
        server = SCAFFOLDServer(config)
        server.train()
