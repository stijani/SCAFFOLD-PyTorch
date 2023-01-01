from base import ServerBase
from client.fedavg import FedAvgClient
from config.options import CONFIG_CIFAR10, CONFIG_MNIST
import argparse

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


class FedAvgServer(ServerBase):
    def __init__(self, args):
        super(FedAvgServer, self).__init__(args, "FedAvg")
        self.trainer = FedAvgClient(
            backbone=self.backbone,
            logger=self.logger,
            args=self.args
        )


if __name__ == "__main__":
    # hyperparameter tuning
    if config["tunable_params_vs_values"]:
        exp_name = config["exp_name"]
        for tunable_param, values in config["tunable_params_vs_values"].items():
            for value in values:
                config[tunable_param] = value
                config["exp_name"] = f"{exp_name}_{tunable_param}: {value}"
                server = FedAvgServer(config)
                server.train()

    # one-off training
    else:
        server = FedAvgServer(config)
        server.train()
