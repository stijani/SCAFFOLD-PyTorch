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
    def __init__(self):
        super(FedAvgServer, self).__init__(config, "FedAvg")
        self.trainer = FedAvgClient(
            backbone=self.backbone,
            dataset=self.args["dataset"],
            processed_data_dir = self.args["processed_data_dir"],
            batch_size=self.args["batch_size"],
            valset_ratio=self.args["valset_ratio"],
            testset_ratio=self.args["testset_ratio"],
            local_epochs=self.args["local_epochs"],
            local_lr=self.args["local_lr"],
            lr_schedule_step=self.args["lr_schedule_step"],
            lr_schedule_rate=self.args["lr_schedule_rate"],
            momentum=self.args["momentum"],
            logger=self.logger,
            gpu=self.args["gpu"],
        )


if __name__ == "__main__":
    server = FedAvgServer()
    server.run()
