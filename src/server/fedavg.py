from base import ServerBase
from client.fedavg import FedAvgClient
from config.options import CONFIG_CIFAR10, CONFIG_MNIST
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True, help="config object to use")
cmd_args = vars(ap.parse_args())

# configs
configs = {
    "cifar10": CONFIG_CIFAR10,
    "mnist": CONFIG_MNIST
}

config = configs[cmd_args["config"]]


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
            momentum=self.args["momentum"],
            logger=self.logger,
            gpu=self.args["gpu"],
        )


if __name__ == "__main__":
    server = FedAvgServer()
    server.run()
