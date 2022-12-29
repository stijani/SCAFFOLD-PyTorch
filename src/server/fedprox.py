# from base import ServerBase
# from client.fedprox import FedProxClient
# from config.util import get_args

from base import ServerBase
from client.fedprox import FedProxClient
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


class FedProxServer(ServerBase):
    def __init__(self):
        super(FedProxServer, self).__init__(config, "FedProx")

        self.trainer = FedProxClient(
            # backbone=self.backbone(self.args.dataset),
            # dataset=self.args.dataset,
            # batch_size=self.args.batch_size,
            # valset_ratio=self.args.valset_ratio,
            # testset_ratio=self.args.testset_ratio,
            # local_epochs=self.args.local_epochs,
            # local_lr=self.args.local_lr,
            # logger=self.logger,
            # gpu=self.args.gpu,
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
            mu=self.args["fedprox_mu"],
            logger=self.logger,
            gpu=self.args["gpu"],
        )


if __name__ == "__main__":
    server = FedProxServer()
    server.run()
