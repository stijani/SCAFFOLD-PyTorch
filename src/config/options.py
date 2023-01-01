import os

project_dir = os.path.abspath(".")
base = "/home/stijani/projects"
sub_project = "phd/paper-2/benchmark-experiments-results" 
data_dir = f"{base}/dataset"
dataset = "cifar10"
num_clients = 10
client_frac = 1 #############
niid = None
processed_data = f"niid-{niid}-client-{num_clients}" if niid else f"iid-client-{num_clients}"
metric_filename = "hyperparam-tunning-beta-unbiased-update.csv" #"hyperparam-tunning-mu-fedprox.csv"
# gpu = 1



CONFIG_CIFAR10 = {
    # experiment
    "exp_name": "code-test", # will be overidden by cli arg
    "metric_file_dir": os.path.join(base, sub_project, dataset, processed_data),
    "metric_filename": metric_filename,
    "log_dir": f"{base}/phd/paper-2/test",
    "exp_output_base": f"{base}/phd/paper-2/test",
    "tmp_dir": f"{base}/phd/paper-2/tmp_dir",
    "init_weigth_file": "/home/stijani/projects/repos/SCAFFOLD-PyTorch/pretrained_weights/lenet5_init.pt",

    # data
    "dataset": dataset,
    "processed_data_dir": f"/home/stijani/projects/dataset/fl-datasets/{dataset}/{processed_data}",
    #"processed_data_dir": f"/home/stijani/projects/dataset/fl-datasets-test/{dataset}/{processed_data}",
    "global_test_data_dir": f"/home/stijani/projects/dataset/{dataset}",
    "valset_ratio": 0,
    "testset_ratio": 0,
    
    # local training
    "local_epochs": 50, ###############
    "local_lr": 1e-2, ################
    "unbiased_update_lr": 1e-1,
    # "momentum": 0.9,
    "momentum": None,
    "batch_size": 64,
    # set value only if we want to tune hyperparams, otherwise, set to None
    #"tunable_params_vs_values": {"local_lr": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]},
    #"tunable_params_vs_values": {"mu": [1.0, 0.5, 1.0, 10, 50, 100, 1000]},
    "tunable_params_vs_values": {"global_lr": [1.0, 0.5, 1.0, 10, 50, 100, 1000]},
    #"tunable_params_vs_values": None, #{"beta": [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]},

    # algorithm specific
    "batch_size_unbiased_step": 4000,
    "beta": 0.8,
    "mu": 10000.0,

    # global training
    "global_epochs": 1,#500,
    "global_lr": 1.0, # used only by scaffold
    "lr_schedule_step": None, #250, # or None
    "lr_schedule_rate": None, #0.1, # or None
    "client_num_per_round": int(num_clients * client_frac),
    "global_test_period": 1,
    "save_period": 10000000,

    # others
    "verbose_gap": 10000000,
    #"gpu": gpu, # gpu is suppied as a cli argument
    "log": 0,
    "seed": 17,
    
}

CONFIG_MNIST = {
    
}

