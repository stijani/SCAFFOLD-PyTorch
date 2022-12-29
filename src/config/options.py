import os

project_dir = os.path.abspath(".")
base = "/home/stijani/projects"
sub_project = "phd/paper-2/benchmark-experiments-results" 
data_dir = f"{base}/dataset"
dataset = "cifar10"
num_clients = 10
client_frac = 1
niid = None
processed_data = f"niid-{niid}-client-{num_clients}" if niid else f"iid-client-{num_clients}"
metric_filename = "learning_rate_decay_test.csv"
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
    "local_epochs": 100,
    "local_lr": 1e-3,
    # "momentum": 0.9,
    "momentum": None,
    "batch_size": 64,
    # set value only if we want to tune hyperparams, otherwise, set to None
    "tunable_params_vs_values": None, #{"local_lr": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]},

    # algorithm specific
    "batch_size_unbiased_step": 4000,
    "beta": 0.8,
    "fedprox_mu": 1.0,

    # global training
    "global_epochs": 1000,#2000,
    "lr_schedule_step": 250, # or None
    "lr_schedule_rate": 0.1, # or None
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

