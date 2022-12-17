import os

project_dir = os.path.abspath(".")
base = "/home/stijani/projects"
sub_project = "phd/paper-2/benchmark-experiments-results" 
data_dir = f"{base}/dataset"
dataset = "cifar10"
num_clients = 10
client_frac = 1
niid = 1
processed_data = f"niid-{niid}-client-{num_clients}" if niid else f"iid-client-{num_clients}"
metric_filename = "fedavg"
metric_filename = "unbiased_update"
gpu = 2



CONFIG_CIFAR10 = {
    # experiment
    "run_summary": "code-test",
    "metric_file_dir": os.path.join(base, sub_project, dataset, processed_data),
    "metric_filename": metric_filename,
    "log_dir": f"{base}/phd/paper-2/test",
    "exp_output_base": f"{base}/phd/paper-2/test",
    "tmp_dir": f"{base}/phd/paper-2/tmp_dir",

    # data
    "dataset": dataset,
    "processed_data_dir": f"/home/stijani/projects/dataset/fl-datasets/{dataset}/{processed_data}",
    "global_test_data_dir": f"/home/stijani/projects/dataset/{dataset}",
    "valset_ratio": 0,
    "testset_ratio": 0,
    
    # local training
    "local_epochs": 100,
    "local_lr": 1e-3,
    "momentum": 0.9,
    "batch_size": 64,

    # unbiased approach
    "batch_size_unbiased_step": 4000,
    "beta": 0.8,

    # global training
    "global_epochs": 2000,
    "client_num_per_round": int(num_clients * client_frac),
    "global_test_period": 1,
    "save_period": 10000000,

    # others
    "verbose_gap": 10000000,
    "gpu": gpu,
    "log": 0,
    "seed": 17,
    
}

CONFIG_MNIST = {
    
}

