import os

project_dir = os.path.abspath(".")
data_dir = "/home/stijani/projects/dataset"
dataset = "cifar10"
data_partition = "dirichlet"

config = {
    # experiment
    "run_summary": "code-test",
    #"filename_acc": os.path.join(project_dir, "experiments_benchmark", dataset, data_partition, f"{exp_name}/acc/{metric_filename}.csv"), 
    #"filename_loss": os.path.join(project_dir, "experiments_benchmark", dataset, data_partition, "hyper-parameter-tuning/loss/proxy_data_perc.csv"),

    # data
    "dataset": "cifar10",
    "global_test_data_dir": "/home/stijani/projects/dataset/cifar10",
    
    # local training
    "local_epochs": 100,
    "local_lr": 1e-2,
    "batch_size": 32,

    # global training
    "global_epochs": 100,
    "client_num_per_round": 10,
    "global_test_period": 1,
    "save_period": 10000000,

    # others
    "verbose_gap": 10000000,
    "gpu": 2,
    "log": 0,
    "seed": 17,
    
}