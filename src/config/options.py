import os

project_dir = os.path.abspath(".")
base = "/home/stijani/projects" 
data_dir = f"{base}/dataset"
dataset = "cifar10"
data_partition = "dirichlet"



CONFIG_CIFAR10 = {
    # experiment
    "run_summary": "code-test",
    #"filename_acc": os.path.join(project_dir, "experiments_benchmark", dataset, data_partition, f"{exp_name}/acc/{metric_filename}.csv"), 
    #"filename_loss": os.path.join(project_dir, "experiments_benchmark", dataset, data_partition, "hyper-parameter-tuning/loss/proxy_data_perc.csv"),
    "log_dir": f"{base}/phd/paper-2/test",
    "exp_output_base": f"{base}/phd/paper-2/test",
    "tmp_dir": f"{base}/phd/paper-2/tmp_dir",

    # data
    "dataset": dataset,
    "processed_data_dir": f"/home/stijani/projects/dataset/fl-datasets/{dataset}/iid-client-10",
    "global_test_data_dir": f"/home/stijani/projects/dataset/{dataset}",
    "valset_ratio": 0,
    "testset_ratio": 0,
    
    # local training
    "local_epochs": 100,
    "local_lr": 1e-2,
    "batch_size": 32,

    # global training
    "global_epochs": 100,
    "client_num_per_round": 5,
    "global_test_period": 1,
    "save_period": 10000000,

    # others
    "verbose_gap": 10000000,
    "gpu": 2,
    "log": 0,
    "seed": 17,
    
}

CONFIG_MNIST = {
    # experiment
    "run_summary": "code-test",
    #"filename_acc": os.path.join(project_dir, "experiments_benchmark", dataset, data_partition, f"{exp_name}/acc/{metric_filename}.csv"), 
    #"filename_loss": os.path.join(project_dir, "experiments_benchmark", dataset, data_partition, "hyper-parameter-tuning/loss/proxy_data_perc.csv"),

    # data
    "dataset": "cifar10",
    "global_test_data_dir": "/home/stijani/projects/dataset/cifar10",
    "valset_ratio": 0.1,
    "testset_ratio": 0.1,
    
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

