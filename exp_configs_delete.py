base = "/home/stijani/projects" 

config_test = {
    "exp_base_path": f"{base}/repos/SCAFFOLD-PyTorch",
    "data_base_path": f"{base}/dataset",
    "exp_output_base": f"{base}/phd/paper-2/test",
    "exp_name": "test",

    # data
    "data_args_file": f"{base}/dataset/fl_datasets/cifar10/args.json",
    "data_pickle_dir": f"{base}/dataset/fl_datasets/cifar10/pickles",
}


config_cifar10 = {
    "exp_base_path": f"{base}/repos/SCAFFOLD-PyTorch",
    "data_base_path": f"{base}/dataset",
    "exp_name": "test",
}

config_mnist = {
    "exp_base_path": f"{base}/repos/SCAFFOLD-PyTorch",
    "data_base_path": f"{base}/dataset",
    "exp_name": "test",
}