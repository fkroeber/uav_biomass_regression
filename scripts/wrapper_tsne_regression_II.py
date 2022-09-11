if __name__ == "__main__":
    # import itertools
    import subprocess
    import sys
    import os

    # script to execute
    pyscript = "/home/krober_f/biomass_pyrenees/scripts/tsne_regression_net.py"

    # define hyperparams
    lr = "0.001"
    optimiser = "AdamW"
    loss = "L1"
    net = "resnet"
    epochs = "50"
    out_model = "True"
    n_trials = 10

    # input data
    base_path = "/home/krober_f/biomass_pyrenees/data/point_tiles/splits"
    data_paths = [os.path.join(base_path, split) for split in os.listdir(base_path)]
    data_paths.sort()

    # create parent folder for saving all results
    save_dir = "/home/krober_f/biomass_pyrenees/results/final_tsne_regression"
    os.makedirs(save_dir, exist_ok=True)

    # perform trainings
    for var in ["coverage", "height", "volume"]:
        print(f"---training nets for var: {var}---")
        for split_path in data_paths:
            split_name = os.path.split(split_path)[-1]
            # define read paths
            path_train = os.path.join(split_path, "augmented", var, "train")
            path_val = os.path.join(split_path, "augmented", var, "val")
            path_test = os.path.join(split_path, "augmented", var, "test")
            print(f"\tdata set: {split_name}")
            # perform each training n_times
            for trial in range(n_trials):
                out_path = os.path.join(save_dir, split_name, var, f"mod_{trial}")
                # train & evaluate model
                try:
                    subprocess.check_output(
                        [
                            sys.executable,
                            pyscript,
                            "--path_train",
                            path_train,
                            "--path_val",
                            path_val,
                            "--path_test",
                            path_test,
                            "--out_path",
                            out_path,
                            "--out_model",
                            out_model,
                            "--net",
                            net,
                            "--lr",
                            lr,
                            "--loss",
                            loss,
                            "--optimiser",
                            optimiser,
                            "--epochs",
                            epochs,
                        ],
                        stderr=subprocess.STDOUT,
                    )
                except subprocess.CalledProcessError as e:
                    print(e.output.decode())
                # info
                print(f"\t\t{trial+1}/{n_trials} training(s) performed")
