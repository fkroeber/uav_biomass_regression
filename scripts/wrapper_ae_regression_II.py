if __name__ == "__main__":
    # import itertools
    import subprocess
    import sys
    import os

    # script to execute
    pyscript = "/home/krober_f/biomass_pyrenees/scripts/ae_regression_net.py"

    # create parent folder for saving all results
    save_dir = "/home/krober_f/biomass_pyrenees/results/exp_ae_regression_dropout"
    os.makedirs(save_dir, exist_ok=True)

    # define hyperparams
    augments = ["False", "True"]
    dropouts = ["0", "0.2"]
    bottleneck = "4"
    lr = "0.001"
    optimiser = "Adam"
    loss = "L1"
    n_trials = 5

    # input data
    vars = ["coverage", "height", "volume"]
    path_ae = "/home/krober_f/biomass_pyrenees/results/exp_ae"
    base_path = "/home/krober_f/biomass_pyrenees/data/point_tiles/splits"
    data_paths = [os.path.join(base_path, split) for split in os.listdir(base_path)]
    data_paths.sort()

    # create parent folder for saving all results
    save_dir = "/home/krober_f/biomass_pyrenees/results/final_ae_regression"
    os.makedirs(save_dir, exist_ok=True)
    out_model = "True"

    # perform trainings
    count = 0
    t_count = len(augments) * len(dropouts) * len(data_paths) * len(vars) * n_trials
    for dropout in dropouts:
        for augment in augments:
            for var in vars:
                for split_path in data_paths:
                    split_name = os.path.split(split_path)[-1]
                    # define read paths
                    path_train = os.path.join(split_path, var, "train")
                    path_val = os.path.join(split_path, var, "val")
                    path_test = os.path.join(split_path, var, "test")
                    # perform each training n_times
                    for trial in range(n_trials):
                        out_path = os.path.join(
                            save_dir,
                            f"drop_{dropout}_aug_{augment}",
                            split_name,
                            var,
                            f"mod_{trial}",
                        )
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
                                    "--path_ae",
                                    path_ae,
                                    "--out_path",
                                    out_path,
                                    "--out_model",
                                    out_model,
                                    "--augment",
                                    augment,
                                    "--bottleneck",
                                    bottleneck,
                                    "--dropout",
                                    dropout,
                                    "--lr",
                                    lr,
                                    "--loss",
                                    loss,
                                    "--optimiser",
                                    optimiser,
                                ],
                                stderr=subprocess.STDOUT,
                            )
                        except subprocess.CalledProcessError as e:
                            print(e.output.decode())
                        # info
                        count += 1
                        print(f"{count}/{t_count} trained & evaluated.")
