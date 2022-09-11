if __name__ == "__main__":
    # import itertools
    import subprocess
    import sys
    import os

    # script to execute
    pyscript = "/home/krober_f/biomass_pyrenees/scripts/ae_regression_net.py"

    # input dirs
    base_path = "/home/krober_f/biomass_pyrenees/data/point_tiles/splits_small"
    data_paths = [os.path.join(base_path, split) for split in os.listdir(base_path)]

    # create parent folder for saving all results
    save_dir = "/home/krober_f/biomass_pyrenees/results/exp_ae_regression"
    os.makedirs(save_dir, exist_ok=True)

    # define hyperparams to test
    lrs = ["0.01", "0.001"]
    optimisers = ["Adam", "AdamW"]
    losses = ["L1", "L2"]
    trials = 2

    # perform tests
    t_count = len(lrs) * len(optimisers) * len(losses) * len(data_paths) * trials
    count = 0
    for lr in lrs:
        for optimiser in optimisers:
            for loss in losses:
                for trial in range(trials):
                    for data_path in data_paths:
                        # define read paths
                        path_train = f"{data_path}/coverage/train"
                        path_val = f"{data_path}/coverage/val"
                        path_test = f"{data_path}/coverage/test"
                        path_ae = "/home/krober_f/biomass_pyrenees/results/exp_ae"
                        out_path = os.path.join(save_dir, f"mod_{count}")
                        out_model = "False"
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
