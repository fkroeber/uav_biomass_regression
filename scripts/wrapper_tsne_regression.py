if __name__ == "__main__":
    # import itertools
    import subprocess
    import sys
    import os

    # script to execute
    pyscript = "/home/krober_f/biomass_pyrenees/scripts/tsne_regression_net.py"

    # create parent folder for saving all results
    save_dir = "/home/krober_f/biomass_pyrenees/results/exp_tsne_regression"
    os.makedirs(save_dir, exist_ok=True)

    # define hyperparams to test
    lrs = ["0.01", "0.001"]
    optimisers = ["Adam", "AdamW"]
    losses = ["L1", "L2"]
    nets = ["resnet", "efficientnet"]
    trials = 5

    # perform tests
    count = 0
    for lr in lrs:
        for optimiser in optimisers:
            for loss in losses:
                for net in nets:
                    for trial in range(trials):
                        # define read paths
                        path_train = f"/home/krober_f/biomass_pyrenees/data/point_tiles/splits_small/split_{trial+1}/augmented/coverage/train"
                        path_val = f"/home/krober_f/biomass_pyrenees/data/point_tiles/splits_small/split_{trial+1}/augmented/coverage/val"
                        path_test = f"/home/krober_f/biomass_pyrenees/data/point_tiles/splits_small/split_{trial+1}/augmented/coverage/test"
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
                                ],
                                stderr=subprocess.STDOUT,
                            )
                        except subprocess.CalledProcessError as e:
                            print(e.output.decode())
                        # info
                        count += 1
                        print(
                            f"Hyperparameter configuration {count} trained & evaluated."
                        )
