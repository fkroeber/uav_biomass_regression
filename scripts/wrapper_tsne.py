if __name__ == "__main__":
    # import itertools
    import subprocess
    import sys
    import os

    # script to execute
    pyscript = "/home/krober_f/biomass_pyrenees/scripts/tsne_data_synthesis.py"

    # number of test/train/val splits
    n_splits = 10

    # perform tests
    for i in range(n_splits):
        all_tiles_dir = "/home/krober_f/biomass_pyrenees/data/all_tiles/train"
        point_tiles_dir = "/home/krober_f/biomass_pyrenees/data/point_tiles"
        save_dir = (
            f"/home/krober_f/biomass_pyrenees/data/point_tiles/splits_small/split_{i+1}"
        )
        max_sample = "1000"
        try:
            subprocess.check_output(
                [
                    sys.executable,
                    pyscript,
                    "--all_tiles",
                    all_tiles_dir,
                    "--point_tiles",
                    point_tiles_dir,
                    "--save_dir",
                    save_dir,
                    "--max_sample",
                    max_sample,
                ],
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            print(e.output.decode())
