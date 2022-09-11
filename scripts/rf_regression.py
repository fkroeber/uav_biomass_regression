##### model A (RF-based regression) #####

# load standard libraries
import os
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm

# load rf & metrics libraries
from sklearn.ensemble import RandomForestRegressor
from torch import Tensor
import torchmetrics as metrics

# standard settings
config = {}
config[
    "path_feats_df"
] = "/home/krober_f/biomass_pyrenees/results/final_rf_regression/feature_df.csv"
config[
    "out_path"
] = "/home/krober_f/biomass_pyrenees/results/final_rf_regression/test_accs_only_spectral.csv"
config["path_splits"] = "/home/krober_f/biomass_pyrenees/data/point_tiles/splits"
config["feats"] = "only_spectral"
config["n_trials"] = 100

# rf regressor
class rf_base:
    def __init__(self, **kwargs):
        # get configuration
        self.path_feats_df = kwargs.get("path_feats_df")
        self.feats = kwargs.get("feats")
        self.path_split = kwargs.get("path_split")
        self.var = kwargs.get("var")

    def get_train_val_test(self):
        # read train-val-test split
        tile_splits = []
        split_path_var = os.path.join(self.path_split, self.var)
        for path, subdirs, files in os.walk(split_path_var):
            if len(files):
                for file in files:
                    portion = os.path.split(os.path.dirname(path))[1]
                    tile_name = file.split(".tif")[0]
                    tile_splits.append(
                        pd.Series({"portion": portion, "tile_name": tile_name})
                    )
        self.ts = pd.DataFrame(tile_splits)
        self.train_ts = self.ts[self.ts["portion"] == "train"]["tile_name"]
        self.val_ts = self.ts[self.ts["portion"] == "val"]["tile_name"]
        self.test_ts = self.ts[self.ts["portion"] == "test"]["tile_name"]
        # get features & labels
        self.feats_df = pd.read_csv(self.path_feats_df)
        feats = self.feats_df.columns[
            [
                x not in ["tile_name", "height", "volume", "coverage"]
                for x in self.feats_df.columns
            ]
        ]
        if self.feats == "only_spectral":
            feats = feats[["share" in x for x in feats]]
        # get features & labels for each portion
        train_idxs = self.feats_df["tile_name"].isin(self.train_ts)
        val_idxs = self.feats_df["tile_name"].isin(self.val_ts)
        test_idxs = self.feats_df["tile_name"].isin(self.test_ts)
        self.X_train = self.feats_df[train_idxs][feats]
        self.X_val = self.feats_df[val_idxs][feats]
        self.X_test = self.feats_df[test_idxs][feats]
        self.y_train = self.feats_df[train_idxs][self.var]
        self.y_val = self.feats_df[val_idxs][self.var]
        self.y_test = self.feats_df[test_idxs][self.var]

    def train_predict(self, seed):
        self.mod = RandomForestRegressor(random_state=seed).fit(
            self.X_train, self.y_train
        )
        self.y_pred_val = self.mod.predict(self.X_val)
        self.y_pred_test = self.mod.predict(self.X_test)

    def evaluate(self):
        warnings.filterwarnings("ignore", category=UserWarning)
        accs = metrics.MetricCollection(
            [
                metrics.MeanAbsoluteError(),
                metrics.MeanSquaredError(),
                metrics.PearsonCorrCoef(),
                metrics.SpearmanCorrCoef(),
                metrics.R2Score(),
            ]
        )
        # compute val metrics
        accs.update(Tensor(self.y_pred_val), Tensor(np.array(self.y_val)))
        self.val_acc = accs.compute()
        self.val_acc = {k: float(v.numpy()) for k, v in self.val_acc.items()}
        accs.reset()
        # compute test metrics
        accs.update(Tensor(self.y_pred_test), Tensor(np.array(self.y_test)))
        self.test_acc = accs.compute()
        self.test_acc = {k: float(v.numpy()) for k, v in self.test_acc.items()}
        accs.reset()


### main script to execute ###
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Perform random forest regression across multiple train-val-test splits"
    )
    # in paths
    parser.add_argument(
        "--path_feats_df",
        default=config["path_feats_df"],
        help="path to df with features & labels",
    )
    parser.add_argument(
        "--path_splits",
        default=config["path_splits"],
        help="path to folder with several train-val-test splits",
    )
    # rf parameters
    parser.add_argument(
        "--feats",
        default=config["feats"],
        choices=["all", "only_spectral"],
        help="features to be used for rf regression",
    )
    parser.add_argument(
        "--n_trials",
        default=config["n_trials"],
        type=int,
        help="number of runs per model configuration - eval on test is done on the one with highest val acc",
    )
    # out paths
    parser.add_argument(
        "--out_path",
        default=config["out_path"],
        help="path to save results as a csv file",
    )
    # parse args
    config = vars(parser.parse_args())

    # perform rf regression for each split & var
    splits_res = {}
    s_splits = os.listdir(config["path_splits"])
    for s_split in tqdm(s_splits):
        config["path_split"] = os.path.join(config["path_splits"], s_split)
        split_nr = os.path.split(config["path_split"])[1]
        for var in ["coverage", "height", "volume"]:
            config["var"] = var
            rf = rf_base(**config)
            rf.get_train_val_test()
            # perform rf for each setting multiple times
            runs = []
            for seed in np.arange(0, config["n_trials"]):
                rf.train_predict(seed)
                rf.evaluate()
                run_accs = pd.Series(
                    {
                        "seed": seed,
                        "acc_val": rf.val_acc["MeanAbsoluteError"],
                        "acc_test": rf.test_acc["MeanAbsoluteError"],
                    }
                )
                runs.append(run_accs)
            runs = pd.DataFrame(runs)
            best_mod_idx = int(runs.iloc[runs.idxmin()["acc_val"], :]["seed"])
            # get best result per run
            rf.train_predict(best_mod_idx)
            rf.evaluate()
            # write accuracies to summarising df
            splits_res[f"{split_nr}_{var}"] = rf.test_acc

    splits_res = pd.DataFrame(splits_res).reset_index()
    splits_res = splits_res.melt(id_vars="index")
    splits_res["split"] = [x.rsplit("_", 1)[0] for x in splits_res["variable"]]
    splits_res["variable"] = [x.rsplit("_", 1)[1] for x in splits_res["variable"]]
    splits_res.pivot(
        index=["variable", "split"], columns="index", values="value"
    ).reset_index()
    splits_res.to_csv(config["out_path"], index=False)
