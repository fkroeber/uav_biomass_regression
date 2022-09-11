### imports & settings
import numpy as np
import pandas as pd
import os
import shutil
from tqdm import tqdm

import geopandas as gpd
import rasterio
from osgeo import gdal
from pointpats import PointPattern
from shapely.geometry import box

from imblearn.over_sampling import RandomOverSampler
from MulticoreTSNE import MulticoreTSNE as TSNE

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, models


### define input paths/dirs
base_path = "C:/Users/felix/Desktop/internship_letg/data/biomass"
config = {}
config["all_tiles_dir"] = os.path.join(base_path, "all_tiles", "train")
config["point_tiles_dir"] = os.path.join(base_path, "point_tiles")
config["save_dir"] = os.path.join(base_path, "point_tiles", "split_v1")
config["max_sample"] = 5000
config["balance"] = True
config["verbose"] = True


### main class
class data_generator:
    def __init__(self, **kwargs):
        # get arguments
        self.save_dir = kwargs.get("save_dir")
        self.all_tiles = kwargs.get("all_tiles")
        self.point_tiles = kwargs.get("point_tiles")
        self.verbose = kwargs.get("verbose")
        self.max_sample = kwargs.get("max_sample")
        self.balance = kwargs.get("balance")
        # generate save folder
        os.makedirs(self.save_dir, exist_ok=False)
        # create initial logs
        self.logs = dict()
        intro_desc = """
        The parent folder contains the intial train/val/test splits using tiles for which actual in-situ-measurements are available.
        The current folder 'augmented' contains synthesised data consisting of other tiles that are similar to the labelled ones. 
        The augmented data set sizes may vary for the individual variables 'coverage', 'height' and 'volume' as random oversampling
        ensuring approximately same class strength across all histogram bins is implemented during data generation. 'tiles_augmented.csv' 
        contains an overview on all tiles (actual in_situ-measurments & synthesised ones) with their names and (interpolated pseudo-)values. 
        This summary can be used to investigate the tsne-based data synthesis more closely. Note that this list is created prior to performing 
        the subsetting to achieve approximately equal class distributions. Hence, the tiles forming the test/val/train for each variable may 
        contain some of the tiles multiple times (random oversampling) while omitting others (capping at a maximum of 10k tiles).
        """
        self.logs["General structure"] = intro_desc
        self.logs["--- Settings ---"] = ""
        for par, val in kwargs.items():
            self.logs["--- Settings ---"] += f"{par}: {val}\n"

    # divide tiles into train/val/test
    def train_val_test_div(self):
        # print info
        step_msg = (
            f"\n--- Step I: Divide labelled tiles into train-val-test portions ---\n"
        )
        self.logs[step_msg] = ""
        if self.verbose:
            print(step_msg)
        # define params & dirs
        split_ratio = np.array([0.6, 0.2, 0.2])
        train_cnn = os.path.join(self.save_dir, "train")
        val_cnn = os.path.join(self.save_dir, "val")
        test_cnn = os.path.join(self.save_dir, "test")
        for fold in [train_cnn, val_cnn, test_cnn]:
            os.makedirs(fold)
        # get existing tiles & corresponding sites
        tiles = [
            x
            for x in os.listdir(os.path.join(self.point_tiles, "all"))
            if x[:3] != "CMB"
        ]
        sites = [
            "".join(x for x in t if not x.isdigit()).split(".tif")[0] for t in tiles
        ]
        tiles_df = pd.DataFrame({"tiles": tiles, "sites": sites})
        n_samples = len(tiles_df)
        # assign samples to portions
        for frac, portion in zip(split_ratio, ["train", "val", "test"]):
            # draw samples & assign to portion
            p = (frac * n_samples) / len(tiles_df)
            p = p if p <= 1 else 1
            tiles_samples = tiles_df.groupby("sites").sample(frac=p)
            for i, row in tiles_samples.iterrows():
                shutil.copy(
                    os.path.join(self.point_tiles, "all", row["tiles"]),
                    os.path.join(locals()[f"{portion}_cnn"]),
                )
            # exclude drawn samples from pool of tiles
            idxs = [x for x in tiles_df.index if x not in tiles_samples.index]
            tiles_df = tiles_df.loc[idxs, :]
        # get final split ratio
        sr = np.array(
            [
                len(os.listdir(train_cnn)) / len(tiles),
                len(os.listdir(val_cnn)) / len(tiles),
                len(os.listdir(test_cnn)) / len(tiles),
            ]
        )
        sr = f"{sr[0]:.0%}-{sr[1]:.0%}-{sr[2]:.0%}"
        # get count per site
        train_t, val_t, test_t = (
            os.listdir(train_cnn),
            os.listdir(val_cnn),
            os.listdir(test_cnn),
        )
        site_count = []
        for set in [train_t, val_t, test_t]:
            sites = [
                "".join(x for x in t if not x.isdigit()).split(".tif")[0] for t in set
            ]
            site_c = pd.Series(sites).value_counts()
            site_count.append(site_c)
        site_count = pd.concat(site_count, axis=1)
        site_count = site_count.rename(columns={0: "train", 1: "val", 2: "test"})
        # put train/val/test tiles into folders for each variable of interest
        height_cnn = os.path.join(self.save_dir, "height")
        volume_cnn = os.path.join(self.save_dir, "volume")
        coverage_cnn = os.path.join(self.save_dir, "coverage")
        point_meas = pd.read_csv(os.path.join(self.point_tiles, "meas_tiles.csv"))
        for set in [train_cnn, val_cnn, test_cnn]:
            set_name = os.path.split(set)[-1]
            for tile in os.listdir(set):
                jeton = tile.split(".tif")[0]
                in_situ_meas = point_meas[point_meas["JETON"] == jeton]
                dir_1 = os.path.join(
                    height_cnn,
                    set_name,
                    str(float(in_situ_meas["Hauteur moyenne vegetation (cm)"])),
                )
                dir_2 = os.path.join(
                    volume_cnn,
                    set_name,
                    str(float(in_situ_meas["Volume de la vegetation (m3)"])),
                )
                dir_3 = os.path.join(
                    coverage_cnn,
                    set_name,
                    str(float(in_situ_meas["Recouvrement de vegetation (%)"])),
                )
                os.makedirs(dir_1, exist_ok=True)
                os.makedirs(dir_2, exist_ok=True)
                os.makedirs(dir_3, exist_ok=True)
                shutil.copy(os.path.join(set, tile), os.path.join(dir_1))
                shutil.copy(os.path.join(set, tile), os.path.join(dir_2))
                shutil.copy(os.path.join(set, tile), os.path.join(dir_3))
            shutil.rmtree(set)
        # print info
        print_msg = (
            f"split ratio (train-val-test): {sr}\n"
            f"distribution across sites:\n{site_count}"
        )
        self.logs[step_msg] = print_msg
        if self.verbose:
            print(print_msg)

    # define modified resnet
    class resnet18_truncated(models.ResNet):
        def __init__(self, num_classes=1000, pretrained=True, **kwargs):
            super().__init__(
                block=models.resnet.BasicBlock,
                layers=[2, 2, 2, 2],
                num_classes=num_classes,
                **kwargs,
            )
            if pretrained:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                weights = models.ResNet18_Weights.verify(weights)
                self.load_state_dict(weights.get_state_dict(progress=True))

        def _forward_impl(self, x: torch.Tensor):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return x

    # custom image loader
    class CustomImageFolder(ImageFolder):
        def __getitem__(self, index: int):
            path, target = self.samples[index]
            sample = self.raster_loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            target = os.path.split(path)[-1].split(".tif")[0]
            return sample, target

        def raster_loader(self, path):
            ds = gdal.Open(path)
            arr = ds.ReadAsArray().astype("float32")
            arr = arr / (2 ** 16)
            arr = np.transpose(arr, (1, 2, 0))
            return arr

    # normalise to [0, 1] range
    def _norm_range(self, x):
        range = np.max(x) - np.min(x)
        shifted = x - np.min(x)
        return shifted / range

    # transforms to fit pre-trained resnet (normalised RGB input)
    def _transf(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x[:3, :, :]),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    # forward pass
    def _apply_model(self, model, dl):
        model.eval()
        with torch.no_grad():
            for batch, (data, labels) in tqdm(
                enumerate(dl), total=len(dl), disable=not self.verbose
            ):
                data = data.to("cuda", dtype=torch.float)
                output = model(data)
                output = output.view(output.shape[0], -1).cpu().numpy()
                if batch == 0:
                    features = output
                    tile_names = list(labels)
                else:
                    features = np.concatenate((features, output))
                    tile_names = [*tile_names, *list(labels)]
        return tile_names, features

    # compute resnet features
    def apply_resnet(self):
        # print info
        step_msg = f"\n\n--- Step II: Apply resnet to all tiles as preprocessing for evaluation of similarities ---\n"
        self.logs[step_msg] = ""
        if self.verbose:
            print(step_msg)
        # load data
        ds_meas = self.CustomImageFolder(
            os.path.join(self.save_dir, "coverage", "train"), transform=self._transf()
        )
        ds_tbc = self.CustomImageFolder(self.all_tiles, transform=self._transf())
        # print info
        print_msg = (
            f"{len(ds_meas)} labelled tiles with actual in situ measurements considered\n"
            f"{len(ds_tbc)} tiles with missing labels considered\n"
        )
        self.logs[step_msg] = print_msg
        if self.verbose:
            print(print_msg)
        # apply resnet
        dl_meas = DataLoader(ds_meas, batch_size=64)
        dl_tbc = DataLoader(ds_tbc, batch_size=64)
        self.labels_meas, self.df_meas = self._apply_model(
            self.resnet18_truncated().cuda(), dl_meas
        )
        self.labels_tbc, self.df_tbc = self._apply_model(
            self.resnet18_truncated().cuda(), dl_tbc
        )
        # print info
        print_msg = f"{self.df_meas.shape[1]} features were calculated for every tile"
        self.logs[step_msg] += print_msg
        if self.verbose:
            print("\n" + print_msg)

    # assigning in situ measurements to tiles if available
    def _in_situ_meas(self, in_situ_df, tile_label, var):
        if sum(in_situ_df["JETON"] == tile_label):
            return float(in_situ_df[var][in_situ_df["JETON"] == tile_label])
        else:
            return -1

    # get tiles that are similar to the existing tiles
    def get_similar_tiles(self):
        # print info
        step_msg = f"\n\n--- Step III: Labelling of tiles based on similarity assessment via t-SNE ---\n"
        self.logs[step_msg] = ""
        if self.verbose:
            print(step_msg)
        # perform tsne for actual data synthesis
        n_trials = 10
        tiles_syn = []
        tiles_insitu = []
        point_meas = pd.read_csv(os.path.join(self.point_tiles, "meas_tiles.csv"))
        # compose df by mixing known & unknown tiles (equal proportion)
        np.random.seed(42)
        r_ind = np.random.choice(
            np.arange(0, len(self.df_tbc)), size=len(self.df_tbc), replace=False
        )
        r_ind = [
            r_ind[x : x + len(self.df_meas)]
            for x in np.arange(0, len(self.df_tbc), len(self.df_meas))
        ]
        for batch in tqdm(
            r_ind,
            total=len(r_ind),
            desc="assigning pseudo-values to tiles",
            disable=not self.verbose,
        ):
            # get values
            df = np.concatenate((self.df_meas, self.df_tbc[batch]))
            labels = [*self.labels_meas, *[self.labels_tbc[x] for x in batch]]
            tsne_detailed = []
            tsne_summary = []
            for trial, seed in enumerate(range(40, 40 + n_trials)):
                # perform tsne
                tsne = TSNE(n_jobs=1, n_components=2, perplexity=25, random_state=seed)
                tse = tsne.fit_transform(df)
                tx = self._norm_range(tse[:, 0])
                ty = self._norm_range(tse[:, 1])
                # create comprehensive df with tile names, tsne components & in-situ-meaurements
                tsne_df = pd.DataFrame(
                    {
                        "tile": labels,
                        "in_situ": [
                            1 if sum(point_meas["JETON"] == x) else 0 for x in labels
                        ],
                        "height": [
                            self._in_situ_meas(
                                point_meas, x, "Hauteur moyenne vegetation (cm)"
                            )
                            for x in labels
                        ],
                        "volume": [
                            self._in_situ_meas(
                                point_meas, x, "Volume de la vegetation (m3)"
                            )
                            for x in labels
                        ],
                        "coverage": [
                            self._in_situ_meas(
                                point_meas, x, "Recouvrement de vegetation (%)"
                            )
                            for x in labels
                        ],
                        "trial": trial,
                        "tx": tx,
                        "ty": ty,
                    }
                )
                # calculate stats for point pattern
                pp = PointPattern(tsne_df[["tx", "ty"]])
                knn = list(pp.find_pairs(pp.mean_nnd))
                # metrics based on available in-situ measurements
                knn_meas = [
                    (tsne_df["coverage"][pair[0]], tsne_df["coverage"][pair[1]])
                    for pair in knn
                ]
                knn_meas = [pair for pair in knn_meas if -1 not in pair]
                knn_meas = np.array(knn_meas).transpose()
                r_corr = np.corrcoef(knn_meas)[0, 1]
                rmse = np.linalg.norm(knn_meas[0, :] - knn_meas[1, :]) / np.sqrt(
                    knn_meas.shape[1]
                )
                # calculate unknown values for new tiles as nn average
                idx_unclass = tsne_df.index[tsne_df["in_situ"] == 0]
                for idx in idx_unclass:
                    if idx in set([i for pair in knn for i in pair]):
                        pair_occurence = [idx in pair for pair in knn]
                        knn_pairs = [
                            pair for (pair, idx) in zip(knn, pair_occurence) if idx
                        ]
                        knn_ = list(set([i for pair in knn_pairs for i in pair]))
                        knn_ = [x for x in knn_ if x not in idx_unclass]
                        if len(knn_):
                            meas_avg = tsne_df.loc[knn_].mean(numeric_only=True).round()
                            tsne_df.loc[idx, "volume"] = meas_avg["volume"]
                            tsne_df.loc[idx, "coverage"] = meas_avg["coverage"]
                            tsne_df.loc[idx, "height"] = meas_avg["height"]
                # compile overall statistics
                tsne_stats_df = pd.Series(
                    {
                        "trial": trial,
                        "kl_div": tsne.kl_divergence_,
                        "mean_nn_dist": pp.mean_nnd,
                        "r": r_corr,
                        "rmse": rmse,
                    }
                )
                # append stats
                tsne_detailed.append(tsne_df)
                tsne_summary.append(tsne_stats_df)
            # concat dfs
            tsne_detailed = pd.concat(tsne_detailed)
            tsne_summary = pd.DataFrame(tsne_summary)
            # select t-SNE embedding based on low kl divergence & rmse values
            tsne_summary["rank"] = (
                tsne_summary.rank()["kl_div"] + tsne_summary.rank()["rmse"]
            )
            best_trial = int(tsne_summary.sort_values("rank").head(1)["trial"])
            best_tsne = tsne_detailed[tsne_detailed["trial"] == best_trial]
            assigned_vals = best_tsne[
                (best_tsne["in_situ"] == 0) & (best_tsne["coverage"] != -1)
            ]
            # append to df
            tiles_syn.append(assigned_vals)
            tiles_insitu.append(best_tsne[best_tsne["in_situ"] == 1])
        # compile all synthesisable & in_situ tiles in one df
        self.tiles_syn = pd.concat(tiles_syn)
        self.tiles_insitu = pd.concat(tiles_insitu).drop_duplicates("tile")
        all_tiles = pd.concat([self.tiles_insitu, self.tiles_syn])
        all_tiles = all_tiles[["tile", "in_situ", "height", "volume", "coverage"]]
        all_tiles.to_csv(os.path.join(self.save_dir, "tiles_augmented.csv"))
        # print info
        print_msg = f"{self.tiles_syn.shape[0]} tiles were labelled and can be used as synthesised data"
        self.logs[step_msg] = print_msg
        if self.verbose:
            print("\n" + print_msg)

    # check class balance
    def _subset_uniform_classhist(self, df, var):
        range = np.linspace(df[var].min(), df[var].max(), 6)
        steps = [
            (range[idx], range[idx + 1])
            for idx, _ in enumerate(range)
            if idx + 1 < len(range)
        ]

        def categorise_range(x):
            for idx, step in enumerate(steps):
                if (x >= step[0]) and (x <= step[1]):
                    return idx

        df = df.assign(
            **{f"{var}_cat": lambda df: [categorise_range(x) for x in df[var]]}
        )
        df = df.reset_index(drop=True)
        X, y = np.array(df.index).reshape(-1, 1), df[f"{var}_cat"]
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X, y)
        df = pd.concat([df[df.index == x] for x in X_res.reshape(-1)])
        return df

    def assign_similar_tiles(self):
        # print info
        if self.balance:
            step_msg = f"\n\n--- Step IV: Create augmented data sets based on synthesised data considering class balance ---\n"
        else:
            step_msg = f"\n\n--- Step IV: Create augmented data sets based on synthesised data ---\n"
        self.logs[step_msg] = ""
        if self.verbose:
            print(step_msg)
        # copy relevant tiles for each variable of interest
        for var in ["coverage", "height", "volume"]:
            # get synthesised tiles to be copied
            if self.balance:
                tiles_syn = self._subset_uniform_classhist(self.tiles_syn, var)
            else:
                tiles_syn = self.tiles_syn
            if len(tiles_syn) > self.max_sample:
                tiles_syn = tiles_syn.sample(self.max_sample)
            # copy existing folder structure into subfolder
            sub_dir = os.path.join(self.save_dir, "augmented")
            shutil.copytree(
                os.path.join(self.save_dir, var), os.path.join(sub_dir, var)
            )
            # augmenting training data set with synthesised/labelled one
            for i, row in tiles_syn.iterrows():
                src_tile = os.path.join(self.all_tiles, "uav", f"{row['tile']}.tif")
                dest_dir = os.path.join(
                    sub_dir, var, "train", str(row[var]), f"{row['tile']}.tif"
                )
                os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
                file_exist = True
                ind = 0
                while file_exist:
                    if os.path.exists(dest_dir):
                        ind += 1
                        dest_dir = os.path.join(
                            sub_dir,
                            var,
                            "train",
                            str(row[var]),
                            f"{row['tile']}_{ind}.tif",
                        )
                    else:
                        file_exist = False
                        shutil.copy(src_tile, dest_dir)
            # print info
            print_msg = f"{tiles_syn.shape[0]} synthesised tiles added to the original ones for variable '{var}'"
            self.logs[step_msg] += print_msg + "\n"
            if self.verbose:
                print(print_msg)

    def remove_overlapping_tiles(self):
        # print info
        step_msg = f"\n--- Step V: Remove augmented tiles spatially overlapping the validation & training samples ---\n"
        self.logs[step_msg] = ""
        if self.verbose:
            print("\n" + step_msg)
        # create mask based on val & test portion
        mask = []
        for path, subdirs, files in os.walk(
            os.path.join(self.save_dir, "augmented", "coverage")
        ):
            for name in files:
                if os.path.split(os.path.split(path)[0])[-1] in ["val", "test"]:
                    tile_path = os.path.join(path, name)
                    tile_crs = rasterio.open(tile_path).crs
                    tile_bbox = rasterio.open(tile_path).bounds
                    tile_bbox = gpd.GeoSeries(
                        box(tile_bbox[0], tile_bbox[1], tile_bbox[2], tile_bbox[3])
                    )
                    mask.append(tile_bbox)
        mask = gpd.GeoDataFrame(pd.concat(mask, ignore_index=True))
        mask = mask.set_geometry(0)
        mask = mask.set_crs(tile_crs)
        # remove ovlp tiles from each variables training portions
        for var in ["coverage", "height", "volume"]:
            rm_count = 0
            for path, subdirs, files in os.walk(
                os.path.join(self.save_dir, "augmented", var)
            ):
                for name in files:
                    if os.path.split(os.path.split(path)[0])[-1] == "train":
                        tile_path = os.path.join(path, name)
                        tile_crs = rasterio.open(tile_path).crs
                        tile_bbox = rasterio.open(tile_path).bounds
                        tile_bbox = gpd.GeoSeries(
                            box(tile_bbox[0], tile_bbox[1], tile_bbox[2], tile_bbox[3])
                        )
                        tile_bbox = (
                            gpd.GeoDataFrame(tile_bbox)
                            .set_geometry(0)
                            .set_crs(tile_crs)
                        )
                        if len(mask.overlay(tile_bbox, how="intersection")):
                            os.remove(tile_path)
                            if not len(os.listdir(os.path.dirname(tile_path))):
                                shutil.rmtree(os.path.dirname(tile_path))
                            rm_count += 1
            # print info
            print_msg = f"{rm_count} tiles removed for training set '{var}' due to spatial overlap"
            self.logs[step_msg] += print_msg + "\n"
            if self.verbose:
                print(print_msg)

    def summary(self):
        # print info
        step_msg = f"\n--- Summary ---\n"
        self.logs[step_msg] = ""
        if self.verbose:
            print("\n" + step_msg)
        print_msg = "variable: training samples (original) -> training samples (incl. synthesised)"
        self.logs[step_msg] += print_msg + "\n"
        if self.verbose:
            print(print_msg)
        # count number of tiles
        for var in ["coverage", "height", "volume"]:
            orig_samples = os.path.join(self.save_dir, var, "train")
            synt_samples = os.path.join(self.save_dir, "augmented", var, "train")
            s_orig = []
            for path, subdirs, files in os.walk(orig_samples):
                for file in files:
                    s_orig.append(file)
            s_synt = []
            for path, subdirs, files in os.walk(synt_samples):
                for file in files:
                    s_synt.append(file)
            print_msg = f"{var}: {len(s_orig)} -> {len(s_synt)}"
            self.logs[step_msg] += print_msg + "\n"
            if self.verbose:
                print(print_msg)
        # write logs
        with open(
            os.path.join(self.save_dir, "augmented", "logs.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            for k_l1, v_l1 in self.logs.items():
                f.write(k_l1)
                f.write("\n")
                f.write(v_l1)
                f.write("\n")


if __name__ == "__main__":
    import argparse
    from argparse import RawTextHelpFormatter

    parser = argparse.ArgumentParser(
        description="Generate augmented train/val/test data sets for biomass cnn",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--all_tiles",
        type=str,
        default=config["all_tiles_dir"],
        help="path to the set of tiles to be used to augment the existing tiles",
    )
    parser.add_argument(
        "--point_tiles",
        type=str,
        default=config["point_tiles_dir"],
        help="path to folder containing... \n...one subfolder called 'all' containing all tiles for which reliable in-situ-measurements actually exist\n...one file called 'meas_tiles.csv' containing corresponding in situ measurements for these tiles",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=config["save_dir"],
        help="save path for splitted & augmented data sets",
    )
    parser.add_argument(
        "--balance",
        type=str,
        choices=["True", "False"],
        default=str(config["balance"]),
        help="perform random oversampling to balance values",
    )
    parser.add_argument(
        "--max_sample",
        type=int,
        default=config["max_sample"],
        help="approximate upper limit for the number of tiles generated",
    )
    parser.add_argument(
        "--verbose",
        type=str,
        choices=["True", "False"],
        default=str(config["verbose"]),
        help="extended information returned during execution",
    )

    # parse args
    config = vars(parser.parse_args())
    # convert boolean arguments
    if config["balance"] == "True":
        config["balance"] = True
    else:
        config["balance"] = False
    if config["verbose"] == "True":
        config["verbose"] = True
    else:
        config["verbose"] = False

    # main
    data_gen = data_generator(**config)
    data_gen.train_val_test_div()
    data_gen.apply_resnet()
    data_gen.get_similar_tiles()
    data_gen.assign_similar_tiles()
    data_gen.remove_overlapping_tiles()
    data_gen.summary()
