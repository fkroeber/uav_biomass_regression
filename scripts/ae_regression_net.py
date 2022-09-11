##### model B - part II (AE-based regression) #####

### configurations ###
# imports
import os
import sys
import platform
import psutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchmetrics as metrics
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchsummary import summary

import glob
from datetime import date, datetime
from osgeo import gdal

import pdb

# hyperparameters & net parameters
config = {}
config["augment"] = False
config["dropout"] = 0
config["bottleneck"] = 4
config["batch_size"] = 128
config["epochs"] = 500
config["lr"] = 1e-3
config["optimiser"] = "AdamW"
config["loss"] = "L1"

# read paths
basepath = "c:/Users/felix/Desktop/internship_letg/data/biomass"

config["path_train"] = os.path.join(
    basepath,
    f"point_tiles/splits/split_1/coverage/train",
)
config["path_val"] = os.path.join(basepath, f"point_tiles/splits/split_1/coverage/val")
config["path_test"] = os.path.join(
    basepath,
    f"point_tiles/splits/split_1/coverage/test",
)
config["path_ae"] = "c:/Users/felix/Desktop/internship_letg/results/ae_trained_full"

# write paths
todays_date = "".join(str(date.today()).split("-")[1:])
config["out_path"] = os.path.join(
    "c:\\Users\\felix\\Desktop\\internship_letg\\results",
    f"experiments_{todays_date}",
)
config["out_model"] = True
config["verbose"] = True

### general helper functions ###
# function for formatting timing objects
def td_format(td_object):
    seconds = int(td_object.total_seconds())
    periods = [
        ("day", 60 * 60 * 24),
        ("hour", 60 * 60),
        ("minute", 60),
        ("second", 1),
    ]
    strings = []
    for period_name, period_seconds in periods:
        if seconds > period_seconds:
            period_value, seconds = divmod(seconds, period_seconds)
            has_s = "s" if period_value > 1 else ""
            strings.append("%s %s%s" % (period_value, period_name, has_s))

    return ", ".join(strings)


# function for pretty printing sizes in B
def format_bsize(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}{suffix}"


### main model ###
class regression_model:
    def __init__(self, **kwargs):
        self.train_data = kwargs.get("path_train")
        self.val_data = kwargs.get("path_val")
        self.test_data = kwargs.get("path_test")
        self.ae_path = kwargs.get("path_ae")
        self.out_path = kwargs.get("out_path")
        self.out_mod = kwargs.get("out_model")
        self.augment = kwargs.get("augment")
        self.dropout = kwargs.get("dropout")
        self.bottleneck = kwargs.get("bottleneck")
        self.batch_size = kwargs.get("batch_size")
        self.epochs = kwargs.get("epochs")
        self.lr = kwargs.get("lr")
        self.optimiser = kwargs.get("optimiser")
        self.loss = kwargs.get("loss")
        self.verbose = kwargs.get("verbose")

        self.out_log = os.path.join(self.out_path, "logs.txt")
        self.out_model = os.path.join(self.out_path, "model_trained.pt")
        self.out_lossplot = os.path.join(self.out_path, "loss_plot.png")

        # info & log
        sys_info = {}
        sys_info["Datetime"] = f"{datetime.now().replace(microsecond=0)}"
        sys_info["System OS"] = platform.uname().system
        sys_info["System name"] = platform.uname().node
        sys_info["CPU"] = platform.uname().machine
        sys_info["Physical cores"] = psutil.cpu_count(logical=False)
        sys_info["Total cores"] = psutil.cpu_count(logical=True)
        sys_info["GPU"] = torch.cuda.get_device_properties(torch.device("cuda", 0)).name
        sys_info["GPU memory"] = format_bsize(
            torch.cuda.get_device_properties(torch.device("cuda", 0)).total_memory
        )
        self.logs = dict()
        self.logs["system info"] = sys_info
        self.logs["config"] = config

    def dataloading(self):
        # custom loader for uint16 bit, multi-channel imgs
        class CustomImageFolder(ImageFolder):
            def __getitem__(self, index: int):
                path, target = self.samples[index]
                sample = self.raster_loader(path)
                if self.transform is not None:
                    sample = self.transform(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                return sample, target

            def raster_loader(self, path):
                ds = gdal.Open(path)
                arr = ds.ReadAsArray().astype("float32")
                arr = arr / (2 ** 16)
                arr = np.transpose(arr, (1, 2, 0))
                return arr

        # transforms
        class BrightnessTransform:
            def __init__(self, brs):
                self.brs = brs

            def __call__(self, x):
                br = np.random.choice(self.brs)
                x_adj = x * br
                x_adj = torch.where(x_adj > 1, 1, x_adj)
                return x_adj

        class HueTransform:
            def __init__(self, hue_shifts):
                self.hue_shifts = hue_shifts

            def __call__(self, x):
                hue_shift = np.random.choice(self.hue_shifts)
                band = np.random.choice(x.shape[0])
                x_adj = x
                x_adj[band, ...] = x[band, ...] * hue_shift
                x_adj = torch.where(x_adj > 1, 1, x_adj)
                return x_adj

        brightness_transf = BrightnessTransform(brs=[0.8, 1, 1.2])
        hue_transf = HueTransform(hue_shifts=[0.8, 1, 1.2])
        transf_plain = transforms.Compose([transforms.ToTensor()])
        transf_augment = transforms.Compose(
            [transforms.ToTensor(), brightness_transf, hue_transf]
        )

        # data loading
        if self.augment:
            self.ds_train = CustomImageFolder(self.train_data, transform=transf_augment)
        else:
            self.ds_train = CustomImageFolder(self.train_data, transform=transf_plain)
        self.ds_val = CustomImageFolder(self.val_data, transform=transf_plain)
        self.ds_test = CustomImageFolder(self.test_data, transform=transf_plain)

        # multiprocessing on windows doesnt work
        # turn it off as temporary fix
        # https://discuss.pytorch.org/t/dataloader-multiprocessing-error-cant-pickle-odict-keys-objects-when-num-workers-0/43951
        if os.name == "nt":
            num_workers = 0
            pin_memory = False
        elif os.name == "posix":
            num_workers = 4
            pin_memory = True

        self.dl_train = DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.dl_val = DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.dl_test = DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # logs & info
        self.img_shape = self.ds_train[0][0].shape
        n_train = len(self.ds_train)
        n_val = len(self.ds_val)
        n_test = len(self.ds_test)
        n = n_train + n_val + n_test
        split_ratio = f"{n_train/n:.0%} - {n_val/n:.0%} - {n_test/n:.0%}"
        n_batches = int(np.ceil(len(self.ds_train.samples) / self.batch_size))
        self.logs["data info"] = {
            "Total dataset size": n,
            "Train-Valid-Test split ratio": split_ratio,
            "Training dataset size": n_train,
            "Validation dataset size": n_val,
            "Test dataset size": n_test,
            "Tile shape": self.img_shape,
            "Number of training batches": n_batches,
        }

    def def_net(self):
        # load trained ae model
        sys.path.append(self.ae_path)
        ae = torch.load(os.path.join(self.ae_path, "model_trained.pt"))
        encoder = ae.encoder
        for param in encoder.parameters():
            param.requires_grad = False

        # define subsequent regression net
        def single_conv(
            in_channels, out_channels, dropout, kernel_size, stride, padding=1
        ):
            return nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Dropout2d(p=dropout),
            )

        class regression_cnn(nn.Module):
            def __init__(self, nlayers_bottleneck, dropout_rate):
                super(regression_cnn, self).__init__()
                n = nlayers_bottleneck
                p = dropout_rate
                self.cnn = nn.Sequential(
                    single_conv(32, n, kernel_size=1, stride=1, dropout=0),
                    single_conv(n, n * 2, kernel_size=3, stride=2, dropout=p),
                    single_conv(n * 2, n * 4, kernel_size=3, stride=2, dropout=p),
                    single_conv(n * 4, n * 8, kernel_size=3, stride=2, dropout=p),
                )
                self.regress = nn.Sequential(
                    nn.AvgPool2d(6), nn.Flatten(), nn.Linear(n * 8, 1), nn.Sigmoid()
                )

            def forward(self, x):
                x = self.cnn(x)
                x = self.regress(x)
                x = 100 * x
                return x

        # create stacked model
        class stacked_model(nn.Module):
            def __init__(self, nlayers_bottleneck, dropout_rate):
                super(stacked_model, self).__init__()
                self.ae = ae.encoder
                self.regression_cnn = regression_cnn(nlayers_bottleneck, dropout_rate)

            def forward(self, x):
                x = self.ae(x)
                x = self.regression_cnn(x)
                return x

        self.reg_mod = stacked_model(self.bottleneck, self.dropout)

        # logs & info
        mod_summary = summary(
            self.reg_mod,
            self.img_shape,
            col_names=("input_size", "output_size", "num_params", "kernel_size"),
            verbose=0,
        )
        self.logs["net"] = {"summary": str(mod_summary)}

    def train_eval(self):
        # early stopping
        # credits: https://github.com/Bjarten/early-stopping-pytorch
        class EarlyStopping:
            def __init__(
                self,
                patience=25,
                verbose=False,
                delta=0,
                path="checkpoint.pt",
                trace_func=print,
            ):
                self.patience = patience
                self.verbose = verbose
                self.counter = 0
                self.best_score = None
                self.early_stop = False
                self.val_loss_min = np.Inf
                self.delta = delta
                self.path = path
                self.trace_func = trace_func

            def __call__(self, val_loss, model):
                score = -val_loss
                if self.best_score is None:
                    self.best_score = score
                    self.save_checkpoint(val_loss, model)
                elif score < self.best_score + self.delta:
                    self.counter += 1
                    if self.verbose:
                        self.trace_func(
                            f"EarlyStopping counter: {self.counter} out of {self.patience}"
                        )
                    if self.counter >= self.patience:
                        self.early_stop = True
                else:
                    self.best_score = score
                    self.save_checkpoint(val_loss, model)
                    self.counter = 0

            def save_checkpoint(self, val_loss, model):
                torch.save(model.state_dict(), self.path)
                self.val_loss_min = val_loss

        # initialise training
        model = self.reg_mod.cuda()
        if self.loss == "L1":
            criterion = nn.L1Loss()
        elif self.loss == "L2":
            criterion = nn.MSELoss()
        if self.optimiser == "Adam":
            optimiser = torch.optim.Adam(model.parameters(), lr=self.lr)
        elif self.optimiser == "AdamW":
            optimiser = torch.optim.AdamW(model.parameters(), lr=self.lr)
        early_stopping = EarlyStopping(path=self.out_model)

        # intitialse acc metrics & logs
        metric_collection = metrics.MetricCollection(
            [
                metrics.MeanAbsoluteError(),
                metrics.MeanSquaredError(),
                metrics.PearsonCorrCoef(),
                metrics.SpearmanCorrCoef(),
                metrics.R2Score(),
            ]
        )
        acc_metrics = metric_collection.to(torch.device("cuda", 0))
        train_losses = []
        valid_losses = []
        valid_accs = []
        epochs_log = [" "]

        # perform training
        start_train = datetime.now()
        for epoch in range(1, self.epochs + 1):
            train_loss = []
            valid_loss = []
            valid_acc = []

            model.train()
            for batch, (data, labels_idxs) in enumerate(self.dl_train, 1):
                # get data & labels
                data = data.to("cuda", dtype=torch.float)
                labels = [float(self.ds_train.classes[x]) for x in labels_idxs]
                labels = torch.Tensor(labels).to("cuda", dtype=torch.float)
                # forward pass
                optimiser.zero_grad()
                output = model(data)
                # backward pass
                loss = criterion(output.view(-1), labels)
                loss.backward()
                # parameter update
                optimiser.step()
                # log
                train_loss.append(loss.item())
                # clean up
                del data, output, loss
                torch.cuda.empty_cache()

            model.eval()
            with torch.no_grad():
                for batch, (data, labels_idxs) in enumerate(self.dl_val, 1):
                    # get data & labels
                    data = data.to("cuda", dtype=torch.float)
                    labels = [float(self.ds_val.classes[x]) for x in labels_idxs]
                    labels = torch.Tensor(labels).to("cuda", dtype=torch.float)
                    # forward pass
                    output = model(data)
                    loss = criterion(output.view(-1), labels)
                    valid_loss.append(loss.item())
                    # batch accuracy
                    acc_metrics.update(output.view(-1), labels)
                    valid_acc_ = acc_metrics.compute()
                    valid_acc_ = {i: k.item() for i, k in valid_acc_.items()}
                    valid_acc.append(valid_acc_)
                    acc_metrics.reset()
                    # clean up
                    del data, output, loss
                    torch.cuda.empty_cache()

            # calculate average loss & acc across single epoch
            train_loss = np.average(train_loss)
            valid_loss = np.average(valid_loss)
            valid_acc = dict(pd.DataFrame(valid_acc).mean())
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)

            # logs & info
            epoch_digits = len(str(self.epochs))
            log_loss = (
                f"[{epoch:>{epoch_digits}}/{self.epochs:>{epoch_digits}}] "
                + f"train_loss: {train_loss:.5f}"
                + f" | valid_loss: {valid_loss:.5f}"
            )
            log_acc = " | ".join([f"{i}: {k:.5f}" for i, k in valid_acc.items()])
            if self.verbose:
                print(log_loss)
            epochs_log.append(log_loss + " | " + log_acc)

            # early_stopping
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # timing
        end_train = datetime.now()
        t_total = end_train - start_train
        t_epoch = t_total / len(train_losses)

        # load & save best model
        model.load_state_dict(torch.load(self.out_model))
        self.reg_mod_trained = model
        if not self.out_mod:
            os.remove(self.out_model)

        self.train_losses = train_losses
        self.valid_losses = valid_losses
        self.valid_accs = valid_accs

        # evaluation on test data
        model.eval()
        test_acc = metric_collection.to(torch.device("cuda", 0))
        with torch.no_grad():
            test_loss = []
            test_acc = []
            for batch, (data, labels_idxs) in enumerate(self.dl_test, 1):
                # get data & labels
                data = data.to("cuda", dtype=torch.float)
                labels = [float(self.ds_test.classes[x]) for x in labels_idxs]
                labels = torch.Tensor(labels).to("cuda", dtype=torch.float)
                # forward pass
                output = model(data)
                loss = criterion(output.view(-1), labels)
                test_loss.append(loss.item())
                # batch accuracy
                acc_metrics.update(output.view(-1), labels)
                test_acc_ = acc_metrics.compute()
                test_acc_ = {i: k.item() for i, k in test_acc_.items()}
                test_acc.append(test_acc_)
                acc_metrics.reset()
            test_loss = np.average(test_loss)
            test_acc = dict(pd.DataFrame(test_acc).mean())
            self.test_accs = {i: k.item() for i, k in test_acc.items()}

        # logs & info
        epochs_log = ["\n".join(x) for x in [epochs_log]][0]
        self.logs["training"] = {
            "loss & accuracy evolution": epochs_log,
            "training time (total)": f"{td_format(t_total)}",
            "training time (epoch)": f"{t_epoch.seconds} seconds",
            "test loss (trained model)": f"{test_loss:.5f}",
        }
        self.logs["evaluation"] = {
            "accuracies (test data)": "\n".join(
                [*[" "], *[f"{i}: {k:.5f}" for i, k in self.test_accs.items()]]
            ),
        }

    def summarise_train(self):
        def plot_epoch_stats():
            df_accs = pd.DataFrame(self.valid_accs)
            n_acc_measures = df_accs.shape[1]

            fig = plt.figure(
                figsize=(10, 5 + 3 * n_acc_measures), constrained_layout=True
            )
            fig.suptitle("Loss evolution")
            subfigs = fig.subfigures(
                nrows=2, ncols=1, height_ratios=[5, 3 * n_acc_measures]
            )

            loss_plot = subfigs[0]
            ax = loss_plot.subplots(nrows=1, ncols=1)
            ax.plot(
                np.arange(0.5, len(self.train_losses) + 0.5),
                self.train_losses,
                label="training loss",
            )
            ax.plot(
                np.arange(1, len(self.valid_losses) + 1),
                self.valid_losses,
                label="validation loss",
                color="orange",
            )
            minposs = self.valid_losses.index(min(self.valid_losses)) + 1
            if minposs < len(self.valid_losses):
                ax.axvline(
                    minposs,
                    linestyle="--",
                    color="r",
                    label="early stopping",
                )
            ax.set_ylabel("loss")
            ax.set_xlim(0, len(self.train_losses) + 1)
            ax.set_ylim(0, self.train_losses[0])
            ax.legend()

            acc_plots = subfigs[1]
            acc_plots.suptitle("Accuracy metrics")
            axs = acc_plots.subplots(nrows=n_acc_measures, ncols=1, sharex=True)

            for idx, ax in enumerate(axs):
                acc_meas = df_accs.columns[idx]
                acc_series = df_accs[acc_meas]
                ax.plot(np.arange(1, len(acc_series) + 1), acc_series, color="orange")
                ax.set_xlim(0, len(self.train_losses) + 1)
                ax.set_ylabel(acc_meas)

            return fig

        epoch_stats = plot_epoch_stats()
        epoch_stats.savefig(
            self.out_lossplot,
            bbox_inches="tight",
        )

    def logging(self):
        with open(self.out_log, "w", encoding="utf-8") as f:
            for k_l1, v_l1 in self.logs.items():
                f.write("--------")
                f.write(k_l1)
                f.write("--------")
                f.write("\n")
                for k_l2, v_l2 in v_l1.items():
                    f.write(f"{k_l2}: {v_l2}")
                    f.write("\n")
                f.write("\n")


### main script to execute ###
if __name__ == "__main__":
    import argparse
    from argparse import RawTextHelpFormatter

    parser = argparse.ArgumentParser(
        description="Train & evaluate AE-based regression model for biomass estimation",
        formatter_class=RawTextHelpFormatter,
    )
    # input paths
    parser.add_argument(
        "--path_train", default=config["path_train"], help="path to train dataset name "
    )
    parser.add_argument(
        "--path_val", default=config["path_val"], help="path to val dataset name "
    )
    parser.add_argument(
        "--path_test", default=config["path_test"], help="path to test dataset name "
    )
    parser.add_argument(
        "--path_ae",
        default=config["path_ae"],
        help="path to folder with...\n a.) a file called 'model_trained.pt' describing the trained AE model\n b.) a file called 'model_architecture.py' describing the net structure of the trained model",
    )
    # output params
    parser.add_argument(
        "--out_path",
        default=config["out_path"],
        help="path to save all results (model, plots, logs,...)",
    )
    parser.add_argument(
        "--out_model",
        type=str,
        choices=["True", "False"],
        default=str(config["out_model"]),
        help="saving trained model (parameters & architecture)",
    )
    parser.add_argument(
        "--verbose",
        type=str,
        choices=["True", "False"],
        default=str(config["verbose"]),
        help="extended information returned during execution",
    )
    # net parameters
    parser.add_argument(
        "--augment",
        type=str,
        choices=["True", "False"],
        default=config["augment"],
        help="augmenting data set via brightness & hue transform",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=config["dropout"],
        help="dropout rate applied in conv layers",
    )
    parser.add_argument(
        "--bottleneck",
        type=int,
        default=config["bottleneck"],
        help="number of layers in bottleneck",
    )
    # hyperparams
    parser.add_argument(
        "--batch_size", type=int, default=config["batch_size"], help="size of the batch"
    )
    parser.add_argument(
        "--epochs", type=int, default=config["epochs"], help="Maximal number of epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=config["lr"], help="Initial learning rate"
    )
    parser.add_argument(
        "--optimiser",
        type=str,
        default=config["optimiser"],
        choices=["Adam", "AdamW"],
        help="steepest gradient algorithm",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default=config["loss"],
        choices=["L1", "L2"],
        help="Loss function to use - L1 uses the absolute pixelwise error, L2 the squared one",
    )
    # parse args (also booleans)
    config = vars(parser.parse_args())
    for conf in ["out_model", "verbose", "augment"]:
        if config[conf] == "True":
            config[conf] = True
        else:
            config[conf] = False
    # path creation
    if not os.path.exists(config["out_path"]):
        os.makedirs(config["out_path"])
    if len(os.listdir(config["out_path"])):
        raise OSError("Out dir isn't empty!")
    # regression model execution
    rg = regression_model(**config)
    rg.dataloading()
    rg.def_net()
    rg.train_eval()
    rg.summarise_train()
    rg.logging()
