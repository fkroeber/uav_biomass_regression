##### model B - part I (AE) #####

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
from torchvision.utils import save_image
from torchsummary import summary

import glob
from datetime import date, datetime
from inspect import getsource
from osgeo import gdal
from PIL import Image, ImageDraw, ImageFont
from textwrap import dedent

import pdb

# hyperparameters
config = {}
config["net"] = "single_conv_flat"
config["batch_size"] = 64
config["epochs"] = 500
config["lr"] = 1e-2
config["optimiser"] = "AdamW"
config["loss"] = "L1"
config["img_interval"] = 10

# read paths
small_test = False
if small_test:
    suffix = "_small"
else:
    suffix = ""
config["train_data"] = os.path.join(
    "c:\\Users\\felix\\Desktop\\internship_letg\\data\\biomass\\all_tiles",
    f"train{suffix}",
)
config["val_data"] = os.path.join(
    "c:\\Users\\felix\\Desktop\\internship_letg\\data\\biomass\\all_tiles",
    f"val{suffix}",
)
config["test_data"] = os.path.join(
    "c:\\Users\\felix\\Desktop\\internship_letg\\data\\biomass\\all_tiles",
    f"test{suffix}",
)
config["test_vis"] = os.path.join(
    "c:\\Users\\felix\\Desktop\\internship_letg\\data\\biomass\\all_tiles",
    "vis_reconstruct_tiles\\set_full",
)

# write paths
todays_date = "".join(str(date.today()).split("-")[1:])
config["out_path"] = os.path.join(
    "c:\\Users\\felix\\Desktop\\internship_letg\\results",
    f"experiments_{todays_date}",
)

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
class ae_model:
    def __init__(self, **kwargs):
        self.train_data = kwargs.get("path_train")
        self.val_data = kwargs.get("path_val")
        self.test_data = kwargs.get("path_test")
        self.test_data_vis = kwargs.get("path_test_vis")
        self.out_path = kwargs.get("out_path")
        self.img_interval = kwargs.get("save_interval")
        self.net = kwargs.get("net")
        self.batch_size = kwargs.get("batch_size")
        self.epochs = kwargs.get("epochs")
        self.lr = kwargs.get("lr")
        self.optimiser = kwargs.get("optimiser")
        self.loss = kwargs.get("loss")
        self.verbose = kwargs.get("verbose")

        self.out_log = os.path.join(self.out_path, "logs.txt")
        self.out_mod1 = os.path.join(self.out_path, "model_architecture.py")
        self.out_mod2 = os.path.join(self.out_path, "model_params.pt")
        self.out_mod3 = os.path.join(self.out_path, "model_trained.pt")
        self.out_lossplot = os.path.join(self.out_path, "loss_plot.png")
        self.out_imggif = os.path.join(self.out_path, "reconstruction.gif")
        self.out_imgs = os.path.join(self.out_path, "reconstructed_tiles")
        if not os.path.exists(self.out_imgs):
            os.makedirs(self.out_imgs)

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

        brightness_transf = BrightnessTransform(brs=[0.8, 0.9, 1, 1.1, 1.2])
        hue_transf = HueTransform(hue_shifts=[0.8, 0.9, 1, 1.1, 1.2])
        transf1 = transforms.Compose(
            [transforms.ToTensor(), brightness_transf, hue_transf]
        )
        transf2 = transforms.Compose([transforms.ToTensor()])

        # data loading
        self.ds_train = CustomImageFolder(self.train_data, transform=transf2)
        self.ds_val = CustomImageFolder(self.val_data, transform=transf2)
        self.ds_test = CustomImageFolder(self.test_data, transform=transf2)
        self.ds_test_vis = CustomImageFolder(self.test_data_vis, transform=transf2)

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
        self.dl_test_vis = DataLoader(self.ds_test_vis, batch_size=100)

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
        def single_conv(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 3, stride=stride, padding=1, bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )

        def single_conv_t(in_channels, out_channels, stride, out_padding, f_act="relu"):
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    3,
                    stride=stride,
                    output_padding=out_padding,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True) if f_act == "relu" else nn.Sigmoid(),
            )

        def double_conv(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 3, stride=stride, padding=1, bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def double_conv_t(in_channels, out_channels, stride, out_padding, f_act="relu"):
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    3,
                    stride=stride,
                    output_padding=out_padding,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    out_channels, out_channels, 3, padding=1, bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True) if f_act == "relu" else nn.Sigmoid(),
            )

        class ae(nn.Module):
            def forward(self, x):
                x = self.encoder(x)
                x = self.decoder(x)
                return x

        class ae_single_conv_flat(ae):
            def __init__(self, in_bands):
                super(ae_single_conv_flat, self).__init__()
                self.encoder = nn.Sequential(
                    single_conv(in_bands, 16, stride=2),
                    single_conv(16, 32, stride=2),
                    single_conv(32, 32, stride=2),
                )
                self.decoder = nn.Sequential(
                    single_conv_t(32, 32, stride=2, out_padding=1),
                    single_conv_t(32, 16, stride=2, out_padding=0),
                    single_conv_t(
                        16, in_bands, stride=2, out_padding=0, f_act="sigmoid"
                    ),
                )

        class ae_single_conv_deep(ae):
            def __init__(self, in_bands):
                super(ae_single_conv_deep, self).__init__()
                self.encoder = nn.Sequential(
                    single_conv(in_bands, 16, stride=2),
                    single_conv(16, 32, stride=2),
                    single_conv(32, 64, stride=2),
                    single_conv(64, 128, stride=2),
                )
                self.decoder = nn.Sequential(
                    single_conv_t(128, 64, stride=2, out_padding=1),
                    single_conv_t(64, 32, stride=2, out_padding=1),
                    single_conv_t(32, 16, stride=2, out_padding=0),
                    single_conv_t(
                        16, in_bands, stride=2, out_padding=0, f_act="sigmoid"
                    ),
                )

        class ae_double_conv_flat(ae):
            def __init__(self, in_bands):
                super(ae_double_conv_flat, self).__init__()
                self.encoder = nn.Sequential(
                    double_conv(in_bands, 16, stride=2),
                    double_conv(16, 32, stride=2),
                    double_conv(32, 32, stride=2),
                )
                self.decoder = nn.Sequential(
                    double_conv_t(32, 32, stride=2, out_padding=1),
                    double_conv_t(32, 16, stride=2, out_padding=0),
                    double_conv_t(
                        16, in_bands, stride=2, out_padding=0, f_act="sigmoid"
                    ),
                )

        class ae_double_conv_deep(ae):
            def __init__(self, in_bands):
                super(ae_double_conv_deep, self).__init__()
                self.encoder = nn.Sequential(
                    double_conv(in_bands, 16, stride=2),
                    double_conv(16, 32, stride=2),
                    double_conv(32, 64, stride=2),
                    double_conv(64, 128, stride=2),
                )
                self.decoder = nn.Sequential(
                    double_conv_t(128, 64, stride=2, out_padding=1),
                    double_conv_t(64, 32, stride=2, out_padding=1),
                    double_conv_t(32, 16, stride=2, out_padding=0),
                    double_conv_t(
                        16, in_bands, stride=2, out_padding=0, f_act="sigmoid"
                    ),
                )

        # write net architecture to external file
        src_code = getsource(ae_model.def_net)
        src_code = src_code[(len(src_code.split("\n")[0]) + 1) :]
        src_code = src_code.split("# write net architecture to external file")[0]
        src_code = "import torch.nn as nn\n\n" + dedent(src_code)
        with open(self.out_mod1, "w", encoding="utf-8") as f:
            f.write(src_code)

        # choose net architecture
        # self.ae_mod = locals()[f"ae_{self.net}"](self.img_shape[0])
        sys.path.append(os.path.dirname(self.out_mod1))
        exec(
            f"import {os.path.split(self.out_mod1)[-1].split('.py')[0]} as ae_architecture"
        )
        self.ae_mod = eval(f"ae_architecture.ae_{self.net}")(self.img_shape[0])

        # logs & info
        mod_summary = summary(
            self.ae_mod,
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
                patience=10,
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
        model = self.ae_mod.cuda()
        if self.loss == "L1":
            criterion = nn.L1Loss()
        elif self.loss == "L2":
            criterion = nn.MSELoss()
        if self.optimiser == "Adam":
            optimiser = torch.optim.Adam(model.parameters(), lr=self.lr)
        elif self.optimiser == "AdamW":
            optimiser = torch.optim.AdamW(model.parameters(), lr=self.lr)
        early_stopping = EarlyStopping(path=self.out_mod2)

        # intitialse acc metrics & logs
        metric_collection = metrics.MetricCollection(
            [
                metrics.MeanAbsoluteError(),
                metrics.MeanSquaredError(),
                metrics.SpectralAngleMapper(),
                metrics.PeakSignalNoiseRatio(),
                # metrics.StructuralSimilarityIndexMeasure(),
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
            for batch, (data, _) in enumerate(self.dl_train, 1):
                data = data.to("cuda", dtype=torch.float)
                # forward pass
                optimiser.zero_grad()
                output = model(data)
                # backward pass
                loss = criterion(output, data)
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
                for data, _ in self.dl_val:
                    data = data.to("cuda", dtype=torch.float)
                    output = model(data)
                    loss = criterion(output, data)
                    valid_loss.append(loss.item())
                    # batch accuracy
                    acc_metrics.update(output, data)
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

            # plot reconstructed tiles
            if epoch % self.img_interval == 0:
                with torch.no_grad():
                    for data, _ in self.dl_test_vis:
                        data = data.to("cuda", dtype=torch.float)
                        output = model(data)
                out = output.cpu().detach()
                imgs = out[:, :3, :, :]
                save_image(
                    imgs, os.path.join(self.out_imgs, f"epoch_{epoch}.png"), nrow=5
                )

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
        model.load_state_dict(torch.load(self.out_mod2))
        self.ae_mod_trained = model
        torch.save(self.ae_mod_trained, self.out_mod3)

        self.train_losses = train_losses
        self.valid_losses = valid_losses
        self.valid_accs = valid_accs

        # evaluation on test data
        model.eval()
        test_acc = metric_collection.to(torch.device("cuda", 0))
        with torch.no_grad():
            test_loss = []
            test_acc = []
            for data, _ in self.dl_test:
                data = data.to("cuda", dtype=torch.float)
                output = model(data)
                loss = criterion(output, data)
                test_loss.append(loss.item())
                # batch accuracy
                acc_metrics.update(output, data)
                test_acc_ = acc_metrics.compute()
                test_acc_ = {i: k.item() for i, k in test_acc_.items()}
                test_acc.append(test_acc_)
                acc_metrics.reset()
            test_loss = np.average(test_loss)
            test_acc = dict(pd.DataFrame(test_acc).mean())
            self.test_accs = {i: k.item() for i, k in test_acc.items()}

        # remove all tiles constructed for idxs exceeding min_loss_idx
        min_loss_epoch = self.valid_losses.index(min(self.valid_losses)) + 1
        tiles = os.listdir(self.out_imgs)
        tiles_epochs = [int(x.split("_")[1].split(".png")[0]) for x in tiles]
        rm_epochs = [x > min_loss_epoch for x in tiles_epochs]
        rm_tiles = [i for (i, excl) in zip(tiles, rm_epochs) if excl]
        for tile in rm_tiles:
            os.remove(os.path.join(self.out_imgs, tile))

        # plot expected & actual output of trained model
        with torch.no_grad():
            for data, _ in self.dl_test_vis:
                data = data.to("cuda", dtype=torch.float)
                output = model(data)
            out = output.cpu().detach()
            imgs_in = data[:, :3, :, :]
            imgs_out = out[:, :3, :, :]
            save_image(imgs_in, os.path.join(self.out_imgs, "actual_tiles.png"), nrow=5)
            save_image(
                imgs_out,
                os.path.join(self.out_imgs, f"epoch_{min_loss_epoch}.png"),
                nrow=5,
            )

        # logs & info
        epochs_log = ["\n".join(x) for x in [epochs_log]][0]
        self.logs["training"] = {
            "loss & accuracy evolution": epochs_log,
            "training time (total)": f"{td_format(t_total)}",
            "training time (epoch)": f"{t_epoch.seconds} seconds",
            "test loss (trained model)": f"{test_loss:.5f}",
        }
        self.logs["evaluation"] = {
            "accuracies (test data)": "\n\t".join(
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

        # create epoch animation
        def create_gif():
            if os.name == "nt":
                font = ImageFont.truetype("arial.ttf", 20)
            elif os.name == "posix":
                font = ImageFont.truetype("FreeMonoBold.ttf", 20)
            frames = [
                Image.open(image) for image in glob.glob(f"{self.out_imgs}/epoch*.png")
            ]
            if len(frames) > 0:
                frame_names = [
                    os.path.split(x.filename)[-1].split(".png")[0] for x in frames
                ]
                epochs_idx = np.argsort([int(x.split("_")[-1]) for x in frame_names])
                frames_draw = [ImageDraw.Draw(x) for x in frames]
                [
                    fr.text((10, 10), t, font=font)
                    for fr, t in zip(frames_draw, frame_names)
                ]
                frames = [frames[idx] for idx in epochs_idx]
                frame_one = frames[0]
            else:
                frames = []
                frame_one = []
            return frame_one, frames

        epoch_stats = plot_epoch_stats()
        epoch_stats.savefig(
            self.out_lossplot,
            bbox_inches="tight",
        )
        # try:
        #     reconst_gif, frames = create_gif()
        #     reconst_gif.save(
        #         self.out_imggif,
        #         format="GIF",
        #         append_images=frames,
        #         save_all=True,
        #         duration=250,
        #         loop=0,
        #     )
        # except AttributeError:
        #     pass

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
    import shutil

    parser = argparse.ArgumentParser(
        description="Train & Evaluate AE model for dimensionality reduction"
    )
    # paths
    parser.add_argument(
        "--path_train", default=config["train_data"], help="path to train dataset name "
    )
    parser.add_argument(
        "--path_val", default=config["val_data"], help="path to val dataset name "
    )
    parser.add_argument(
        "--path_test", default=config["test_data"], help="path to test dataset name "
    )
    parser.add_argument(
        "--path_test_vis",
        default=config["test_vis"],
        help="path to test dataset to be used for plotting reconstructed tiles",
    )
    # output params
    parser.add_argument(
        "--out_path",
        default=config["out_path"],
        help="path to save all results (model, plots, logs,...)",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=config["img_interval"],
        help="interval for saving images",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=config["verbose"],
        help="extended information returned during execution",
    )
    # model to use
    parser.add_argument(
        "--net", default=config["net"], help="ae net architecture to use"
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
    # parse args
    config = vars(parser.parse_args())

    # path creation
    shutil.rmtree(config["out_path"])
    if not os.path.exists(config["out_path"]):
        os.makedirs(config["out_path"])
    if len(os.listdir(config["out_path"])):
        raise OSError("Out dir isn't empty!")

    # ae model execution
    ae = ae_model(**config)
    ae.dataloading()
    ae.def_net()
    ae.train_eval()
    ae.summarise_train()
    ae.logging()
