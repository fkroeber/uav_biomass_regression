# -*- coding: utf-8 -*-
# modified version of geopatch package (https://github.com/Hejarshahabi/GeoPatch)
# changes implemented
# enable loading of vrt (not only .tif)
# speed-up assessing number of patches to be created (load metadata instead of img_data)
# processes adapted/shortened to data without labels
# masking option implemented (for non-rectangular tifs)
# resampling option implemented
# rescaling/hist stretching option implemented

import os
import numpy as np
import rasterio as rs
import rasterio.plot as rp
import tqdm
from contextlib import contextmanager
from rasterio import Affine, MemoryFile
from rasterio.enums import Resampling
from skimage import exposure


class Generator:
    """
    Using the package, you will be able to turn
    satellite imagery into patches with various shape sizes
    and apply resampling and masking on the fly

    parameters:
    -----------
    image:              String that shows path to imagery or opened Datareader object

    patch_size:         Integer
                        moving window size to generate image patches (in m)

    stride:             Integer
                        Percentage of overlap

    channel_first:      Boolean
                        True if the first index is the number of image bands and False if it is the last index.

    out_res:            Integer
                        Specifies the resolution of the patches. Default is same as input.

    """

    def __init__(
        self, image=None, patch_size=None, stride=None, out_res=None, channel_first=True
    ):

        # read image input
        if type(image) == rs.io.DatasetReader:
            self.img = image
            self.in_res = self.img.meta["transform"][0]
        elif type(image) == str and (image[-3:] == "tif" or image[-3:] == "vrt"):
            self.image = image
            self.in_res = rs.open(self.image).meta["transform"][0]
        else:
            raise ValueError("Input datareader nor path to image.")

        # read other parameters
        self.patch_size = int(round(patch_size / self.in_res))
        self.stride = int(round((1 - stride) * self.patch_size))
        self.channel = channel_first
        self.out_res = out_res

        # check if out put transformation necessary
        if self.out_res:
            if not np.isclose(self.in_res, self.out_res, rtol=0.01):
                self.resamp = True
                self.resamp_factor = self.in_res / self.out_res
                # self.patch_size = int(round(self.patch_size * self.resamp_factor))
                # self.stride = int(round(self.stride * self.resamp_factor))
                self.patch_size = int(
                    round((patch_size / self.in_res) * self.resamp_factor)
                )
                self.stride = int(
                    round(
                        ((1 - stride) * (patch_size / self.in_res)) * self.resamp_factor
                    )
                )
                print(
                    f"Resampling will be applied as desired resolution doesn't match the resolution of the input data\n"
                    f"Input resolution: {round(self.in_res,5)}\n"
                    f"Output resolution: {round(self.out_res,5)}\n"
                )
            else:
                self.resamp = False
        else:
            self.resamp = False

    # credits: https://gis.stackexchange.com/questions/329434/creating-an-in-memory-rasterio-dataset-from-numpy-array/329439#329439
    @contextmanager
    def resample_raster(self):
        # read data
        if hasattr(self, "img"):
            raster = self.img
        else:
            raster = rs.open(self.image)
        # adjust metadata
        t = raster.transform
        transform = Affine(
            t.a / self.resamp_factor, t.b, t.c, t.d, t.e / self.resamp_factor, t.f
        )
        height = int(round(raster.height * self.resamp_factor))
        width = int(round(raster.width * self.resamp_factor))
        profile = raster.profile
        profile.update(transform=transform, driver="GTiff", height=height, width=width)
        # read & resample data
        data = raster.read(
            out_shape=(raster.count, height, width),
            resampling=Resampling.bilinear,
        )
        # write data
        with MemoryFile() as memfile:
            with memfile.open(**profile) as dataset:
                dataset.write(data)
                del data

            with memfile.open() as dataset:
                yield dataset

    def readData(self, rescale=False):
        # perform resampling
        if self.resamp:
            print(f"Reading & Resampling the data...")
            with self.resample_raster() as resampled:
                self.img = resampled
                self.imgarr = self.img.read()
        else:
            print(f"Reading the data...")
            if not hasattr(self, "img"):
                self.img = rs.open(self.image)
            self.imgarr = self.img.read()

        self.band = self.img.count
        self.row = self.img.width
        self.col = self.img.height

        # perform contrast stretching
        print(f"Performing contrast stretching...")
        if rescale:
            if not hasattr(self, "rescale_stats"):
                self.rescale_stats = np.zeros((2, 4))
                for band in range(0, self.imgarr.shape[0]):
                    self.rescale_stats[0, band] = np.percentile(
                        self.imgarr[band, :, :], 0.1
                    )
                    self.rescale_stats[1, band] = np.percentile(
                        self.imgarr[band, :, :], 99.9
                    )

            img_arr_transp = np.transpose(self.imgarr, (1, 2, 0))
            for band in range(0, img_arr_transp.shape[2]):
                pu, po = self.rescale_stats[0, band], self.rescale_stats[1, band]
                img_arr_transp[..., band] = exposure.rescale_intensity(
                    img_arr_transp[..., band], in_range=(pu, po)
                )

            self.imgarr = np.transpose(img_arr_transp, (2, 0, 1))

        return self.img, self.imgarr

    def readMetaData(self):
        self.band = rs.open(self.image).meta["count"]
        self.row = rs.open(self.image).meta["width"]
        self.col = rs.open(self.image).meta["height"]
        if self.resamp:
            self.row = int(round(self.row * self.resamp_factor))
            self.col = int(round(self.col * self.resamp_factor))

        return self.band, self.row, self.col

    def data_dimension(self):
        self.readMetaData()
        print(f"The shape of (resampled) image is: {self.band, self.row, self.col} \n")

    def patch_info(self, verbose=True):
        self.readMetaData()
        x_dim = self.col
        y_dim = self.row

        self.X = ((x_dim - self.patch_size) // self.stride) * self.stride + 1
        self.Y = ((y_dim - self.patch_size) // self.stride) * self.stride + 1
        self.total_patches = (((x_dim - self.patch_size) // self.stride) + 1) * (
            ((y_dim - self.patch_size) // self.stride) + 1
        )

        if verbose:
            print(
                f"The number of total non-augmented patches that can be generated\n"
                f"based on patch size ({self.patch_size}*{self.patch_size}) and stride ({self.stride}) is {self.total_patches}\n"
            )

    def write_tile(self, name, transform):
        # write data
        with rs.open(
            name,
            "w",
            driver="GTiff",
            count=self.band,
            dtype=self.imgarr.dtype,
            width=self.patch_size,
            height=self.patch_size,
            transform=transform,
            crs=self.img.crs,
        ) as raschip:
            raschip.write(self.img_patch)

    def save_Geotif(
        self,
        folder_name="tiles",
        suffix=None,
        eval_mask=False,
        mask_val=0,
        thres=0.01,
        rescale=False,
        silent=False,
    ):
        """
        parameters:
        -----------
        folder_name:        String
                            Passing the folder name as string like "tif" so a folder
                            with that name will be generated in the current working directory
                            to save Geotif image patches there in the sub-folders called "patch" and "label"

        suffix:             String
                            If specified, these suffixes will be used for naming the single tiles. Should be unique,
                            otherwise current tile will be overwritten in each iteration.

        eval_mask:          Boolean
                            If True, tiles will be checked for mask_vals. If the the total percentage
                            of pixels equal to mask_val in all channels exceed thres, tiles will not be written to disk

        rescale:            Boolean
                            If True, spectral rescaling in terms of contrast stretching will be applied. This is done equally
                            for all the patches based on the stats calculated for the whole scene (.tif/.vrt).

        """
        # read data
        self.readData(rescale=rescale)
        self.patch_info(verbose=False)
        # create folder
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        # start patching
        print(f"Start patching...")
        # pdb.set_trace()
        with tqdm.tqdm(
            range(self.total_patches),
            desc="Patch Counter",
            unit=" Patch",
            disable=silent,
        ) as pbar:
            index = 1
            patch_counter = 0
            for i in range(0, self.X, self.stride):
                for j in range(0, self.Y, self.stride):
                    self.img_patch = self.imgarr[
                        :, i : i + self.patch_size, j : j + self.patch_size
                    ]
                    self.x_cord, self.y_cord = (
                        j * self.img.transform[0] + self.img.transform[2]
                    ), (self.img.transform[5] + i * self.img.transform[4])
                    transform = [
                        self.img.transform[0],
                        0,
                        self.x_cord,
                        0,
                        self.img.transform[4],
                        self.y_cord,
                    ]
                    # create patch name
                    if suffix:
                        patch_name = folder_name + "/" + suffix
                    else:
                        patch_name = folder_name + "/" + str(index) + "_img.tif"
                    # evaluate if patch should be written according to mask
                    if eval_mask == True:
                        if self.img_patch.size > 0:
                            inval_pxls = (self.img_patch.sum(axis=0) == mask_val).sum()
                            inval_pxls_prop = inval_pxls / (
                                self.img_patch.size / self.img_patch.shape[0]
                            )
                            if inval_pxls_prop < thres:
                                self.write_tile(patch_name, transform)
                                patch_counter += 1
                    # if no mask is specified write patch in any case
                    else:
                        self.write_tile(patch_name)
                        patch_counter += 1
                    # update counter
                    index += 1
                    pbar.update(1)
            total_patches = pbar.total
            save_patches = patch_counter
            percentage = int((save_patches / total_patches) * 100)
            # print infos
            if eval_mask == True:
                print(
                    f"Masking tiles with more than {thres:.1%} invalid values reduced amount of created patches."
                )
            print(
                f'{len(os.listdir(folder_name))} patches are saved as ".tif" format in "{folder_name}"\n'
            )
