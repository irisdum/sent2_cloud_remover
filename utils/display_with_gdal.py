import glob
import os

import numpy as np
from osgeo import gdal

from utils.converter import convert_array
from utils.display_image import plot_gray, plot_sent2
from utils.old_dispay_vi import histo_val


def plot_allbands_hist(path_tif, ax):
    raster = gdal.Open(path_tif)
    image = raster.ReadAsArray()
    image = np.moveaxis(image, 0, -1)
    if ax is None:
        fig, ax = plt.subplots()
    ax.hist(image.ravel(), bins=256, color='orange', )
    nb = image.shape[-1]
    if nb == 5:
        nb -= 1
    l_legend = []
    l_color = ["red", "green", "blue", "gray", "pink"]
    for b in range(nb):
        ax.hist(image[:, :, b].ravel(), bins=256, color=l_color[b], alpha=0.5)
        l_legend += ["Bande {}".format(b + 1)]
        ax.set_xlabel('Intensity Value')
        ax.set_ylabel('Count')
        ax.legend(l_legend)
    if ax is None:
        plt.show()


def open_sentinel2(path_s2_dir, bands=None):
    if bands is None:
        bands = ["B4", "B3", "B2", "B8"]
    l_arr = []
    for b in bands:
        lpath_img = glob.glob(path_s2_dir + "{}*.img".format(b))
        assert len(lpath_img) > 0, "no image found at {}".format(path_s2_dir + "{}*.img".format(b))
        path_img = lpath_img[0]
        l_arr += open_array(path_img)
    return np.array(l_arr)


def open_array(path_img):
    """

    Args:
        path_img: a string path to the image

    Returns:
     a numpy array
    """
    assert os.path.isfile(path_img),"No file found at this path {}".format(path_img)
    raster = gdal.Open(path_img, gdal.GA_ReadOnly)
    return raster.ReadAsArray()


def display_image(path_image, mode=None, name_image=None, bound_x=None, bound_y=None, band=0, cm_band=True, ax=None):
    """
    :param path_image:
    :param mode:
    :param name_image:
    :param bound_x:
    :param bound_y:
    :param band: useful only if mode GRAY
    :return:
    """
    assert mode in [None, "GRAY", "RGB", "NIR",
                    "CLOUD_MASK"], "The display mode {} is undefined please select in [GRAY,RGB,NIR,CLOUD_MASK]".format(
        mode)
    raster = gdal.Open(path_image)
    if name_image is None:
        name_image = path_image.split("/")[-1]
    raster_array = raster.ReadAsArray()
    raster_array = convert_array(raster_array, mode=mode)

    if mode is None:
        nband = raster_array.shape[0]
        if nband == 2:  # sentinel 1
            size_x, size_y = raster_array.shape[1], raster_array.shape[2]
            raster_array = np.array(
                [np.zeros((size_x, size_y)), raster_array[0, :, :], raster_array[1, :, :], np.zeros((size_x, size_y))])
            mode = "RGB"
        elif nband >= 3:
            mode = "RGB"

        else:
            mode = "GRAY"

    if mode == "GRAY":
        if len(raster_array.shape) > 2:
            plot_gray(raster_array[band, :, :], name_image, ax=ax)
        else:
            plot_gray(raster_array, name_image, ax=ax)
    else:
        plot_sent2(raster_array, mode, name_image=name_image, bound_y=bound_y, bound_x=bound_x, ax=ax)


def info_image(path_img):
    assert os.path.isfile(path_img), "No img found at {}".format(path_img)
    raster = gdal.Open(path_img, gdal.GA_ReadOnly)
    n_band = raster.RasterCount
    print("{} bands found ".format(n_band))
    for b in range(n_band):
        # Read the raster band as separate variable
        band = raster.GetRasterBand(b + 1)
        # Check type of the variable 'band'
        type(band)
        # Data type of the values
        print("The band {} data_tye {}".format(b, gdal.GetDataTypeName(band.DataType)))
        if band.GetMinimum() is None or band.GetMaximum() is None:
            band.ComputeStatistics(0)
            print("Statistics computed.")
        # Print only selected metadata:
        print("[ NO DATA VALUE ] = ", band.GetNoDataValue())  # none
        print("[ MIN ] = ", band.GetMinimum())
        print("[ MAX ] = ", band.GetMaximum())


def create_histo(batch, dict_class, title=""):
    unique, counts = np.unique(batch, return_counts=True)
    dict_histo = dict(zip(unique, counts / np.sum(counts)))
    print(dict_histo)
    histo_val(dict_histo, ax=None, list_class=dict_class.keys(), title=title)