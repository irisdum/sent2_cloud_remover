# File with all the functions used to display mages in jupyter Notebook are written

import glob
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal

from constant.gee_constant import BOUND_X, BOUND_Y, LISTE_BANDE, CONVERTOR, SCALE_S1

def plot_allbands_hist(path_tif,ax):
    raster=gdal.Open(path_tif)
    image=raster.ReadAsArray()
    image=np.moveaxis(image,0,-1)
    if ax is None:
        fig,ax=plt.subplots()
    ax.hist(image.ravel(), bins = 256, color = 'orange', )
    nb=image.shape[-1]
    if nb==5:
        nb-=1
    l_legend=[]
    l_color=["red","green","blue","gray","pink"]
    for b in range(nb):
        ax.hist(image[:, :, b].ravel(), bins = 256, color = l_color[b], alpha = 0.5)
        l_legend+=["Bande {}".format(b+1)]
        ax.set_xlabel('Intensity Value')
        ax.set_ylabel('Count')
        ax.legend(l_legend)
    if ax is None:
        plt.show()


def convert_array(raster_array, scale_s1=SCALE_S1, mode=None):
    if raster_array.dtype == np.uint16:  # sentinel 2 data needs to be converted and rescale
        return uin16_2_float32(raster_array)
    elif raster_array.dtype == np.float32:
        return np.divide(raster_array, scale_s1).astype(np.float32)
    elif mode == "CLOUD_MASK":
        np.where(raster_array == 1, 0, raster_array)  # clear land pixel
        np.where(raster_array == 4, 0, raster_array)  # snow (cf http://www.pythonfmask.org/en/latest/fmask_fmask.html)
        np.where(raster_array == 5, 0, raster_array)  # water
        return np.divide(raster_array, 5).astype(np.float32)
    else:
        return np.divide(raster_array,np.max(raster_array))

def uin16_2_float32(raster_array, max_scale=CONVERTOR):
    scaled_array = np.divide(raster_array, max_scale)
    return scaled_array.astype(np.float32)


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
    raster_array = convert_array(raster_array,mode=mode)
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


def info_image(path_tif):
    raster = gdal.Open(path_tif)
    n_band = raster.RasterCount
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


def plot_gray(raster_array, name_image, bound_x=None, bound_y=None, ax=None):
    if bound_x is None:
        bound_x = BOUND_X
    if bound_y is None:
        bound_y = BOUND_Y
    assert len(raster_array.shape) == 2, "More than one band dim are {}".format(raster_array.shape)
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.set_title(name_image)
    plot_subset_array(raster_array, ax, bound_x=bound_x, bound_y=bound_y)
    if ax is None:
        plt.show()


def find_image_indir(path_dir, image_format):
    """Given a path to a directory and the final format returns a list of all the images which en by this format in the input
    dir"""
    assert image_format in ["vrt", "tif", "SAFE/"], "Wrong format should be vrt or tif SAFE/ but is {}".format(format)
    assert path_dir[-1] == "/", "path should en with / not {}".format(path_dir)
    return glob.glob("{}*.{}".format(path_dir, image_format))


def plot_sent2(raster_array, mode="RGB", name_image="", ax=None, bound_x=None, bound_y=None):
    assert mode in ["RGB", "NIR", "CLOUD_MASK"], "mode {} is undifined should be in RGB or NIR or CLOUD_MASK".format(
        mode)
    assert raster_array.shape[0] >= 3, "Wrong sentinel 2 input format should be at least 4 bands {}".format(
        raster_array.shape[0])
    if ax is None:
        fig, ax = plt.subplots()
    if bound_x is None:
        bound_x = BOUND_X
    if bound_y is None:
        bound_y = BOUND_Y

    if mode == "RGB":
        print("The plot is made with sent2 b4 as band 0, b3 as band 1 and b2 as band 2 in the raster array")
        ax.set_title("{} in RGB".format(name_image))
        raster_array = np.moveaxis(raster_array[:3, :, :], 0, -1)
        plot_subset_array(raster_array, ax, bound_x, bound_y)
    elif mode == "NIR":
        print("The plot is made with sent2 b8 as band 0, b4 as band 1 and b3 as band 2")
        ax.set_title("{} in NIR".format(name_image))
        nir_array = np.array([raster_array[3, :, :], raster_array[0, :, :], raster_array[1, :, :]])
        nir_array = np.moveaxis(nir_array, 0, -1)
        plot_subset_array(nir_array, ax, bound_x, bound_y)
    else:
        print("plot of the cloud mask which is the last band")
        ax.set_title("{} cloud mask".format(name_image))
        plot_subset_array(raster_array[4, :, :], ax, bound_x, bound_y)
    if ax is None:
        plt.show()


def plot_subset_array(raster_array, ax, bound_x, bound_y):
    ax.imshow(raster_array[bound_x[0]:bound_x[1], bound_y[0]:bound_y[1]])


def plot_s2(raster_array, opt="RGB"):
    fig, ax = plt.subplots()
    if opt == "RGB":
        ax.set_title("RGB")
        ax.imshow(np.moveaxis(raster_array[:3, :, :], 0, -1))
    elif opt == "NIR":
        # total_array=np.moveaxis(raster_array[:3,:,:],0,-1) #we put color dim at the end
        nir_array = np.array([raster_array[3, :, :], raster_array[0, :, :], raster_array[1, :, :]])
        print(nir_array.shape)
        ax.set_title("NIR")
        ax.imshow(np.moveaxis(nir_array, 0, -1))
    plt.show()
