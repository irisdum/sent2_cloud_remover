# File with all the functions used to display mages in jupyter Notebook are written
import glob
import matplotlib.pyplot as plt
import numpy as np

from constant.gee_constant import BOUND_X, BOUND_Y


def find_image_indir(path_dir, image_format):
    """Given a path to a directory and the final format returns a list of all the images which en by this format in the input
    dir"""
    assert image_format in ["vrt", "tif"], "Wrong format should be vrt or tif but is {}".format(format)
    assert path_dir[-1] == "/", "path should en with / not {}".format(path_dir)
    return glob.glob("{}.{}".format(path_dir, image_format))


def plot_sent2(raster_array, mode="RGB", name_image="", ax=None, bound_x=None, bound_y=None):
    assert mode in ["RGB", "NIR"], "mode {} is undifined should be in RGB or NIR".format(mode)
    assert raster_array.shape[0] == 4, "Wrong sentinel 2 input format should be 4 bands not {}".format(
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
        raster_array=np.moveaxis(raster_array[:3, :, :], 0, -1)
        plot_subset_array(raster_array,ax,bound_x,bound_y)
    else:
        print("The plot is made with sent2 b8 as band 0, b4 as band 1 and b3 as band 2")
        ax.set_title("{} in NIR".format(name_image))
        nir_array = np.array([raster_array[3, :, :], raster_array[0, :, :], raster_array[1, :, :]])
        plot_subset_array(nir_array,ax,bound_x,bound_y)
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
