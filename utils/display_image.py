# File with all the functions used to display mages in jupyter Notebook are written

import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import adjust_gamma
from constant.gee_constant import BOUND_X, BOUND_Y
from utils.metrics import ssim_batch, batch_psnr, batch_sam


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


def plot_sent2(raster_array, mode="RGB", name_image="", ax=None, bound_x=None, bound_y=None,gammaCorr=True):
    """

    Args:
        raster_array: The array of S2 pixels to plot
        mode: string could be RGB or NIR
        name_image:
        ax: matplotlib ax on which to plot the image
        bound_x: limit bound x to the output image
        bound_y: limit bound y to the output image
        gammaCorr : bool if true the gamma correction is applied to the array.
    Returns:

    """
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
    if gammaCorr: #
        raster_array=adjust_gamma(raster_array)
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


def rescale_image(raster_array):
    print("Warning the array should be channel last !")
    rescaled_array = np.zeros(raster_array.shape)
    for b in range(raster_array.shape[-1]):
        rescaled_array[:, :, b] = raster_array[:, :, b]

    return rescaled_array


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


def plot_one_band(raster_array, fig, ax, title="", cmap="bone", vminmax=(None, None)):
    # print("Imagse shape {}".format(raster_array))
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(raster_array, cmap=cmap, vmin=vminmax[0], vmax=vminmax[1])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, orientation='vertical')
    if ax is None:
        plt.show()


def plot_all_compar(batch_predict, batch_gt, max_im=100, title=""):
    n = batch_predict.shape[0]
    if n < max_im:
        max_im = n
    # fig,ax2=plt.subplots(n,4,figsize=(15,60))
    lssim, _ = ssim_batch(batch_predict, batch_gt)
    lpsnr, _ = batch_psnr(batch_predict, batch_gt)
    lsam, _ = batch_sam(batch_predict, batch_gt)
    # print(len(lssim))
    for i in range(max_im):
        fig, ax2 = plt.subplots(1, 4, figsize=(20, 20))
        fig.suptitle(title)
        image_pred = batch_predict[i, :, :, :]
        image_gt = batch_gt[i, :, :, :]
        display_final_tile(image_pred, band=[0, 1, 2], ax=ax2[0])
        ax2[0].set_title("Sim True color visualization")
        display_final_tile(image_pred, band=[3, 0, 1], ax=ax2[1])
        ax2[1].set_title("Sim NIR color visualization")
        ax2[2].set_title(
            "Real image ssim {} psnr {} sam {}".format(round(lssim[i], 3), round(lpsnr[i], 3), round(lsam[i], 3)))
        display_final_tile(image_gt, band=[3, 0, 1], ax=ax2[3])
        ax2[3].set_title("Real image NIR color visualisation")
        display_final_tile(image_gt, band=[0, 1, 2], ax=ax2[2])
        plt.show()


def display_final_tile(raster_array, band=None, ax=None):
    # raster_array=np.load(path_npy)
    print(raster_array.shape)
    if ax is None:
        fig, ax = plt.subplots()
    if band is None:
        band = 0
    ax.imshow(raster_array[:, :, band])
    if ax is None:
        plt.show()


def plot_pre_post_pred(image_pre, image_post, image_pred, l_ax=None, L_band=None):
    if l_ax is None:
        fig, l_ax = plt.subplots(1, 3, figsize=(20, 20))
    if L_band is None:
        L_band = [[4, 5, 6], [0, 1, 2], [0, 1, 2]]
    display_final_tile(image_pre, band=L_band[0], ax=l_ax[0])
    l_ax[0].set_title("Pre fire")
    display_final_tile(image_post, band=L_band[1], ax=l_ax[1])
    l_ax[1].set_title("Post fire")
    display_final_tile(image_pred, band=L_band[2], ax=l_ax[2])
    l_ax[2].set_title(" Predict post fire")
    if l_ax is None:
        plt.show()


def display_dvi_class(dvi, ax=None, fig=None):
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(dvi, cmap=plt.cm.get_cmap('afmhot_r', 10), vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, orientation="vertical")
    # plt.colorbar()
    # im.clim(0, 1)
    if ax is None:
        plt.show()


def one_band_hist(b_array, ax=None, r=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.hist(b_array.ravel(), bins=256, range=r, color="red", alpha=0.5)
    ax.set_xlabel('Intensity Value')
    ax.set_ylabel('Count')
    if ax is None:
        plt.show()


