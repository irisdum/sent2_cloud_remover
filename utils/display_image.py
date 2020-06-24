# File with all the functions used to display mages in jupyter Notebook are written

import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
# from skimage.exposure import is_low_contrast,equalize_hist
from constant.fire_severity_constant import DICT_FIRE_SEV_CLASS
from constant.gee_constant import BOUND_X, BOUND_Y, DICT_BAND_X, DICT_BAND_LABEL
from constant.landclass_constant import LISTE_LAND_CLASS, LISTE_COLOR
from utils.converter import convert_array
from utils.fire_severity import get_fire_severity, normalize_cf
from utils.land_classif import load_tile_classif, compute_batch_land_class_stat
from utils.metrics import ssim_batch, batch_psnr, batch_sam
from utils.vi import compute_vi, diff_metric, diff_relative_metric
import matplotlib.colors as colors
import pandas as pd
import seaborn as sn

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


def plot_subset_array(raster_array, ax, bound_x, bound_y, rescaled=False):
    if rescaled:
        raster_array = rescale_image(raster_array)
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


def plot_one_band(raster_array, fig, ax, title=""):
    # print("Imagse shape {}".format(raster_array))
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(raster_array, cmap='bone')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, orientation='vertical')
    if ax is None:
        plt.show()


def display_one_image_vi(raster_array, fig, ax, vi, dict_band=None, title=None, cmap=None, vminmax=(0, 1),
                         path_csv=None, image_id=None):
    raster_vi = compute_vi(raster_array, vi, dict_band,path_csv=path_csv,image_id=image_id)

    if cmap is None:
        cmap = "RdYlGn"
    if ax is None:
        fig, ax = plt.subplots()
    if title is None:
        title = vi
    im = ax.imshow(raster_vi, cmap=cmap, vmin=vminmax[0], vmax=vminmax[1])
    fig.colorbar(im, ax=ax, orientation="vertical")
    ax.set_title(title)
    if ax is None:
        plt.show()


def display_compare_vi(image_pre, image_post, vi, fig, ax, dict_band_pre, dict_band_post, figuresize=None,
                       vminmax=(0, 1), path_csv=None, image_id=None):
    if figuresize is None:
        figuresize = (20, 20)
    if ax is None:
        fig, ax = plt.subplots(1, 4, figsize=figuresize)
    display_one_image_vi(image_pre, fig, ax[0], vi, dict_band_pre, title="vi {} image pre".format(vi), vminmax=vminmax,
                         path_csv=path_csv,image_id=image_id)
    display_one_image_vi(image_post, fig, ax[1], vi, dict_band_post, title="vi {} image post".format(vi),
                         vminmax=vminmax,path_csv=path_csv,image_id=image_id)
    dr_vi = diff_relative_metric(image_pre, image_post, vi, dict_band_pre, dict_band_post,path_csv=path_csv,image_id=image_id)
    d_vi = diff_metric(image_pre, image_post, vi, dict_band_pre, dict_band_post,path_csv=path_csv,image_id=image_id)
    d_im = ax[2].imshow(d_vi, cmap="bwr", vmin=vminmax[0], vmax=vminmax[1])
    ax[2].set_title("differenced {}".format(vi))
    fig.colorbar(d_im, ax=ax[2], orientation="vertical")
    dr_im = ax[3].imshow(dr_vi, cmap="bwr", vmin=vminmax[0], vmax=vminmax[1])
    ax[3].set_title("relative differenced {}".format(vi))
    fig.colorbar(dr_im, ax=ax[3], orientation="vertical")
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

def plot_compare_vi(image_pre_fire, image_post_fire, image_pred, vi, image_id=None, path_csv=None):
    plot_pre_post_pred(image_pre_fire, image_post_fire, image_pred)
    fig, ax = plt.subplots(1, 3, figsize=(40, 10))
    # vi_pre=compute_vi(image_pre_fire,vi)
    if path_csv is not None:
        vminmax=(0,1)
    else:
        vminmax=(-1,1)
    display_one_image_vi(image_pre_fire, fig, ax[0], vi, dict_band={"R": [4], "NIR": [7]}, title='Pre fire', cmap=None,
                         vminmax=vminmax,path_csv=path_csv,image_id=image_id)
    # vi_post=compute_vi(image_post,vi)
    display_one_image_vi(image_post_fire, fig, ax[1], vi, dict_band=None, title='GT post fire', cmap=None,
                         vminmax=vminmax,path_csv=path_csv,image_id=image_id)
    # vi_pred=compute_vi(image_pred,vi)
    display_one_image_vi(image_pred, fig, ax[2], vi, dict_band=None, title='Prediction post fire', cmap=None,
                         vminmax=vminmax,path_csv=path_csv,image_id=image_id)
    plt.show()

def plot_compare_dvi(gt_dvi,pred_dvi):
    fig2, ax2 = plt.subplots(1, 2, figsize=(20, 30))
    display_one_image_vi(gt_dvi, fig2, ax2[0], "identity", dict_band=None, title='GT Relative difference', cmap="OrRd")
    display_one_image_vi(pred_dvi, fig2, ax2[1], "identity", dict_band=None, title='Pred Relative difference',
                         cmap="OrRd")
    plt.show()

def compute_batch_vi(batch_x, batch_predict, batch_gt, max_im=100, vi="ndvi", liste_image_id=None, path_csv=None):
    """:param path_csv path to the csv file which contains min and max value"""
    n = batch_predict.shape[0]
    if n < max_im:
        max_im = n
    if liste_image_id is None:
        liste_image_id=[None for i in range(max_im)]
    for i in range(max_im):
        image_pre_fire = batch_x[i, :, :, :]
        image_post_fire = batch_gt[i, :, :, :]
        image_pred = batch_predict[i, :, :, ]
        print(image_pre_fire.shape, image_post_fire.shape, image_pred.shape)
        plot_compare_vi(image_pre_fire, image_post_fire, image_pred, vi,image_id=liste_image_id[i],path_csv=path_csv)
        gt_dvi = diff_metric(image_pre_fire, image_post_fire, vi, dict_band_pre={"R": [4], "NIR": [7]},
                             dict_band_post=DICT_BAND_LABEL,image_id=liste_image_id[i],path_csv=path_csv)
        pred_dvi = diff_metric(image_pre_fire, image_pred, vi, dict_band_pre=DICT_BAND_X,
                               dict_band_post=DICT_BAND_LABEL,image_id=liste_image_id[i],path_csv=path_csv)
        plot_compare_dvi(gt_dvi, pred_dvi)

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


def plot_landclass(array_lc, ax=None, fig=None,l_land_class=None,vmin=1,vmax=25):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    cmap, norm, boundaries = define_colormap()

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    im = ax.imshow(array_lc, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", ticks=range(vmax-vmin))
    if l_land_class is None:
        l_land_class=LISTE_LAND_CLASS
    cbar.ax.set_yticks()
    cbar.ax.set_yticklabels(l_land_class)
    # ax.legend([mpatches.Patch(color=cmap(b)) for b in boundaries[:-1]],
    #           ['{} - {}'.format(boundaries[i], LISTE_LAND_CLASS[i]) for i in range(23)], loc='center left',
    #           bbox_to_anchor=(1, 0.5))
    if ax is None:
        plt.show()


def define_colormap(list_col=None):
    if list_col is None:
        list_col=LISTE_COLOR
    cmap = colors.ListedColormap(list_col)
    boundaries = [i for i in range(24)]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    return cmap, norm, boundaries


def analyze_vege(path_tile, batch_x, batch_label, path_lc, input_dataset, batch_pred=None,get_stat=True):
    if batch_pred is None:
        ncol = 5
    else:
        ncol = 7
    assert len(path_tile) == batch_x.shape[
        0], "The list name tile len {} does not have the same length of the batch {}".format(len(path_tile),
                                                                                             batch_x.shape[0])
    l_array_lc = load_tile_classif(input_dataset, path_tile, path_lc, max_im=1000)
    if get_stat:
        batch_stat_df = compute_batch_land_class_stat(l_array_lc,path_tile)

    for i in range(len(path_tile)):
        fig, ax = plt.subplots(1, ncol, figsize=(15, 15))
        fig.suptitle(path_tile[i])
        display_final_tile(batch_x[i, :, :, :], band=[4, 5, 6], ax=ax[0])
        display_final_tile(batch_x[i, :, :, :], band=[7, 4, 5], ax=ax[1])
        display_final_tile(batch_label[i, :, :, :], band=[0, 1, 2], ax=ax[2])
        display_final_tile(batch_label[i, :, :, :], band=[3, 1, 2], ax=ax[3])
        plot_landclass(l_array_lc[i], ax=ax[4], fig=fig)
        if get_stat:
            maj_class=batch_stat_df.iloc[i][LISTE_LAND_CLASS].sort_values(ascending=False).apply(lambda row: round(row,3))[:3].to_dict()
            #print(maj_class)
            print(",".join(["{} {}".format(elem,maj_class[elem]) for elem in maj_class]))
            #ax[4].set_title(",".join(["{} {}".format(elem,maj_class[elem]) for elem in maj_class]))
        if batch_pred is not None:
            display_final_tile(batch_pred[i, :, :, :], band=[0, 1, 2], ax=ax[5])
            display_final_tile(batch_pred[i, :, :, :], band=[3, 1, 2], ax=ax[6])
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

def display_fire_severity(fire_array,ax=None,fig=None,dict_burned=None):
    if dict_burned is None:
        dict_burned=DICT_FIRE_SEV_CLASS
    if ax is None:
        fig,ax=plt.subplots()
    im=ax.imshow(fire_array, cmap=plt.cm.get_cmap('afmhot_r',len(dict_burned)),vmin=0,vmax=len(dict_burned))
    cbar=fig.colorbar(im, ax=ax, orientation="vertical")
    print(dict_burned.keys())
    cbar.ax.set_yticklabels(dict_burned.keys())



def display_fire_severity_bysteps(batch_x, batch_predict, batch_gt, max_im=100, vi="ndvi",dict_burned=None,liste_image_id=None,path_csv=None):
    if dict_burned is None:
        dict_burned=DICT_FIRE_SEV_CLASS
    n = batch_predict.shape[0]
    if n < max_im:
        max_im = n
    output_shape = (batch_gt.shape[0], batch_gt.shape[1], batch_gt.shape[1])
    batch_output_sev = np.ones(output_shape)
    batch_pred_sev = np.ones(output_shape)
    if liste_image_id is None:
        liste_image_id=[None for i in range(max_im)]
    for i in range(max_im):
        image_pre_fire = batch_x[i, :, :, :]
        image_post_fire = batch_gt[i, :, :, :]
        image_pred = batch_predict[i, :, :, ]
        print(image_pre_fire.shape, image_post_fire.shape, image_pred.shape)
        plot_compare_vi(image_pre_fire, image_post_fire, image_pred, vi,image_id=liste_image_id[i],path_csv=path_csv)
        gt_dvi = diff_metric(image_pre_fire, image_post_fire, vi, dict_band_pre={"R": [4], "NIR": [7]},
                             dict_band_post=DICT_BAND_LABEL,image_id=liste_image_id[i],path_csv=path_csv)
        pred_dvi = diff_metric(image_pre_fire, image_pred, vi, dict_band_pre=DICT_BAND_X,
                               dict_band_post=DICT_BAND_LABEL,image_id=liste_image_id[i],path_csv=path_csv)
        plot_compare_dvi(gt_dvi, pred_dvi)
        fig2,ax2=plt.subplots(3,2,figsize=(30,20))
        fig2.suptitle("Image {}".format(i))
        display_dvi_class(gt_dvi, ax=ax2[0, 0], fig=fig2)
        display_dvi_class(pred_dvi, ax=ax2[0, 1], fig=fig2)
        # print_array_stat(gt_dvi)
        # print_array_stat(pred_dvi)
        fire_sev_pred = get_fire_severity(pred_dvi, dict_burned)
        fire_sev_gt = get_fire_severity(gt_dvi, dict_burned)
        batch_output_sev[i, :, :] = fire_sev_gt
        batch_pred_sev[i, :, :] = fire_sev_pred
        one_band_hist(gt_dvi, ax=ax2[1, 0])
        one_band_hist(pred_dvi, ax=ax2[1, 1])
        display_fire_severity(fire_sev_gt, ax2[2, 0], fig2,dict_burned=dict_burned)
        display_fire_severity(fire_sev_pred, ax2[2, 1], fig2,dict_burned=dict_burned)
        plt.show()

    return batch_output_sev, batch_pred_sev


def one_band_hist(b_array,ax=None):
    if ax is None:
        fig,ax=plt.subplots()
    ax.hist(b_array.ravel(), bins=256, color="red", alpha=0.5)
    ax.set_xlabel('Intensity Value')
    ax.set_ylabel('Count')
    if ax is None:
        plt.show()


def plot_cfmat(cf_mat,class_firesev=None,title=""):
    if class_firesev is None:
        class_firesev=DICT_FIRE_SEV_CLASS.keys()
    df_cm=pd.DataFrame(cf_mat, index=class_firesev,
                 columns=class_firesev)
    fig,ax=plt.subplots(figsize=(20,20))
    sn.heatmap(df_cm, annot=True,cmap="Blues")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.suptitle(title)
    plt.show()


def plot_all_cfmat(cf_mat,class_firesev=None):
    plot_cfmat(cf_mat,class_firesev,"Confusion matrix not normalized")
    plot_cfmat(normalize_cf(cf_mat,0), class_firesev,"Confusion matrix normalized by column")
    plot_cfmat(normalize_cf(cf_mat, 1), class_firesev,"Confusion matrix normalized by line")

def print_stats(cf_mat,class_firesev):
    print("Nber of True label on each categorie : ")
    tot=cf_mat.astype(np.float).sum(axis=1)
    for i,elem in enumerate(class_firesev):
        print("Classe {} pixel percentage {:.2%}".format(elem,tot[i]/np.sum(tot)))
    n=len(class_firesev)
    print("Accuracy {} Recall {}".format(np.trace(normalize_cf(cf_mat,1))/n,np.trace(normalize_cf(cf_mat,0))/n))

def plot_hist_vege(conf_vege,weights=None):
    w,bins=np.histogram(np.array(conf_vege),range=(1,25),bins=24)
    w=w/conf_vege.size
    if weights is not None:
        w=list(np.divide(np.array(w),np.array(weights)))
    fix,ax=plt.subplots(figsize=(20,5))
    counts, bins, patches=ax.hist(bins[:-1],bins,align="mid",rwidth=0.5,weights=w)
    ax.set_xticks(bins)
    plt.show()
    return w


def proba_wc_vege(batch_classif, batch_confusion, plot=True, N_tot=24, all_val=True, list_class=None):
    unique, counts = np.unique(batch_classif, return_counts=True)
    freq = list(counts / np.sum(counts))
    dic_vege = dict(zip(unique, freq))
    unique_inter, count_inter = np.unique(batch_confusion, return_counts=True)
    unique_inter, count_inter = unique_inter[:-1], count_inter[:-1]
    dic_temp = dict(zip(unique_inter, count_inter))
    freq_final = []
    sum_wc = np.sum(count_inter)
    print(unique, unique_inter)
    for i in range(1, N_tot):
        if i not in unique_inter:
            freq_final += [0]
        else:
            freq_inter = dic_temp[i] / sum_wc
            if i not in dic_vege.keys():
                freq_final += [0]
            else:
                freq_final += [freq_inter / dic_vege[i]]
    print("Init {} Wrongly classified {}".format(np.sum(counts), sum_wc))
    unique_inter, count_inter = np.array(unique_inter), np.array(count_inter)
    dic_final = dict(zip([i for i in range(N_tot)], freq_final))
    if plot:
        histo_val(dic_vege)
        histo_val(dict(zip(unique_inter, count_inter / np.sum(count_inter))))
        histo_val(dic_final)
    if all_val:
        return dic_final, dic_vege, sum_wc, np.sum(counts)
    else:
        return dic_final


def histo_val(dict_freq, ax=None, liste_classe=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 5))

    ax.bar(dict_freq.keys(), dict_freq.values(), tick_label=liste_classe)
    # ax.set_xticks(dict_freq.keys())
    plt.show()


#print(proba_wc_vege(batch_landclass, conf_vege))