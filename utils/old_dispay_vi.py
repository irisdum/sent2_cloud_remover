import numpy as np
import pandas as pd

from constant.gee_constant import DICT_BAND_LABEL, DICT_BAND_X
from constant.landclass_constant import LISTE_LAND_CLASS, LISTE_COLOR
from utils.display_image import display_final_tile, plot_pre_post_pred, display_dvi_class, \
    one_band_hist
from utils.fire_severity import get_fire_severity, normalize_cf
from utils.tif_classif import load_tile_classif, compute_batch_land_class_stat
from constant.fire_severity_constant import DICT_FIRE_SEV_CLASS

import matplotlib.pyplot as plt

from utils.vi import compute_vi, diff_relative_metric, diff_metric


def display_fire_severity(fire_array, ax=None, fig=None, dict_burned=None, cmap='afmhot_r'):
    if dict_burned is None:
        dict_burned = DICT_FIRE_SEV_CLASS
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(fire_array, cmap=plt.cm.get_cmap(cmap, len(dict_burned)), vmin=0, vmax=len(dict_burned))
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", ticks=[i for i in range(len(dict_burned))])
    print(dict_burned.keys())
    cbar.ax.set_yticklabels(dict_burned.keys())

def analyze_vege(path_tile, batch_x, batch_label, path_lc, input_dataset, batch_pred=None, get_stat=True):
    if batch_pred is None:
        ncol = 5
    else:
        ncol = 7
    assert len(path_tile) == batch_x.shape[
        0], "The list name tile len {} does not have the same length of the batch {}".format(len(path_tile),
                                                                                             batch_x.shape[0])
    l_array_lc = load_tile_classif(input_dataset, path_tile, path_lc, max_im=1000)
    if get_stat:
        batch_stat_df = compute_batch_land_class_stat(l_array_lc, path_tile)

    for i in range(len(path_tile)):
        fig, ax = plt.subplots(1, ncol, figsize=(15, 15))
        fig.suptitle(path_tile[i])
        display_final_tile(batch_x[i, :, :, :], band=[4, 5, 6], ax=ax[0])
        display_final_tile(batch_x[i, :, :, :], band=[7, 4, 5], ax=ax[1])
        display_final_tile(batch_label[i, :, :, :], band=[0, 1, 2], ax=ax[2])
        display_final_tile(batch_label[i, :, :, :], band=[3, 1, 2], ax=ax[3])
        plot_landclass(l_array_lc[i], ax=ax[4], fig=fig)
        if get_stat:
            maj_class = batch_stat_df.iloc[i][LISTE_LAND_CLASS].sort_values(ascending=False).apply(
                lambda row: round(row, 3))[:3].to_dict()
            # print(maj_class)
            print(",".join(["{} {}".format(elem, maj_class[elem]) for elem in maj_class]))
            # ax[4].set_title(",".join(["{} {}".format(elem,maj_class[elem]) for elem in maj_class]))
        if batch_pred is not None:
            display_final_tile(batch_pred[i, :, :, :], band=[0, 1, 2], ax=ax[5])
            display_final_tile(batch_pred[i, :, :, :], band=[3, 1, 2], ax=ax[6])
        plt.show()


def display_one_image_vi(raster_array, fig, ax, vi, dict_band=None, title=None, cmap=None, vminmax=(0, 1),
                         path_csv=None, image_id=None):
    raster_vi = compute_vi(raster_array, vi, dict_band, path_csv=path_csv, image_id=image_id)

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
                         path_csv=path_csv, image_id=image_id)
    display_one_image_vi(image_post, fig, ax[1], vi, dict_band_post, title="vi {} image post".format(vi),
                         vminmax=vminmax, path_csv=path_csv, image_id=image_id)
    dr_vi = diff_relative_metric(image_pre, image_post, vi, dict_band_pre, dict_band_post, path_csv=path_csv,
                                 image_id=image_id)
    d_vi = diff_metric(image_pre, image_post, vi, dict_band_pre, dict_band_post, path_csv=path_csv, image_id=image_id)
    d_im = ax[2].imshow(d_vi, cmap="bwr", vmin=vminmax[0], vmax=vminmax[1])
    ax[2].set_title("differenced {}".format(vi))
    fig.colorbar(d_im, ax=ax[2], orientation="vertical")
    dr_im = ax[3].imshow(dr_vi, cmap="bwr", vmin=vminmax[0], vmax=vminmax[1])
    ax[3].set_title("relative differenced {}".format(vi))
    fig.colorbar(dr_im, ax=ax[3], orientation="vertical")
    plt.show()


def plot_compare_vi(image_pre_fire, image_post_fire, image_pred, vi, image_id=None, path_csv=None):
    plot_pre_post_pred(image_pre_fire, image_post_fire, image_pred)
    fig, ax = plt.subplots(1, 3, figsize=(40, 10))
    # vi_pre=compute_vi(image_pre_fire,vi)
    if path_csv is not None:
        vminmax = (0, 1)
    else:
        vminmax = (-1, 1)
    display_one_image_vi(image_pre_fire, fig, ax[0], vi, dict_band={"R": [4], "NIR": [7]}, title='Pre fire', cmap=None,
                         vminmax=vminmax, path_csv=path_csv, image_id=image_id)
    # vi_post=compute_vi(image_post,vi)
    display_one_image_vi(image_post_fire, fig, ax[1], vi, dict_band=None, title='GT post fire', cmap=None,
                         vminmax=vminmax, path_csv=path_csv, image_id=image_id)
    # vi_pred=compute_vi(image_pred,vi)
    display_one_image_vi(image_pred, fig, ax[2], vi, dict_band=None, title='Prediction post fire', cmap=None,
                         vminmax=vminmax, path_csv=path_csv, image_id=image_id)
    plt.show()


def compute_batch_vi(batch_x, batch_predict, batch_gt, max_im=100, vi="ndvi", liste_image_id=None, path_csv=None):
    """:param path_csv path to the csv file which contains min and max value"""
    n = batch_predict.shape[0]
    if n < max_im:
        max_im = n
    if liste_image_id is None:
        liste_image_id = [None for i in range(max_im)]
    for i in range(max_im):
        image_pre_fire = batch_x[i, :, :, :]
        image_post_fire = batch_gt[i, :, :, :]
        image_pred = batch_predict[i, :, :, ]
        print(image_pre_fire.shape, image_post_fire.shape, image_pred.shape)
        plot_compare_vi(image_pre_fire, image_post_fire, image_pred, vi, image_id=liste_image_id[i], path_csv=path_csv)
        gt_dvi = diff_metric(image_pre_fire, image_post_fire, vi, dict_band_pre={"R": [4], "NIR": [7]},
                             dict_band_post=DICT_BAND_LABEL, image_id=liste_image_id[i], path_csv=path_csv)
        pred_dvi = diff_metric(image_pre_fire, image_pred, vi, dict_band_pre=DICT_BAND_X,
                               dict_band_post=DICT_BAND_LABEL, image_id=liste_image_id[i], path_csv=path_csv)
        plot_compare_dvi(gt_dvi, pred_dvi)


def plot_compare_dvi(gt_dvi, pred_dvi):
    fig2, ax2 = plt.subplots(1, 2, figsize=(20, 30))
    display_one_image_vi(gt_dvi, fig2, ax2[0], "identity", dict_band=None, title='GT Relative difference', cmap="OrRd")
    display_one_image_vi(pred_dvi, fig2, ax2[1], "identity", dict_band=None, title='Pred Relative difference',
                         cmap="OrRd")
    plt.show()


def plot_landclass(array_lc, ax=None, fig=None, l_land_class=None, vmin=1, vmax=25):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    cmap, norm, boundaries = define_colormap()

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    im = ax.imshow(array_lc, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", ticks=range(vmax - vmin))
    if l_land_class is None:
        l_land_class = LISTE_LAND_CLASS
    cbar.ax.set_yticks()
    cbar.ax.set_yticklabels(l_land_class)
    # ax.legend([mpatches.Patch(color=cmap(b)) for b in boundaries[:-1]],
    #           ['{} - {}'.format(boundaries[i], LISTE_LAND_CLASS[i]) for i in range(23)], loc='center left',
    #           bbox_to_anchor=(1, 0.5))
    if ax is None:
        plt.show()


def display_fire_severity_bysteps(batch_x, batch_predict, batch_gt, max_im=100, vi="ndvi", dict_burned=None,
                                  liste_image_id=None, path_csv=None):
    if dict_burned is None:
        dict_burned = DICT_FIRE_SEV_CLASS
    n = batch_predict.shape[0]
    if n < max_im:
        max_im = n
    output_shape = (batch_gt.shape[0], batch_gt.shape[1], batch_gt.shape[1])
    batch_output_sev = np.ones(output_shape)
    batch_pred_sev = np.ones(output_shape)
    if liste_image_id is None:
        liste_image_id = [None for i in range(max_im)]
    for i in range(max_im):
        image_pre_fire = batch_x[i, :, :, :]
        image_post_fire = batch_gt[i, :, :, :]
        image_pred = batch_predict[i, :, :, ]
        print(image_pre_fire.shape, image_post_fire.shape, image_pred.shape)
        plot_compare_vi(image_pre_fire, image_post_fire, image_pred, vi, image_id=liste_image_id[i], path_csv=path_csv)
        gt_dvi = diff_metric(image_pre_fire, image_post_fire, vi, dict_band_pre={"R": [4], "NIR": [7]},
                             dict_band_post=DICT_BAND_LABEL, image_id=liste_image_id[i], path_csv=path_csv)
        pred_dvi = diff_metric(image_pre_fire, image_pred, vi, dict_band_pre=DICT_BAND_X,
                               dict_band_post=DICT_BAND_LABEL, image_id=liste_image_id[i], path_csv=path_csv)
        plot_compare_dvi(gt_dvi, pred_dvi)
        fig2, ax2 = plt.subplots(3, 2, figsize=(30, 20))
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
        display_fire_severity(fire_sev_gt, ax2[2, 0], fig2, dict_burned=dict_burned)
        display_fire_severity(fire_sev_pred, ax2[2, 1], fig2, dict_burned=dict_burned)
        plt.show()

    return batch_output_sev, batch_pred_sev


def plot_cfmat(cf_mat, class_firesev=None, title=""):
    if class_firesev is None:
        class_firesev = DICT_FIRE_SEV_CLASS.keys()
    df_cm = pd.DataFrame(cf_mat, index=class_firesev,
                         columns=class_firesev)
    fig, ax = plt.subplots(figsize=(20, 20))
    sn.heatmap(df_cm, annot=True, cmap="Blues")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.suptitle(title)
    plt.show()


def plot_all_cfmat(cf_mat, class_firesev=None):
    plot_cfmat(cf_mat, class_firesev, "Confusion matrix not normalized")
    plot_cfmat(normalize_cf(cf_mat, 0), class_firesev, "Confusion matrix normalized by column")
    plot_cfmat(normalize_cf(cf_mat, 1), class_firesev, "Confusion matrix normalized by line")


def print_stats(cf_mat, class_firesev):
    print("Nber of True label on each categorie : ")
    tot = cf_mat.astype(np.float).sum(axis=1)
    for i, elem in enumerate(class_firesev):
        print("Classe {} pixel percentage {:.2%}".format(elem, tot[i] / np.sum(tot)))
    n = len(class_firesev)
    print("Accuracy {} Recall {}".format(np.trace(normalize_cf(cf_mat, 1)) / n, np.trace(normalize_cf(cf_mat, 0)) / n))


def plot_hist_vege(conf_vege, weights=None):
    w, bins = np.histogram(np.array(conf_vege), range=(1, 25), bins=24)
    w = w / conf_vege.size
    if weights is not None:
        w = list(np.divide(np.array(w), np.array(weights)))
    fix, ax = plt.subplots(figsize=(20, 5))
    counts, bins, patches = ax.hist(bins[:-1], bins, align="mid", rwidth=0.5, weights=w)
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


def histo_val(dict_freq, ax=None, list_class=None, title=""):
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 5))
        fig.suptitle(title)
    ax.bar(dict_freq.keys(), dict_freq.values(), tick_label=list_class, width=0.8)
    if list_class is not None:
        ax.set_xticks([i for i in range(0, len(list_class) + 1)], list_class)
        ax.set_xticklabels(list_class, rotation=70)
    # ax.set_xticks(dict_freq.keys())
    plt.show()


def display_silhouette(labels, silhouette_vals, ax1=None):
    if ax1 is None:
        fig1, ax1 = plt.subplots()
    # Silhouette plot
    y_ticks = []
    y_lower, y_upper = 0, 0
    for i, cluster in enumerate(np.unique(labels)):
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
        ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
        y_lower += len(cluster_silhouette_vals)

    # Get the average silhouette score and plot it
    avg_score = np.mean(silhouette_vals)
    ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')
    ax1.set_yticks([])
    ax1.set_xlim([-0.1, 1])
    ax1.set_xlabel('Silhouette coefficient values')
    ax1.set_ylabel('Cluster labels')
    ax1.set_title('Silhouette plot for the various clusters', y=1.02)


def define_colormap(list_col=None):
    if list_col is None:
        list_col = LISTE_COLOR
    cmap = colors.ListedColormap(list_col)
    boundaries = [i for i in range(24)]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    return cmap, norm, boundaries