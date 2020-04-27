# a python file where all the functions likned withe the preprocessing of the data just before the networks are implemented
# different methods are encode : normalization, centering or standardized values

from constant.gee_constant import DICT_BAND_X, DICT_BAND_LABEL
from utils.display_image import plot_one_band
import matplotlib.pyplot as plt
import numpy as np


def compute_image_stats(arrayX, arraylabel, dict_bandX=None, dictlabel=None, plot=True, stats="mean_std"):
    """Compute the statistics using the X array and its label. Statistics are computed for each band descried in dict_band"""
    assert stats in ["mean_std", "min_max"], "Stats function {} undefined".format(stats)
    if dict_bandX is None:
        dict_bandX = DICT_BAND_X
    if dictlabel is None:
        dictlabel = DICT_BAND_LABEL
    dict_stats = {}
    for band in dict_bandX:
        n_images = len(dict_bandX[band])
        if band in dictlabel:
            n_images += len(dictlabel[band])
        fig, ax = plt.subplots(1, n_images, figsize=(20,20))
        for i, b_index in enumerate(dict_bandX[band]):
            if i == 0:
                band_array = arrayX[:, :, b_index]
                if plot:
                    plot_one_band(band_array, fig, ax[i])
            else:
                # print("we add another band")
                band_array = np.append(band_array, arrayX[:, :, b_index])
                if plot:
                    plot_one_band(arrayX[:, :, b_index], fig, ax[i], title="DATA X band {} index {}".format(band, b_index))
        if band in dictlabel:
            print("{} is also in label".format(band))
            for i, index in enumerate(dictlabel[band]):
                band_array = np.append(band_array, arraylabel[:, :, index])
                plot_one_band(arraylabel[:, :, index], fig, ax[i + len(dict_bandX[band]) - 1], title="LABEL {}".format(band))
        plt.show()
        band_stat1, band_stat2 = compute_band_stats(band_array, stats)
        dict_stats.update({band: (band_stat1, band_stat2)})
    return dict_stats


def compute_band_stats(band_array, stats):
    if stats == "mean_std":
        stat1, stat2 = band_array.mean(), band_array.std()
    else:
        stat1, stat2 = band_array.min(), band_array.max()
    return stat1, stat2


def positive_standardization(pixels, mean, std):
    """pixels an array
    mean a float
    std a float"""
    pixels = (pixels - mean) / std
    pixels = np.clip(pixels, -1.0, 1.0)
    pixels = (pixels + 1.0) / 2.0
    return pixels


def normalization(pixels, min, max):
    pixels = (pixels - min) / (max - min)
    return pixels


def centering(pixels, mean, std):
    return pixels - mean


def rescaling_function(methode):
    if methode == "normalization":
        def method(pixels, stat1, stat2):
            pixels = (pixels - stat1) / (stat2 - stat1)
            return pixels
    elif methode == "standardization":
        def method(pixels, stat1, stat2):
            pixels = (pixels - stat1) / stat2
            pixels = np.clip(pixels, -1.0, 1.0)
            pixels = (pixels + 1.0) / 2.0
            return pixels
    else:
        def method(pixels, stat1, stat2):
            return pixels - stat1
    return method


def rescaling(array_dataX, array_label, dict_band_X, dict_band_label, rescale_type="normalization", plot=True):

    dict_method = {"standardization": "mean_std", "centering": "mean_std", "normalization": "min_max"}
    assert rescale_type in dict_method, "Rescaling undefined {} not in ".format(rescale_type,dict_method)
    dict_stat = compute_image_stats(array_dataX, array_label, dict_bandX=dict_band_X, dictlabel=dict_band_label,
                                    plot=plot, stats=dict_method[rescale_type])
    print("THE STATISTICS {} COMPUTED ARE {}".format(dict_method[rescale_type],dict_stat))
    rescaled_dataX=image_rescaling(array_dataX,dict_band_X,dict_stat,rescaling_function(rescale_type))
    rescaled_label=image_rescaling(array_label,dict_band_label,dict_stat,rescaling_function(rescale_type))
    dict_stat_after=compute_image_stats(rescaled_dataX,rescaled_label,dict_band_X,dict_band_label,plot=plot,stats=dict_method[rescale_type])
    print("AFTER THE RESCALING {} THE STATISTIC {} COMPUTED ARE {} ".format(rescale_type,dict_method[rescale_type],dict_stat_after))
    return rescaled_dataX,rescaled_label


def image_rescaling(data, dict_band, dict_stats, rescale_fun):
    new_array = np.zeros(data.shape)
    for band in dict_band:
        for b_index in dict_band[band]:
            new_array[:, :, b_index] = rescale_fun(data[:, :, b_index], dict_stats[band][0], dict_stats[band][1])
    return new_array
