# a python file where all the functions likned withe the preprocessing of the data just before the networks are implemented
# different methods are encode : normalization, centering or standardized values

from constant.gee_constant import DICT_BAND_X, DICT_BAND_LABEL, DICT_RESCALE, DICT_METHOD
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
        fig, ax = plt.subplots(1, n_images, figsize=(20, 20))
        for i, b_index in enumerate(dict_bandX[band]):
            if i == 0:
                band_array = arrayX[:, :, b_index]
                if plot:
                    plot_one_band(band_array, fig, ax[i])
            else:
                # print("we add another band")
                band_array = np.append(band_array, arrayX[:, :, b_index])
                if plot:
                    plot_one_band(arrayX[:, :, b_index], fig, ax[i],
                                  title="DATA X band {} index {}".format(band, b_index))
        if band in dictlabel:
            print("{} is also in label".format(band))
            for i, index in enumerate(dictlabel[band]):
                band_array = np.append(band_array, arraylabel[:, :, index])
                plot_one_band(arraylabel[:, :, index], fig, ax[i + len(dict_bandX[band]) - 1],
                              title="LABEL {}".format(band))
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
    assert rescale_type in dict_method, "Rescaling undefined {} not in ".format(rescale_type, dict_method)
    dict_stat = compute_image_stats(array_dataX, array_label, dict_bandX=dict_band_X, dictlabel=dict_band_label,
                                    plot=plot, stats=dict_method[rescale_type])
    print("THE STATISTICS {} COMPUTED ARE {}".format(dict_method[rescale_type], dict_stat))
    rescaled_dataX = image_rescaling(array_dataX, dict_band_X, dict_stat, rescaling_function(rescale_type))
    rescaled_label = image_rescaling(array_label, dict_band_label, dict_stat, rescaling_function(rescale_type))
    dict_stat_after = compute_image_stats(rescaled_dataX, rescaled_label, dict_band_X, dict_band_label, plot=plot,
                                          stats=dict_method[rescale_type])
    print("AFTER THE RESCALING {} THE STATISTIC {} COMPUTED ARE {} ".format(rescale_type, dict_method[rescale_type],
                                                                            dict_stat_after))

    return rescaled_dataX, rescaled_label


def rescaling_combined_methods(array_dataX, array_label, dict_band_X, dict_band_label, dict_rescale_type=None,
                               plot=True,dict_stat=None):
    """Rescale combiend on an image"""
    rescaled_arrayX = np.zeros(array_dataX.shape)
    rescaled_label = np.zeros(array_label.shape)
    dict_method = DICT_METHOD
    if dict_rescale_type is None:
        dict_rescale_type = DICT_RESCALE  # by band gives the method used
    for band in dict_rescale_type:
        dx, dlabel = create_dict_bande(band, dict_band_X, dict_band_label)
        rescale_type = dict_rescale_type[band]
        if dict_stat is None: #we compute the stats each time for each band
            dict_stat = compute_image_stats(array_dataX, array_label, dict_bandX=dx, dictlabel=dlabel,
                                        stats=dict_method[rescale_type],plot=plot)
        rescaled_arrayX[:, :, dict_band_X[band]] = rescaling_function(rescale_type)(
            array_dataX[:, :, dict_band_X[band]], dict_stat[band][0], dict_stat[band][1])
        if band in dlabel:
            rescaled_label[:, :, dict_band_label[band]] = rescaling_function(rescale_type)(
                array_label[:, :, dict_band_label[band]], dict_stat[band][0], dict_stat[band][1])

    return rescaled_arrayX,rescaled_label


def rescale_on_batch(batch_X,batch_label,dict_band_X=None,dict_band_label=None,dict_rescale_type=None):
    """Rescale combined on a batch of images"""
    batch_size=batch_X.shape[0]
    if dict_band_label is None:
        dict_band_label=DICT_BAND_LABEL
    if dict_band_X is None:
        dict_band_X=DICT_BAND_X
    rescaled_batch_X=np.zeros(batch_X.shape)
    rescaled_batch_label=np.zeros(batch_label.shape)
    if dict_rescale_type is None:
        dict_rescale_type = DICT_RESCALE  # by band gives the method used
    dict_stat=compute_batch_stats(batch_X,batch_label,dict_band_X,dict_band_label,dict_rescale_type,dict_method=None)
    #print("FINAL DICT STAT {}".format(dict_stat))
    for i in range(batch_size): #Rescale all the image on the batch
       rescaled_batch_X[i,:,:,:],rescaled_batch_label[i,:,:,:]=rescaling_combined_methods(batch_X[i,:,:,:],batch_label[i,:,:,:],dict_band_X,
                                                                                          dict_band_label,dict_rescale_type,plot=False,dict_stat=dict_stat)
    return rescaled_batch_X,rescaled_batch_label


def compute_batch_stats(batch_X,batch_label,dict_band_X,dict_band_label,dict_rescale_type,dict_method=None):
    if dict_method is None:
        dict_method=DICT_METHOD
    batch_size = batch_X.shape[0]
    list_batch_stat = [] #list of all the dict stat of each batch
    dict_stat=dict(zip([i for i in dict_rescale_type],[(0,0) for i in range(len(dict_rescale_type))]))
    #print("dict_stat",dict_stat)
    for i in range(batch_size):  # go over all the tiles in the batch tile to compute the stats
        array_dataX=batch_X[i,:,:,:]
        array_label=batch_label[i,:,:,:]
        stat_one_batch={}
        for band in dict_rescale_type:
            dx, dlabel = create_dict_bande(band, dict_band_X, dict_band_label) #small dicts with only band
            rescale_type = dict_rescale_type[band]
            sub_dict_stat = compute_image_stats(array_dataX, array_label, dict_bandX=dx, dictlabel=dlabel,
                                            stats=dict_method[rescale_type], plot=False)
            stat_one_batch.update(sub_dict_stat)
        list_batch_stat+=[stat_one_batch]
    assert len(list_batch_stat)==batch_size, "Not enought stat has been computed {}".format(list_batch_stat)
    #print("THE LIST OF THE BATHC STATS IS {}".format(list_batch_stat))
    #initialize dict
    for i in range(len(list_batch_stat)):
        #print(list_batch_stat[i])
        for band in list_batch_stat[i]:
            #print("band",band,dict_stat)
            stat1=dict_stat[band][0]+list_batch_stat[i][band][0]
            stat2 = dict_stat[band][1] + list_batch_stat[i][band][1]
            dict_stat.update({band:(stat1,stat2)})
    # DIVIDE BY THE BS
    for band in dict_stat:
        stat1=dict_stat[band][0]/batch_size
        stat2 = dict_stat[band][1] / batch_size
        dict_stat.update({band:(stat1,stat2)})
    #print("THE BATCH STATISTICS ARE {}".format(dict_stat))
    return dict_stat

def create_dict_bande(band, dict_bandX, dict_band_label):
    dX = {}
    dLabel = {}
    if band in dict_bandX:
        dX.update({band: dict_bandX[band]})
    if band in dict_band_label:
        dLabel.update({band: dict_band_label[band]})
    return dX, dLabel


def image_rescaling(data, dict_band, dict_stats, rescale_fun):
    new_array = np.zeros(data.shape)
    for band in dict_band:
        for b_index in dict_band[band]:
            new_array[:, :, b_index] = rescale_fun(data[:, :, b_index], dict_stats[band][0], dict_stats[band][1])
    return new_array


