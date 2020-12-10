# a python file where all the functions likned withe the preprocessing of the data just before the networks are implemented
# different methods are encode : normalization, centering or standardized values
import os

from typing import Tuple

from sklearn.impute import KNNImputer

from constant.gee_constant import DICT_BAND_X, DICT_BAND_LABEL, DICT_METHOD, DICT_TRANSLATE_BAND, \
    CONVERTOR
from sklearn.preprocessing import StandardScaler, RobustScaler
from constant.processing_constant import DICT_RESCALE, DICT_GROUP_BAND_LABEL, DICT_GROUP_BAND_X, S1_BANDS, S2_BANDS, \
    DICT_RESCALE_TYPE, DICT_SCALER, FACTEUR_STD_S2, FACTEUR_STD_S1, DATA_RANGE
from utils.image_find_tbx import extract_tile_id, find_csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def plot_one_band(raster_array, fig, ax, title="", cmap="bone"):
    """:param raster_array a numpy array
    Function that plot an np array with a colorbar"""
    # print("Imagse shape {}".format(raster_array))
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(raster_array, cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, orientation='vertical')
    if ax is None:
        plt.show()


def compute_image_stats(arrayX, arraylabel, dict_bandX=None, dictlabel=None, plot=False, stats="mean_std"):
    """Compute the statistics using the X array and its label. Statistics are computed for each band described in dict_band
    :param arrayX : a np array  with shape (n_image,m,m,n_channel)
    :param arraylabel : a np array  with shape (n_image,m,m,n_channel) wich correspond to the label array """
    assert stats in ["mean_std", "min_max"], "Stats function {} undefined".format(stats)
    if dict_bandX is None:
        dict_bandX = DICT_BAND_X
    if dictlabel is None:
        dictlabel = DICT_BAND_LABEL
    dict_stats = {}
    for band in dict_bandX:  # First go through all the image band defined in dictbandX
        n_images = len(dict_bandX[band])
        # if band in dictlabel:
        # n_images += len(dictlabel[band])
        if plot:
            fig, ax = plt.subplots(1, n_images, figsize=(20, 20))
        for i, b_index in enumerate(dict_bandX[band]):  # go through all the index given for this band in dataX
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
        if band in dictlabel:  # if this band is also in label
            # print("{} is also in label".format(band))
            for i, index in enumerate(dictlabel[band]):
                band_array = np.append(band_array, arraylabel[:, :, index])
                if plot:
                    plot_one_band(arraylabel[:, :, index], fig, ax[i + len(dict_bandX[band]) - 1],
                                  title="LABEL {}".format(band))
        if plot:
            plt.show()
        band_stat1, band_stat2 = compute_band_stats(band_array, stats)
        dict_stats.update({band: (band_stat1, band_stat2)})
    return dict_stats


def compute_one_image_stat(array_image, dict_band, stats):
    dict_stat = {}
    for band in dict_band:
        band_array = array_image[:, :, dict_band[band]]
        band_stat1, band_stat2 = compute_band_stats(band_array, stats)
        dict_stat.update({band: (band_stat1, band_stat2)})
    return dict_stat


def normalize_one_image(array_image, dict_band, rescale_type="normalization11", dict_method=None):
    if dict_method is None:
        dict_method = DICT_METHOD
    dict_stat = compute_one_image_stat(array_image, dict_band=dict_band, stats=dict_method[rescale_type])
    rescaled_image = image_rescaling(array_image, dict_band, dict_stat, rescaling_function(rescale_type))
    return rescaled_image


def compute_band_stats(band_array, stats):
    if stats == "mean_std":
        stat1, stat2 = band_array.mean(), band_array.std()
    else:
        stat1, stat2 = band_array.min(), band_array.max()
    # print(stat1,stat2)
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
    # print("we use {}".format(methode))
    if methode == "normalization":
        def method(pixels, stat1, stat2):
            pixels = (pixels - stat1) / (stat2 - stat1)
            return pixels
    elif methode == "normalization11":  # normalize between -1 and 1
        def method(pixels, stat1, stat2):
            pixels = 2 * (pixels - stat1) / (stat2 - stat1) - 1
            return pixels
    elif methode == "center_norm11":
        def method(pixels, stat1, stat2):
            val = (stat2 - stat1) / 2
            pixels = (pixels - val) / val
            return pixels
    elif methode == "center_norm11_r":
        def method(pixels, stat1, stat2):
            val = (stat2 - stat1) / 2
            pixels = pixels * val + val
            return pixels
    elif methode == "standardization11":
        def method(pixels, stat1, stat2):
            pixels = (pixels - stat1) / stat2
            # pixels = np.clip(pixels, -1.0, 1.0)
            return pixels
    elif methode == "standardization":
        def method(pixels, stat1, stat2):
            pixels = (pixels - stat1) / stat2
            pixels = np.clip(pixels, -1.0, 1.0)
            pixels = (pixels + 1.0) / 2.0
            return pixels
    elif methode == "normalization11_r":
        def method(pixels, stat1, stat2):
            pixels = (pixels + 1) / 2 * (stat2 - stat1) + stat1
            return pixels
    elif methode == "normalization_r":
        def method(pixels, stat1, stat2):
            pixels = pixels * (stat2 - stat1) + stat1
            return pixels
    elif methode == "centering_r":
        def method(pixels, stat1, stat2):
            return pixels + stat1
    else:
        def method(pixels, stat1, stat2):
            return pixels - stat1
    return method


def reverse_norm_function(methode):
    assert methode in ["normalization11_r", "normalization_r",
                       "centering_r"], "Reverse function not defined for {}".format(methode)
    if methode == "normalization11_r":
        def method(pixels, stat1, stat2):
            pixels = (pixels + 1) / 2 * (stat2 - stat1) + 1
            return pixels
    elif methode == "normalization_r":
        def method(pixels, stat1, stat2):
            pixels = pixels * (stat2 - stat1) + stat1
            return pixels
    else:
        def method(pixels, stat1, stat2):
            return pixels + stat1
    return method


def rescaling(array_dataX, array_label, dict_band_X, dict_band_label, rescale_type="normalization11", plot=False):
    dict_method = DICT_METHOD
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
                               plot=False, dict_stat=None):
    """Rescale combiend on an image"""
    rescaled_arrayX = np.zeros(array_dataX.shape)
    rescaled_label = np.zeros(array_label.shape)
    dict_method = DICT_METHOD
    if dict_rescale_type is None:
        dict_rescale_type = DICT_RESCALE  # by band gives the method used
    for band in dict_rescale_type:
        dx, dlabel = create_dict_bande(band, dict_band_X, dict_band_label)
        rescale_type = dict_rescale_type[band]
        if dict_stat is None:  # we compute the stats each time for each band
            dict_stat = compute_image_stats(array_dataX, array_label, dict_bandX=dx, dictlabel=dlabel,
                                            stats=dict_method[rescale_type], plot=plot)
        rescaled_arrayX[:, :, dict_band_X[band]] = rescaling_function(rescale_type)(
            array_dataX[:, :, dict_band_X[band]], dict_stat[band][0], dict_stat[band][1])
        if band in dlabel:
            rescaled_label[:, :, dict_band_label[band]] = rescaling_function(rescale_type)(
                array_label[:, :, dict_band_label[band]], dict_stat[band][0], dict_stat[band][1])

    return rescaled_arrayX, rescaled_label


def conv1D_dim(tuple_dim):
    return (tuple_dim[0] * tuple_dim[1] * tuple_dim[2] * tuple_dim[3], 1)


def rescale_array(batch_X: np.array, batch_label, dict_group_band_X=None, dict_group_band_label=None,
                  dict_rescale_type=None, s1_log=True, dict_scale=None, invert=False, s2_bands=S2_BANDS,
                  s1_bands=S1_BANDS, fact_scale2=FACTEUR_STD_S2, fact_scale1=FACTEUR_STD_S1, clip_s2=True) -> Tuple[np.array, np.array, dict]:
    """

    Args:
        clip_s2:
        fact_scale: float, the S2 bands will be multiplied by this factor after rescaling. Will only be appled to the
        bands defined in s2_bands
        batch_X: a numpy array
        batch_label: a numpy array , should have the same three first dimensions. Last dimension is the sprectral one
        dict_group_band_X: a dictionnary, gives indication on the index localisation of the band in the batch_X
        dict_group_band_label: a dictionnary, gives indication on the index localisation of the band in the batch_label
        dict_rescale_type: a dictionnary, for each group band defined in the input dictionnaires, gives the string
        method to use, the string should be associated in the method define in sklearn_scale
        s1_log : boolean, if set to True the sar bands are going to pe passed through log10(x+10) function
    Returns:
        rescaled_batch_X : a numpy array, the rescaled batch_X
        ,rescaled_batch_label : a numpy array the rescaled batch_label
        dict_scale : a dictionnary, keys are string and values are the sklearn.processing Scaler. For each group band,
        gives it corresponding Scaler method.


    """
    dict_scaler = {}
    if dict_group_band_label is None:
        dict_group_band_label = DICT_GROUP_BAND_LABEL
    if dict_group_band_X is None:
        dict_group_band_X = DICT_GROUP_BAND_X
    if dict_rescale_type is None:
        dict_rescale_type = DICT_RESCALE_TYPE
    if dict_scale is None:
        dict_scale = {}
        for bands in s1_bands:
            dict_scale.update({bands: None})
        for bands in s2_bands:
            dict_scale.update({bands: None})

    rescaled_batch_X = np.zeros(batch_X.shape)
    rescaled_batch_label = np.zeros(batch_label.shape)
    # we deal with S1 normalization
    for group_bands in s1_bands:
        # all s1 band are in dict_band_X
        data_sar_band = batch_X[:, :, :, dict_group_band_X[group_bands]]
        nbands=len(dict_group_band_X[group_bands])
        if s1_log:
            data_nan_sar = np.copy(data_sar_band)
            data_nan_sar[data_nan_sar < 0] = float("nan")
            print("Remove the negative values in order to have no error in the log : negative value will be replaced using"
                  "knn algorithm")
            data_sar_band = replace_batch_nan_knn(data_nan_sar,[i for i in range(nbands)])
            data_sar_band = 10 * np.log10(data_sar_band)
        init_shape = data_sar_band.shape
        data_flatten_sar_band = data_sar_band.reshape(
            conv1D_dim(data_sar_band.shape))  # Modify into 2D array as required for sklearn
        output_data, sar_scale = sklearn_scale(dict_rescale_type[group_bands], data_flatten_sar_band,
                                               scaler=dict_scale[group_bands], fact_scale=fact_scale1,invert=invert)
        rescaled_batch_X[:, :, :, dict_group_band_X[group_bands]] = output_data.reshape(init_shape)  # reshape it
        dict_scaler.update({group_bands: sar_scale})
    for group_bands in s2_bands:
        m = batch_X.shape[0]  # the nber of element in batch_X
        data = np.concatenate((batch_X[:, :, :, dict_group_band_X[group_bands]],
                               batch_label[:, :, :, dict_group_band_label[group_bands]]))
        global_shape = data.shape
        data_flatten = data.reshape(conv1D_dim(data.shape))

        flat_rescale_data, scale_s2 = sklearn_scale(dict_rescale_type[group_bands], data_flatten,
                                                    scaler=dict_scale[group_bands], invert=invert,
                                                    fact_scale=fact_scale2)
        if clip_s2: #we clip between -1 and 1
            flat_rescale_data=np.clip(flat_rescale_data,DATA_RANGE[0],DATA_RANGE[1])

        rescale_global_data = flat_rescale_data.reshape(global_shape)
        # print("rescale_global_shape {} sub {} fit in {} & label {}".format(rescale_global_data.shape,
        #                                                         rescale_global_data[:m , :, :, :].shape,
        #                                                         rescaled_batch_X[:, :, :, dict_group_band_X[group_bands]].shape,rescaled_batch_label.shape))
        rescaled_batch_X[:, :, :, dict_group_band_X[group_bands]] = rescale_global_data[:m, :, :, :]
        rescaled_batch_label[:, :, :, dict_group_band_label[group_bands]] = rescale_global_data[m:, :, :, :]
        dict_scaler.update({group_bands: scale_s2})

    return rescaled_batch_X, rescaled_batch_label, dict_scaler


def sklearn_scale(scaling_method, data, scaler=None, invert=False, fact_scale=1):
    """
    Args:
        scaling_method: string, name of the method currently only StandardScaler works
        data: input data array to be rescaled
        scaler : a sklearn Scaler
    Returns:
        data_rescale : the rescaled input numpy array (data)
        scaler :  the sklearn.processing method used
    """
    assert scaling_method in ["StandardScaler"], "The method name is not defined {}".format(scaling_method)
    if scaling_method == "StandardScaler":
        if scaler is None:
            print("No scaler was defined before")
            scaler = StandardScaler()
            scaler.fit(data)
        else:
            if invert:
                return scaler.inverse_transform(data * 1 / fact_scale), scaler
        data_rescale = scaler.transform(data)
        return data_rescale * fact_scale, scaler
    else:
        return data, None


def rescale_on_batch(batch_X, batch_label, dict_band_X=None, dict_band_label=None, dict_rescale_type=None,
                     l_s2_stat=None, dict_method=None):
    # TODO remove this function, should ne be used !!!!!
    """

    Args:
        batch_X: numpy array, the input X
        batch_label: numpy array , the label Y
        dict_band_X: dictionnary, gives for each band, its location in the batch_X
        dict_band_label: dictionnary, gives for each band its location (index) in the label
        dict_rescale_type: dictionnary, gives the method name to apply
        l_s2_stat:
        dict_method:

    Returns:

    """
    batch_size = batch_X.shape[0]
    if dict_band_label is None:
        dict_band_label = DICT_BAND_LABEL
    if dict_band_X is None:
        dict_band_X = DICT_BAND_X
    rescaled_batch_X = np.zeros(batch_X.shape)
    rescaled_batch_label = np.zeros(batch_label.shape)
    if dict_rescale_type is None:
        dict_rescale_type = DICT_RESCALE  # by band gives the method used
    # print("Before compute batch stat mean {} min {} max {}".format(np.mean(batch_X),np.min(batch_X),np.max(batch_X)))
    dict_stat = compute_batch_stats(batch_X, batch_label, dict_band_X, dict_band_label, dict_rescale_type,
                                    dict_method=dict_method)
    for i in range(batch_size):  # Rescale all the image on the batch
        if l_s2_stat is not None:  # TODO adapt to extract the mean for the batch
            # print("BEFORE UPDATE {}".format(dict_stat))
            dict_s2_stat = l_s2_stat[
                i]  # WARNING this is hardcoded as we only use batch of 1 !! this should be completly
            for b in dict_s2_stat:  # We replace the s2 value computed by the values from the csv
                assert b in dict_stat.keys(), "The key from the csv stats {} is not in the original dict_stat {}".format(
                    b, dict_stat.keys())
                dict_stat.update({b: dict_s2_stat[b]})
        rescaled_batch_X[i, :, :, :], rescaled_batch_label[i, :, :, :] = rescaling_combined_methods(batch_X[i, :, :, :],
                                                                                                    batch_label[i, :, :,
                                                                                                    :], dict_band_X,
                                                                                                    dict_band_label,
                                                                                                    dict_rescale_type,
                                                                                                    plot=False,
                                                                                                    dict_stat=dict_stat)
    return rescaled_batch_X, rescaled_batch_label


def compute_batch_stats(batch_X, batch_label, dict_band_X, dict_band_label, dict_rescale_type, dict_method=None):
    if dict_method is None:
        dict_method = DICT_METHOD
    batch_size = batch_X.shape[0]
    list_batch_stat = []  # list of all the dict stat of each batch
    dict_stat = dict(zip([i for i in dict_rescale_type], [(0, 0) for i in range(len(dict_rescale_type))]))
    # print(" in compute_batch_stat dict_stat",dict_stat)
    for i in range(batch_size):  # go over all the tiles in the batch tile to compute the stats
        array_dataX = batch_X[i, :, :, :]
        array_label = batch_label[i, :, :, :]
        stat_one_batch = {}
        for band in dict_rescale_type:
            # print("compute stat band ")
            dx, dlabel = create_dict_bande(band, dict_band_X, dict_band_label)  # small dicts with only band
            rescale_type = dict_rescale_type[band]
            sub_dict_stat = compute_image_stats(array_dataX, array_label, dict_bandX=dx, dictlabel=dlabel,
                                                stats=dict_method[rescale_type], plot=False)
            stat_one_batch.update(sub_dict_stat)
            # print("compute stat band ",stat_one_batch)
        list_batch_stat += [stat_one_batch]
    assert len(list_batch_stat) == batch_size, "Not enough stat has been computed {}".format(list_batch_stat)
    # print("THE LIST OF THE BATHC STATS IS {}".format(list_batch_stat))
    # initialize dict
    for i in range(len(list_batch_stat)):
        # print(list_batch_stat[i])
        for band in list_batch_stat[i]:
            # print("band",band,dict_stat)
            stat1 = dict_stat[band][0] + list_batch_stat[i][band][0]
            stat2 = dict_stat[band][1] + list_batch_stat[i][band][1]
            dict_stat.update({band: (stat1, stat2)})
    # DIVIDE BY THE BS
    for band in dict_stat:
        stat1 = dict_stat[band][0] / batch_size
        stat2 = dict_stat[band][1] / batch_size
        dict_stat.update({band: (stat1, stat2)})
    # print("THE BATCH STATISTICS ARE {}".format(dict_stat))
    return dict_stat


def create_dict_bande(band, dict_bandX, dict_band_label):
    """returns a two dictionnary with maximum one key which correspond to band"""
    dX = {}
    dLabel = {}
    if band in dict_bandX:
        dX.update({band: dict_bandX[band]})
    if band in dict_band_label:
        dLabel.update({band: dict_band_label[band]})
    return dX, dLabel


def image_rescaling(data, dict_band, dict_stats, rescale_fun):
    """:param data a np array
    :param dict_band : a dictionnary that gives you all the indication on what band has which index
    :param dict_stats : a dictionnary with the staistic value for each band
    :param rescale_fun : a function, that takes, array, stat1, stat2 ast an input and rescale the array and output a new array"""
    new_array = np.zeros(data.shape)
    for band in dict_band:
        for b_index in dict_band[band]:
            new_array[:, :, b_index] = rescale_fun(data[:, :, b_index], dict_stats[band][0], dict_stats[band][1])
    return new_array


def stat_from_csv(path_tile, dir_csv, dict_translate_band=None):
    """:param path_tile path to the npy tile array svaed
    :param dir_csv path to the directory which contains the nomralized csv file
    :param dict_band a dictionnary which contains the dict_band information
     :returns a dictionnary {B2:(min,max) ...}

    Args:
        dict_translate_band: """
    if dir_csv is None:
        return None
    assert os.path.isdir(dir_csv), "No directory at {}".format(dir_csv)
    assert dir_csv[-1] == "/", "The path to the dir cvs should end with / not {}".format(dir_csv)
    if dict_translate_band is None:
        dict_translate_band = DICT_TRANSLATE_BAND
    image_id = extract_tile_id(path_tile).split(".")[0] + ".tif"
    dict_stat = {}
    for band in dict_translate_band:
        # print("Working with band {}".format(band))
        band_name = dict_translate_band[band]
        min, max = get_minmax_fromcsv(image_id, find_csv(dir_csv, band), band)
        dict_stat.update({band_name: (min, max)})
    # print("The stats found from csv are {}".format(dict_stat))
    return dict_stat

    # apply a normalization without computing the stats !


def get_minmax_fromcsv(tile_id, path_csv, band, convert=CONVERTOR):
    """:param : tile_id a string
    :param path_csv : the directory which contains the csv
    :returns band_min,band_max"""
    id_col = "tile_id"
    if band == "ndvi":
        id_col = "name"
    assert type(band) == type("u"), "The input should be a string not a {}".format(band)
    df = pd.read_csv(path_csv, sep=",", header=0)
    df.head(5)
    # print(df.head(5))
    # print(df.columns)
    name_col = ["{}_min".format(band), "{}_max".format(band)]
    # print("Looking for {}".format(tile_id))
    subf_df = df[df[id_col] == tile_id]
    assert subf_df.shape[0] == 1, "Wrong number of image found {}".format(subf_df)
    dict_res = subf_df.iloc[0].to_dict()
    # print("Resultat min, max from {} : {}".format(path_csv,dict_res))
    # print("We divide the res by this {} as it was used to rescale the data in the dataset ".format(CONVERTOR))
    return dict_res[name_col[0]] / convert, dict_res[name_col[1]] / convert


def get_ndvi_minmax_fromcsv(tile_id, path_csv, vi):  # TODO make one isnge function to read the csv
    assert type(vi) == type("u"), "The input should be a string not a {}".format(vi)
    df = pd.read_csv(path_csv, sep=",", header=0)
    df.head(5)
    # print(df.head(5))
    # print(df.columns)
    name_col = ["{}_min".format(vi), "{}_max".format(vi)]
    subf_df = df[df["tile_id"] == tile_id]
    assert subf_df.shape[0] == 1, "Wrong number of image found {}".format(subf_df)
    dict_res = subf_df.iloc[0].to_dict()
    return dict_res[name_col[0]], dict_res[name_col[1]]


def knn_model(data):
    knn=KNNImputer(n_neighbors=5)
    return knn.fit_transform(data)

def replace_batch_nan_knn(batch,lband_index):
    print("Important the index of the bands in lband_index should be index that follow each other")
    knn_batch=np.copy(batch)
    for b in lband_index:
        list_arr_band=Parallel(n_jobs=-1)(delayed(knn_model)(data) for data in batch[:,:,:,b])
        knn_batch[:,:,:,b]=np.array(list_arr_band)
    return knn_batch