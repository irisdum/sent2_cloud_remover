# All the functions used to load the tiles
import os

import numpy as np
from joblib import Parallel, delayed

from constant.gee_constant import LISTE_BANDE
from constant.processing_constant import FACT_STD_S2, S1_BANDS, S2_BANDS, FACT_STD_S1
from constant.storing_constant import XDIR, LABEL_DIR, DICT_ORGA_INT
from utils.converter import convert_array
from utils.image_find_tbx import find_image_indir
from utils.normalize import stat_from_csv, rescale_array


def modify_array(raster_array):
    """raster array is a numpy array usually channel*256*256
    This function modify the array in channel 256*256*channel
    It converts the data type
    The normalization is also taken applied"""

    if raster_array.shape[-1] > 4:  # there is the cloud mask which is the 5th band, should be removed
        # TODO think of a better way more generalized to remove the cloud band
        raster_array = raster_array[:4, :]
    raster_array = np.moveaxis(raster_array, 0, -1)  # transform to channel last
    # TODO normalization
    return convert_array(raster_array)


def count_channel(dict_orga_int=None, tile_size=256) -> dict:
    """

    Args:
        dict_orga_int : a dictionnary which has for each keys, a list of tuple (sent,t)
    Returns:
        A dictionnary which gives for each key of dict_orga_int the nber of bands used.i.e for each Sent the nber
        of bands selected

    """
    channel_dic = {}
    if dict_orga_int is None:
        dict_orga_int = DICT_ORGA_INT
    for key in dict_orga_int:
        count = 0
        for sent, t in dict_orga_int[key]:
            count += len(LISTE_BANDE[sent - 1])  # add the nber of bands which corresponds to sentinel images downloaded
        shape_tuple = (tile_size, tile_size, count)
        channel_dic.update({key: shape_tuple})
    return channel_dic


def load_data(path_directory: str, x_shape=None, label_shape=None, normalization=True, dict_band_X=None,
              dict_band_label=None, dict_rescale_type=None, dict_scale=None, fact_s2=FACT_STD_S2,
              fact_s1=FACT_STD_S1, s2_bands=S2_BANDS, s1_bands=S1_BANDS, clip_s2=True,lim=None):
    """

    Args:
        lim: None or int. if int the max nber of tile which is goint to loaded
        clip_s2:
        dict_scale: None or a dictionnary, the keys are string which correspond to group band names and the values
         are sklearn scaler
        path_directory: string, path to the directory which contains the tiles, (train,test or val)which contains two directory dataX and label
        x_shape: tuple, dimension of the tiles for the tiles considered as the input in the NN (dataX)
        label_shape: tuple, dimension of the tiles considered as label in the NN
        normalization: #TODO modify this parameter
        dict_band_X: dictionary, contains as the keys the band and values are the list of index on which the band are
         placed in the tiles
        dict_band_label:dictionary, as above for the label tiles
        dict_rescale_type: dictionary, for each type of band gives the type of rescaling used

    Returns:

    """
    if x_shape is None:
        x_shape = count_channel(DICT_ORGA_INT)[XDIR]
    if label_shape is None:
        label_shape = count_channel(DICT_ORGA_INT)[LABEL_DIR]
    assert x_shape[0] == label_shape[0], "Label and data does not have the same dimension label {} data {}".format(
        label_shape, x_shape)
    dataX, path_tileX, _ = load_from_dir(path_directory + XDIR, x_shape,lim=lim)  # only need to load once the s
    data_label, path_tile_label, _ = load_from_dir(path_directory + LABEL_DIR, label_shape,lim=lim)
    # print("L_dict_STAT {}".format(ldict_stat))
    if normalization:
        dataX, data_label, dict_scale = rescale_array(dataX, data_label, dict_group_band_X=dict_band_X,
                                                      dict_group_band_label=dict_band_label,
                                                      dict_rescale_type=dict_rescale_type, s1_log=True,
                                                      dict_scale=dict_scale, s2_bands=s2_bands, s1_bands=s1_bands,
                                                      fact_scale2=fact_s2, fact_scale1=fact_s1,clip_s2=clip_s2)
        return dataX, data_label, dict_scale
    assert data_label.shape[0] == dataX.shape[0], "Not the same nber of label {} and dataX {}".format(label_shape,
                                                                                                      x_shape)
    # print("The shape of the data are data {} label {}".format(dataX.shape,data_label.shape))
    return dataX, data_label, None


def load_from_dir(path_dir: str, image_shape: tuple, lim=None):
    """

    Args:
        lim: int or None. if int the max nber of tile which is going to be taken into account
        path_dir: string, path to the directory
        image_shape: tuple, shape of the image

    Returns:
        a numpy array, the list of the tif tile used, the image shape
    """
    assert os.path.isdir(path_dir), "Dir {} does not exist".format(path_dir)
    path_tile = find_image_indir(path_dir, "npy")
    if lim is not None:
        path_tile = path_tile[:lim]  # list of all
    batch_x_shape = (len(path_tile), image_shape[0], image_shape[1], image_shape[-1])
    # data_array = np.zeros(batch_x_shape)
    data_array = np.array(Parallel(n_jobs=1)(delayed(load_one_tile)(tile) for tile in path_tile))
    # for i, tile in enumerate(path_tile):
    #  assert os.path.isfile(tile), "Wrong path to tile {}".format(tile)
    #  data_array[i, :, :, :] = np.load(tile)
    assert data_array.shape == batch_x_shape, "Wrong dimension of the data loaded is {} expected ".format(
        data_array.shape,
        batch_x_shape)
    return data_array, path_tile, None


def load_one_tile(path_tile):
    assert os.path.isfile(path_tile), "Wrong path to tile {}".format(path_tile)
    return np.load(path_tile)


def csv_2_dictstat(path_tile: str, path_dir_csv: str):
    ldict_stat = []
    for i, tile in enumerate(path_tile):
        ldict_stat += [stat_from_csv(path_tile=tile, dir_csv=path_dir_csv)]
    return ldict_stat


def save_images(images, dir_path, ite=0):
    # print(images.shape)
    if len(images.shape) > 3:
        for i in range(images.shape[0]):
            np.save("{}image_{}_ite{}.npy".format(dir_path, i, ite), images[i, :, :, :])
    else:
        np.save(images, "{}image_ite{}.npy".format(dir_path, ite))
