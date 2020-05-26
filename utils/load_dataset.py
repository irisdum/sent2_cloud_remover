# All the functions used to load the tiles
from processing import create_safe_directory
import os
from constant.gee_constant import XDIR, LABEL_DIR, DICT_ORGA, DICT_SHAPE
from constant.model_constant import TRAINING_DIR
from scanning_dataset import find_path
from utils.display_image import find_image_indir
from utils.display_image import convert_array
from osgeo import gdal
import numpy as np

from utils.normalize import rescale_on_batch, stat_from_csv


def make_dataset_hierarchy(path_dataset):
    assert path_dataset[-1] == "/", "Wrong path should end with / not {}".format(path_dataset)
    create_safe_directory(path_dataset)
    for sub_dir in TRAINING_DIR:
        os.mkdir(path_dataset + sub_dir)
        os.mkdir(path_dataset + sub_dir + XDIR)
        os.mkdir(path_dataset + sub_dir + LABEL_DIR)


def tiff_2_array(path_tif):
    assert os
    raster = gdal.Open(path_tif)
    return raster.ReadAsArray()


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


def create_input(image_id, input_dir, output_dir,normalization=True):
    """

    :param normalization:
    :param image_id: id str looks like 01_02.tif
    :param input_dir: path to the dir which contains XDIR and LabelDir
    :param output_dir: path where the npy tiles are going to be stored
    :return:
    """
    data_x=None
    label=None
    assert input_dir[-1] == "/", "Wrong path should end with / not {}".format(input_dir)
    for name_dir in DICT_ORGA:  # there will be one tile created one for x one for y (label)
        final_array = np.zeros(DICT_SHAPE[name_dir])
        count_dim = 0
        for i, sent in enumerate(DICT_ORGA[name_dir]):
            image_path = find_path(input_dir + name_dir + sent, image_id)
            raster_array = modify_array(tiff_2_array(image_path))  # channel last s*s*channel
            assert raster_array.shape[0] == final_array.shape[
                0], "Wrong dimension between final_array {} and raster_array {}".format(final_array.shape,
                                                                                        raster_array.shape)
            final_array[:, :,
            count_dim:count_dim + raster_array.shape[-1]] = raster_array  # we add the array into the final_array
            count_dim += raster_array.shape[-1]
        if name_dir==XDIR:
            data_x=final_array
        else:
            label=final_array
    if normalization:
        rescale_x,rescale_label=rescale_on_batch(data_x,label)
    else:
        rescale_x,rescale_label=data_x,label
    np.save("{}{}.npy".format(output_dir + XDIR, image_id[:-4]), rescale_x)
    np.save("{}{}.npy".format(output_dir + LABEL_DIR, image_id[:-4]), rescale_label)

def prepare_tiles_from_id(list_id, input_dir, output_dir, norm=True):
    """Given a list of id and a directory create and save 2 npy array with corresponds to tile on the input of the model and its label
    :param input_dir is the path to the directory which contains dataX dan label directory"""
    for image_id in list_id:
        create_input(image_id, input_dir, output_dir,normalization=norm)


def create_input_dataset(dict_tiles, input_dir, output_dir,norm=False):
    """dict tiles is type {"train/": ["01_02.tif",...],"val/":[list of id],"test/":[list of id] }
    input_dir should contains label/ and DataX
    output_dir will contains the npy tile created"""
    make_dataset_hierarchy(output_dir)
    for sub_dir in dict_tiles:
        assert sub_dir in TRAINING_DIR, "Issue name directory is {} but should be in {}".format(sub_dir, TRAINING_DIR)
        prepare_tiles_from_id(dict_tiles[sub_dir], input_dir, output_dir + sub_dir,norm=norm)


def load_data(path_directory, x_shape=None, label_shape=None, normalization=True,dict_band_X=None,dict_band_label=None,dict_rescale_type=None,dir_csv=None):
    """:param path_directory : path to the directory (train,test or val) which contains two directory dataX and label """
    if x_shape is None:
        x_shape = DICT_SHAPE[XDIR]
    if label_shape is None:
        label_shape = DICT_SHAPE[LABEL_DIR]
    assert x_shape[0] == label_shape[0], "Label and data does not have the same dimension label {} data {}".format(
        label_shape, x_shape)
    dataX,path_tileX,ldict_stat= load_from_dir(path_directory + XDIR, x_shape,path_dir_csv=dir_csv) #only need to load once the s
    data_label,path_tile_label,_ = load_from_dir(path_directory + LABEL_DIR, label_shape,path_dir_csv=None)
    print("L_dict_STAT {}".format(ldict_stat))
    if normalization:
        dataX,data_label=rescale_on_batch(dataX,data_label,dict_band_X=dict_band_X,dict_band_label=dict_band_label,
                                          dict_rescale_type=dict_rescale_type,l_s2_stat=ldict_stat)
    assert data_label.shape[0] == dataX.shape[0], "Not the same nber of label {} and dataX {}".format(label_shape,
                                                                                                      x_shape)
    print("The shape of the data are data {} label {}".format(dataX.shape,data_label.shape))
    return dataX, data_label


def load_from_dir(path_dir, image_shape, path_dir_csv=None):
    assert os.path.isdir(path_dir),"Dir {} does not exist".format(path_dir)
    path_tile = find_image_indir(path_dir, "npy")
    batch_x_shape = (len(path_tile), image_shape[0], image_shape[1], image_shape[-1])
    data_array = np.zeros(batch_x_shape)
    if path_dir_csv is None:
        return data_array, path_tile,None
    else:
        ldict_stat=[]
        for i, tile in enumerate(path_tile):
            data_array[i, :, :, :] = np.load(tile)
            ldict_stat += [stat_from_csv(path_tile=tile, dir_csv=path_dir_csv)]
        return data_array,path_tile,ldict_stat



def save_images(images, dir_path,ite=0):
    #print(images.shape)
    if len(images.shape) >3:
        for i in range(images.shape[0]):
            np.save( "{}image_{}_ite{}.npy".format(dir_path, i,ite),images[i, :, :, :])
    else:
        np.save(images, "{}image_ite{}.npy".format(dir_path,ite))