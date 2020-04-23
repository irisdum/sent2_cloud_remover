# All the functions used to load the tiles
from processing import create_safe_directory
import os
from constant.gee_constant import XDIR, LABEL_DIR, DICT_ORGA,DICT_SHAPE
from constant.model_constant import TRAINING_DIR
from scanning_dataset import find_path
from utils.display_image import convert_array
from osgeo import gdal
import numpy as np


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

    if raster_array.shape[-1]>4: #there is the cloud mask which is the 5th band, should be removed
        #TODO think of a better way more generalized to remove the cloud band
        raster_array=raster_array[:4,:]
    raster_array=np.moveaxis(raster_array, 0, -1) #transform to channel last
    #TODO normalization
    return convert_array(raster_array)


def create_input(image_id, input_dir, output_dir):
    """

    :param image_id: id str looks like 01_02.tif
    :param input_dir: path to the dir which contains XDIR and LabelDir
    :param output_dir: path where the npy tiles are going to be stored
    :param info_output_shape: None or a dictionnary which contains the info of the shape of the tiles
    :return:
    """
    assert input_dir[-1] == "/", "Wrong path should end with / not {}".format(input_dir)
    for name_dir in DICT_ORGA:#there will be one tile created one for x one for y (label)
        final_array=np.zeros(DICT_SHAPE[name_dir])
        count_dim=0
        for i,sent in enumerate(DICT_ORGA[name_dir]):
            image_path = find_path(input_dir + name_dir + sent, image_id)
            raster_array = modify_array(tiff_2_array(image_path)) # channel last s*s*channel
            assert raster_array.shape[0]==final_array.shape[0], "Wrong dimension between final_array {} and raster_array {}".format(final_array.shape,raster_array.shape)
            final_array[:,:,count_dim:count_dim+raster_array.shape[-1]]=raster_array #we add the array into the final_array
            count_dim+=raster_array.shape[-1]
        np.save("{}{}.npy".format(output_dir+name_dir,image_id[:-4]),final_array)


def prepare_tiles_from_id(list_id,input_dir, output_dir):
    """Given a list of id and a directory create and save 2 npy array with corresponds to tile on the input of the model and its label
    :param input_dir is the path to the directory which contains dataX dan label directory"""
    for image_id in list_id:
        create_input(image_id,input_dir,output_dir)


def create_input_dataset(dict_tiles,input_dir,output_dir):
    """dict tiles is type {"train/": ["01_02.tif",...],"val/":[list of id],"test/":[list of id] }
    input_dir should contains label/ and DataX
    output_dir will contains the npy tile created"""
    make_dataset_hierarchy(output_dir)
    for sub_dir in dict_tiles:
        assert sub_dir in TRAINING_DIR, "Issue name directory is {} but should be in {}".format(sub_dir,TRAINING_DIR)
        prepare_tiles_from_id(dict_tiles[sub_dir],input_dir,output_dir+sub_dir)

