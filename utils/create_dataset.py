import os
from typing import List

import numpy as np
from osgeo import gdal

from constant.model_constant import TRAINING_DIR
from constant.storing_constant import DICT_ORGA_INT, DICT_ORGA, XDIR, LABEL_DIR
from utils.image_find_tbx import find_path, create_safe_directory
from utils.load_dataset import count_channel, modify_array
from utils.normalize import rescale_on_batch


def create_input(image_id: str, input_dir: str, output_dir: str, normalization=False, tile_size=256):
    """

    Args:
        image_id: string, corresponds to the image_id (line_id,column_id) that comes from the tiling
        input_dir: string, path to the input directory which contains XDIR and LabelDir
        output_dir: string, path to the output dir where the npy tiles are going to be stored
        normalization: bool if used normalization is applied, recommended not to use it

    Returns:
        For an id, combine the bands found in storing_constant.DICT_ORGA, and respect the order of the dirctory
    """
    data_x = None
    label = None
    assert input_dir[-1] == "/", "Wrong path should end with / not {}".format(input_dir)
    final_shape = count_channel(DICT_ORGA_INT, tile_size=tile_size)
    for name_dir in DICT_ORGA:  # there will be one tile created one for x one for y (label)
        final_array = np.zeros(final_shape[name_dir])
        count_dim = 0
        for i, sent in enumerate(DICT_ORGA[name_dir]):  # goes through the list of subdirectories (Sent**date**)
            image_path = find_path(input_dir + name_dir + sent, image_id)
            raster_array = modify_array(tiff_2_array(image_path))  # channel last s*s*channel
            assert raster_array.shape[0] == final_array.shape[
                0], "Wrong dimension between final_array {} and raster_array {}".format(final_array.shape,
                                                                                        raster_array.shape)
            final_array[:, :,
            count_dim:count_dim + raster_array.shape[-1]] = raster_array  # we add the array into the final_array
            count_dim += raster_array.shape[-1]
        if name_dir == XDIR:
            data_x = final_array
        else:
            label = final_array
    if normalization:
        rescale_x, rescale_label = rescale_on_batch(data_x,
                                                    label)  # TODO modify the use of rescale, or remove normalization optio
    else:
        rescale_x, rescale_label = data_x, label
    np.save("{}{}.npy".format(output_dir + XDIR, image_id[:-4]), rescale_x)
    np.save("{}{}.npy".format(output_dir + LABEL_DIR, image_id[:-4]), rescale_label)


def tiff_2_array(path_tif: str):
    assert os
    raster = gdal.Open(path_tif)
    return raster.ReadAsArray()


def prepare_tiles_from_id(list_id: List[str], input_dir: str, output_dir: str, norm=False, tile_size=256):
    """
    This function goes through a list of id. For each idea the create_input function is applied :
     we create dataX tile and label tile
    Args:
        list_id: list of string, list of id
        input_dir: string,the path to the directory which contains dataX dan label directory
        output_dir: string, path to the output directory
        norm: boolean, if True a normalization is applied (not recommended)

    Returns:

    """

    for image_id in list_id:
        create_input(image_id, input_dir, output_dir, normalization=norm, tile_size=tile_size)


def create_input_dataset(dict_tiles: dict, input_dir: str, output_dir: str, norm=False, tile_size=256):
    """
    Args:
        tile_size: int, the tile size the output tiles will be (tile_size,tile_size,nchannel) dimension
        dict_tiles: dictionnary, describe the tile id used for train, val and test
         ex :  {"train/": ["01_02.tif",...],"val/":[list of id],"test/":[list of id] }
        input_dir: string, path to directory contains label/ and DataX/
        output_dir: string, path to the global dir that will contains will contains the npy tile created
        norm: boolean, recommended False
    Returns:

    """

    make_dataset_hierarchy(output_dir)
    for sub_dir in dict_tiles:
        assert sub_dir in TRAINING_DIR, "Issue name directory is {} but should be in {}".format(sub_dir, TRAINING_DIR)
        prepare_tiles_from_id(dict_tiles[sub_dir], input_dir, output_dir + sub_dir, norm=norm, tile_size=tile_size)


def make_dataset_hierarchy(path_dataset: str):
    assert path_dataset[-1] == "/", "Wrong path should end with / not {}".format(path_dataset)
    create_safe_directory(path_dataset)
    for sub_dir in TRAINING_DIR:
        os.mkdir(path_dataset + sub_dir)
        os.mkdir(path_dataset + sub_dir + XDIR)
        os.mkdir(path_dataset + sub_dir + LABEL_DIR)