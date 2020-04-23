# Python file to scan the dataset and remove the non regular tile
import glob
import argparse
from osgeo import gdal
import numpy as np
import os
from constant.gee_constant import LISTE_BANDE, XDIR, LABEL_DIR, CLOUD_THR
import random

def is_no_data(raster, sent):
    """Given a raster check if there are no data:
    :returns bool true is there are n data value in one of the band of the raster"""
    n_band = raster.RasterCount
    if sent == 2:
        n_band -= 1  # we do not take into account the lats band which is the cloud mask
    for b in range(n_band):
        # Read the raster band as separate variable
        band = raster.GetRasterBand(b + 1)
        # Check type of the variable 'band'
        # type(band)
        # Data type of the values
        # print("The band {} data_tye {}".format(b, gdal.GetDataTypeName(band.DataType)))
        if band.GetMinimum() is None or band.GetMaximum() is None:
            band.ComputeStatistics(0)
            # print("Statistics computed.")
        # Print only selected metadata:
        if band.GetNoDataValue() is not None:
            print("[ NO DATA VALUE ] = ", band.GetNoDataValue())  # none
            return True

    return False


def is_wrong_size(raster_array):
    size_x, size_y = raster_array.shape[1], raster_array.shape[2]
    if size_x == 256 and size_y == 256:
        pass
    else:
        return True


def is_no_signal(raster_array):
    if np.count_nonzero(raster_array) == 0:
        return True
    else:
        return False


def is_s2_cloud(s2_raster_array, cloud_thr=CLOUD_THR):
    """Given a sentinel 2 raster check if the cloud mask band contains cloud pixels value at 2**16-1
    :returns bool """
    cloud_mask_array = s2_raster_array[-1, :, :]
    nb_cloud = np.count_nonzero(cloud_mask_array == 2)
    nb_shadow = np.count_nonzero(cloud_mask_array == 3)
    if nb_cloud + nb_shadow > CLOUD_THR:
        return True
    else:
        return False


def extract_relative_path(path_tif):
    """Given the path to an tif tile returns its relative path within the Sentineli_tj directory"""
    l = path_tif.split("/")
    return "/".join(l[-3:-1])


def extract_tile_id(path_tif):
    return path_tif.split("/")[-1][-9:]


def get_all_tiles_path(path_sent_dir):
    """Given the path to Sentineli_tj directory returns a list of all the paths to all the tiles of the image"""
    assert os.path.isdir(path_sent_dir), "The dir {} does not exist".format(path_sent_dir)
    print("research :  {}**/*.tif".format(path_sent_dir))
    l = glob.glob("{}**/*.tif".format(path_sent_dir), recursive=True)
    assert len(l) > 0, "No image found in {}".format(path_sent_dir)
    return l


def select_rdtiles(list_all_tiles, nb_sample=10, seed=2):
    random.seed(seed)

    assert len(list_all_tiles) > 0, "No tiles found in {}".format(list_all_tiles)
    sample_path = random.sample(list_all_tiles, nb_sample)
    return [extract_tile_id(path) for path in sample_path]


def list_all_conformed_tiles(path_final_dataset, sent_dir="dataX/Sentinel1_t1/"):
    l_unconformed_id = get_unconformed(path_final_dataset)
    list_all_tiles = glob.glob(path_final_dataset + sent_dir + "**/*.tif", recursive=True)
    print("Initial dataset size {}".format(len(list_all_tiles)))
    l_all_id = [extract_tile_id(path) for path in list_all_tiles]
    print(l_all_id[:10])
    for id_tile in l_unconformed_id:
        # print(id_tile in l_unconformed_id)
        # print("remove {} ".format(id_tile))
        l_all_id.remove(id_tile)
    print("Final dataset size {}".format(len(l_all_id)))
    return l_all_id

def find_path(sent_dir, image_id):
    """:returns a string which is the path of the image with the id image_id
    """
    assert os.path.isdir(sent_dir),"Wrong path to dir, path  {} does not exist ".format(sent_dir)
    print(sent_dir + "**/*{}".format(image_id))
    l = glob.glob(sent_dir + "**/*{}".format(image_id), recursive=True)
    assert len(l) > 0, "No image found with id {} at {}".format(image_id, sent_dir)
    return l[0]


def is_conform(path_tile):
    raster = gdal.Open(path_tile)
    raster_array = raster.ReadAsArray()
    assert raster_array.shape[0] in [len(LISTE_BANDE[0]),
                                     len(LISTE_BANDE[1])], "Wrong tile shape {} should be {} or {} bands" \
        .format(raster.shape, len(LISTE_BANDE[0]), len(LISTE_BANDE[1]))

    if is_wrong_size(raster_array):
        print("Image {} wrong size ".format(path_tile.split("/")[-1]))
        return False

    if is_no_signal(raster_array):
        print("Image {} only 0 ".format(path_tile.split("/")[-1]))
        return False

    if raster_array.shape[0] == 5:  # check sentinel 2 conformity
        if is_s2_cloud(raster_array):
            print("Image {} clouds ".format(path_tile.split("/")[-1]))
            return False
        if is_no_data(raster, 2):
            print("Image {} no data ".format(path_tile.split("/")[-1]))
            return False
    else:
        if is_no_data(raster, 1):
            print("Image {} no_data ".format(path_tile.split("/")[-1]))
            return False

    return True


def get_unconformed(path_final_dataset):
    list_sent_dir = [path_final_dataset + XDIR + "Sentinel1_t0/", path_final_dataset + XDIR + "Sentinel1_t1/",
                     path_final_dataset + XDIR + "Sentinel2_t0/", path_final_dataset + LABEL_DIR + "Sentinel2_t1/"]
    list_not_conform = []
    for path_sent_dir in list_sent_dir:
        list_tiles = get_all_tiles_path(path_sent_dir)
        for path_tile in list_tiles:
            if is_conform(path_tile):
                pass
            else:
                # add it to the list of all the not appropriate tiles
                list_not_conform += [extract_tile_id(path_tile)]
    return list(set(list_not_conform))


def main(path_final_dataset, opt_remove=False):
    list_sent_dir = [path_final_dataset + XDIR + "Sentinel1_t0", path_final_dataset + XDIR + "Sentinel1_t1",
                     path_final_dataset + XDIR + "Sentinel2_t0", path_final_dataset + LABEL_DIR + "Sentinel2_t1"]
    list_not_conform = []
    for path_sent_dir in list_sent_dir:
        list_tiles = get_all_tiles_path(path_sent_dir)

        for path_tile in list_tiles:
            if is_conform(path_tile):
                pass
            else:
                # add it to the list of all the not appropriate tiles
                list_not_conform += [extract_relative_path(path_tile)]
                if opt_remove:
                    pass
    print(list(set(list_not_conform)))




def _argparser():
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('--input_dir', type=str, default="/datastore/dum031/data/dataset1/prepro7/build_dataset/",
                        help="path to input  build_dataset directory ")
    parser.add_argument('--remove', type=bool, default=False,
                        help="wether to remove the unconformed tiles from the input directory")

    return parser.parse_args()


if __name__ == '__main__':
    args = _argparser()
    main(args.input_dir, args.remove)
