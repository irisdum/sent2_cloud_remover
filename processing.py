# File which contains all the processing functions , mostly using snap
import argparse
import glob
import os

from tiling import mosaic_image, combine_band, crop_image, tiling
from utils.converter import geojson_2_strcoordo_ul_lr
from constant.gee_constant import LISTE_BANDE, TEMPORARY_DIR, XDIR, LABEL_DIR, DIR_T, EPSG
from utils.image_find_tbx import create_safe_directory
from utils.storing_data import create_tiling_hierarchy


def get_band_scale(raster, b):
    band = raster.GetRasterBand(b)
    if band.GetMinimum() is None or band.GetMaximum() is None:
        band.ComputeStatistics(0)
    return band.GetMinimum(), band.GetMaximum()


def get_path_tile(band, input_dir2, opt="img"):
    """Given the input directory returns a list of all the tiles which representes this band"""
    assert os.path.isdir(input_dir2), "Wrong input directory {}".format(input_dir2)
    assert input_dir2[-1] == "/", "The path of the input dir should end with /"
    l = glob.glob("{}**{}*.{}".format(input_dir2, band, opt), recursive=True)  # In each .data dir take the img image
    assert len(l) > 0, "No images {}{}*.{} found".format(input_dir2, band, opt)
    return l


def reproject_sent(path_image, output_dir, path_geojson):
    """

    Args:
        path_image: string, path
        output_dir:
        path_geojson:

    Returns:

    """
    name = path_image.split("/")[-1]
    str_bbox = geojson_2_bboxcoordo(path_geojson)
    # print("STR BBOX {}".format(str_bbox))
    # print("BEFORE WARP ")
    os.system("gdalinfo {} ".format(path_image))
    os.system("gdalwarp -t_srs {}  {}  -tr 10 10 {}".format(EPSG, path_image, output_dir + name))
    # print("AFTER WARP {}")
    # os.system("gdalinfo {}".format(output_dir+name))
    return output_dir + name


def _argparser():
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('--input_dir', type=str, default="/datastore/dum031/data/dataset1/prepro7/",
                        help="path to input directory ")
    parser.add_argument('--output_dir', type=str, default="/datastore/dum031/data/dataset1/prepro7/build_dataset/",
                        help="path to the output directory")
    parser.add_argument("--bands2", nargs="+", default=None, help="list of all the bands of sentinel 2 format B02,B03")

    parser.add_argument("--bands1", nargs="+", default=None, help="list of all the bands of sentinel1 format vv, vh")
    parser.add_argument("--geojson", default="./confs/train_kangaroo_utm2.geojson", help="path to the zone geojson")
    parser.add_argument("--overlap", type=int, default=0, help="path to the zone geojson")
    return parser.parse_args()


def main(input_dir, output_dir, list_band2, list_band1, path_geojson, overlap):
    create_tiling_hierarchy(output_dir)
    ## Create the dataX folder
    build_tiling_sent(list_band1, 1, input_dir, output_dir, XDIR, 0, path_geojson, overlap=overlap)  # sentinel1 at t1
    build_tiling_sent(list_band2, 2, input_dir, output_dir, XDIR, 0, path_geojson, overlap=overlap)  # sentinel2 at t1
    build_tiling_sent(list_band1, 1, input_dir, output_dir, XDIR, 1, path_geojson, overlap=overlap)  # sentinel1 at t2
    ##LABEL FOLDER
    build_tiling_sent(list_band2, 2, input_dir, output_dir, LABEL_DIR, 1, path_geojson,
                      overlap=overlap)  # sentinel2 at t2


def build_tiling_sent(list_band, sent, input_dir, output_dir, sub_dir, t, path_geojson, overlap):
    """Given a Sentinel and a time build the tiles """
    input_dir_t = input_dir + DIR_T[t]
    list_name_band = create_vrt(list_band, sent, input_dir_t, output_dir + sub_dir + TEMPORARY_DIR,
                                path_geojson)
    output_dir_tile = output_dir + sub_dir + "Sentinel{}_t{}/".format(sent, t)
    tiling_sent(list_name_band, sent, output_dir_tile, path_geojson, t, overlap=overlap)


def tiling_sent(list_image, sent, output_dir, path_geojson, t, overlap):
    """

    Args:
        list_image:
        sent:
        output_dir:
        path_geojson:
        t:
        overlap:

    Returns:

    """
    create_safe_directory(output_dir)
    if sent == 2:
        total_image = combine_band(list_image, output_dir)  # for Sentinel 2 combien the images
    else:
        assert len(list_image) == 1, "More than One image of S1 is found {}".format(list_image)
        total_image = list_image[0]
    # print("BEFORE CROP")
    crop_image_name = crop_image(total_image, path_geojson,
                                 output_dir + "merged_crop_sent{}_t{}.vrt".format(sent, t))
    # print("AFTER CROP")
    os.system("gdalinfo {}".format(crop_image_name))
    shp_file_t1 = tiling(crop_image_name, output_dir, sent, t, overlap=overlap)


def tiling_aus18_map(path_tif, output_dir, path_geojson):
    """Function used to tile the maps of the australian forest vegetation, into the same tiling process of the build_dataset"""

    crop_image_name = crop_image(path_tif, path_geojson,
                                 output_dir + "crop_aus18.vrt")

    os.system("gdalinfo {}".format(crop_image_name))
    shp_file_t1 = tiling(crop_image_name, output_dir, 4, 0)


def create_vrt(list_band, sent, input_dir, output_dir, path_geojson):
    """Given these parameters construct VRT format image For each bands create a mosaic if needed"""
    list_band_vrt = []
    if list_band is None and sent == 2:
        list_band = [b.lower().replace("0", "") for b in
                     LISTE_BANDE[1]]  # liste band of sentinel 2, convert it from B02->b2
        # list_band= LISTE_BANDE[1]
    if list_band is None and sent == 1:
        list_band = LISTE_BANDE[0]
    for b in list_band:
        # reprojection of sentinel 2 images and warp on the input_geojon
        list_image = get_path_tile(b, input_dir)
        print("[INFO] for sent {} we found {} for band {}".format(sent, list_image, b))
        output_name = mosaic_image(list_image, input_dir)  # just regroup the image if they belong to the same mosaic
        print("The image {} has been created".format(output_name))
        # output_name = reproject_sent(output_name, output_dir, path_geojson) #Not needed
        # if sentinel 2 : convert to Float 32
        # if sent==2:
        #     output_name=convert2float32(output_name, output_dir)
        list_band_vrt += [output_name]
    return list_band_vrt


if __name__ == '__main__':
    args = _argparser()
    main(args.input_dir, args.output_dir, args.bands2, args.bands1, args.geojson, args.overlap)
