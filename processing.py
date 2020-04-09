# File which contains all the processing functions , mostly using snap
import argparse
import glob
import os
import shutil
import click

from utils.converter import geojson_2_bboxcoordo, geojson_2_strcoordo_ul_lr
from constant.gee_constant import VAR_NAME, LISTE_BANDE, OVERLAP, TEMPORARY_DIR, XDIR, LABEL_DIR, DIR_T


def crop_image(image_path, path_geojson, output_path):
    assert os.path.isfile(path_geojson), "No path in {}".format(path_geojson)
    # assert os.path.isdir(output_dir),"No dir in {}".format(output_dir)
    str_bbox = geojson_2_strcoordo_ul_lr(path_geojson)
    # print("gdalwarp -cutline  SHAPE_RESTORE_SHX=YES {} {} {}".format(path_shapefile, image_path, output_path))
    # os.system("gdalwarp -cutline  {} {} {}".format(path_shapefile, image_path, output_path)

    os.system("gdal_translate {} {} -a_ullr  {} ".format(image_path, output_path, str_bbox))
    return output_path


def mosaic_image(list_path, output_dir):
    """Given the path to multiple images of the same band create a mosaic"""
    output_name = get_band_image_name(list_path[0], output_dir)

    os.system("gdalbuildvrt  {} {}".format(output_name, list_2_str(list_path)))
    assert os.path.isfile(output_name), "The file has not been created at {}".format(output_name)
    return output_name


def combine_band(list_path_vrt, output_dir):
    """Given a list of all vrt file for a sentinel"""
    output_name = get_name_sent_vrt(list_path_vrt[0], output_dir)
    print("BAND COMBINATION  : gdalbuildvrt -separate {} {}".format(output_name, list_2_str(list_path_vrt)))
    os.system("gdalbuildvrt -separate {} {}".format(output_name, list_2_str(list_path_vrt)))
    print("AFTER COMBINE ")
    os.system("gdalinfo {}".format(output_name))
    return output_name


def compute_the_intersection(image_path, zone_path):
    """:param image_path : path to the image
    :param zone_path to the zone geometry"""
    pass


def orthorectification(image_path, output_path):
    pass


def get_path_tile(band, input_dir):
    """Given the input directory returns a list of all the tiles which representes this band"""
    assert os.path.isdir(input_dir), "Wrong input directory {}".format(input_dir)
    assert input_dir[-1] == "/", "The path of the input dir should end with /"
    l = glob.glob("{}{}*.tif".format(input_dir, band))
    assert len(l) > 0, "No images {}{}*.tif found".format(input_dir, band)
    return l


def list_2_str(list):
    ch = ""
    for p in list:
        ch += "{} ".format(p)
    print(ch)
    return ch


def tiling(image_vrt, output_dir, sent=1, date_t=0):
    print("IMAGE VRT which is going to be tiled {}".format(image_vrt))
    #os.system("gdalinfo {}".format(image_vrt))
    os.system("gdal_retile.py {} -targetDir {} -tileIndex {} --optfile {}".format(image_vrt,output_dir,"tiling_sent{}_t{}_fp.shp".format(sent,date_t),"confs/retile_optfile.txt"))
    return output_dir + "tiling_fp.shp"


def get_band_image_name(image_path, output_dir):
    assert output_dir[-1] == "/", "The path of output dir should end with / {}".format(output_dir)
    image_name = image_path.split("/")[-1]
    return output_dir + image_name.split(VAR_NAME)[0] + ".vrt"


def get_name_sent_vrt(band_vrt, output_dir):
    #print(band_vrt)
    #print(band_vrt.split("/"))
    return output_dir + band_vrt.split("/")[-1][3:]


def reproject_sent(path_image, output_dir, path_geojson):
    name = path_image.split("/")[-1]
    str_bbox = geojson_2_bboxcoordo(path_geojson)
    #print("STR BBOX {}".format(str_bbox))
    #print("BEFORE WARP ")
    os.system("gdalinfo {} ".format(path_image))
    os.system("gdalwarp -t_srs EPSG:4326  {} {}".format(path_image, output_dir + name))
    #print("AFTER WARP {}")
    os.system("gdalinfo {}".format(output_dir+name))
    return output_dir + name


def create_safe_directory(output_dir):
    if os.path.isdir(output_dir):
        if click.confirm(
                'The directory {} already exists, it will remove it do you want to continue?'.format(output_dir),
                default=True):
            print('Ok remove')
            shutil.rmtree(output_dir)
        else:
            return False
    os.makedirs(output_dir)


def _argparser():
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('--input_dir', type=str, default="/datastore/dum031/data/dataset1/prepro7/",
                        help="path to input directory ")
    parser.add_argument('--output_dir', type=str, default="/datastore/dum031/data/dataset1/prepro7/build_dataset/",
                        help="path to the output directory")
    parser.add_argument("--bands2", nargs="+", default=None, help="list of all the bands of sentinel 2 format B02,B03")

    parser.add_argument("--bands1", nargs="+", default=None, help="list of all the bands of sentinel1 format vv, vh")
    parser.add_argument("--geojson", default="./confs/train_kangaroo.geojson", help="path to the zone geojson")
    parser.add_argument("--shp", default="./confs/train_kangaroo.shp", help="path to the esri shapefile")
    return parser.parse_args()


def create_tiling_hierarchy(output_dir):
    create_safe_directory(output_dir)
    for cst in [XDIR, LABEL_DIR]:
   #     print("BUILDING DATA {}".format(cst))
        create_safe_directory(output_dir + cst)
        create_safe_directory(output_dir + cst + TEMPORARY_DIR)


def main(input_dir, output_dir, list_band2, list_band1, path_geojson):
    create_tiling_hierarchy(output_dir)
    ## Create the dataX folder
    build_tiling_sent(list_band1, 1, input_dir, output_dir, XDIR, 0, path_geojson)  # sentinel1 at t1
    build_tiling_sent(list_band1, 1, input_dir, output_dir, XDIR, 1, path_geojson)  # sentinel1 at t2
    build_tiling_sent(list_band2, 2, input_dir, output_dir, XDIR, 0, path_geojson)  # sentinel2 at t1
    ##LABEL FOLDER
    build_tiling_sent(list_band2, 2, input_dir, output_dir, LABEL_DIR, 1, path_geojson)  # sentinel2 at t2


def build_tiling_sent(list_band, sent, input_dir, output_dir, sub_dir, t, path_geojson):
    """Given a Sentinel and a time build the tiles """
    input_dir_t = input_dir + DIR_T[t]
    list_name_band = create_vrt(list_band, sent, input_dir_t, output_dir + sub_dir + TEMPORARY_DIR,
                                path_geojson)
    output_dir_tile = output_dir + sub_dir+ "Sentinel{}_t{}/".format(sent, t)
    tiling_sent(list_name_band, sent, output_dir_tile, path_geojson, t)


def tiling_sent(list_image, sent, output_dir, path_geojson, t):
    create_safe_directory(output_dir)

    total_image = combine_band(list_image, output_dir)
    #print("BEFORE CROP")
    crop_image_name = crop_image(total_image, path_geojson,
                                 output_dir + "merged_crop_sent{}_t{}.vrt".format(sent, t))
    #print("AFTER CROP")
    os.system("gdalinfo {}".format(crop_image_name))
    shp_file_t1 = tiling(crop_image_name, output_dir,sent,t)


def create_vrt(list_band, sent, input_dir, output_dir, path_geojson):
    """Given these parameters construct VRT format image For each bands create a mosaic if needed"""
    list_band_vrt = []
    if list_band is None and sent == 2:
        list_band = [b.lower().replace("0", "") for b in
                     LISTE_BANDE[1]]  # liste band of sentinel 2, convert it from B02->b2
    if list_band is None and sent == 1:
        list_band = LISTE_BANDE[0]
    for b in list_band:
        # reprojection of sentinel 2 images and warp on the input_geojon
        list_image = get_path_tile(b, input_dir)
        output_name = mosaic_image(list_image, input_dir)
        print("The image {} has been created".format(output_name))
        output_name = reproject_sent(output_name, output_dir, path_geojson)
        list_band_vrt += [output_name]
    return list_band_vrt


if __name__ == '__main__':
    args = _argparser()
    main(args.input_dir, args.output_dir, args.bands2, args.bands1, args.geojson)
