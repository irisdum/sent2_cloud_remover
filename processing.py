# File which contains all the processing functions , mostly using snap
import argparse
import glob
import os

from converter import geojson_2_bboxcoordo
from gee_constant import VAR_NAME, LISTE_BANDE, OVERLAP, TEMPORARY_DIR, TILING_DIR


def crop_image(image_path, path_shapefile, output_path):
    assert os.path.isfile(path_shapefile), "No path in {}".format(path_shapefile)
    # assert os.path.isdir(output_dir),"No dir in {}".format(output_dir)
    print("gdalwarp -cutline  SHAPE_RESTORE_SHX=YES {} {} {}".format(path_shapefile, image_path, output_path))
    os.system("gdalwarp -cutline   {} {} {}".format(path_shapefile, image_path, output_path))
    return output_path

def mosaic_image(list_path, output_dir,path_geojson):
    """Given the path to multiple images of the same band create a mosaic"""
    output_name = get_band_image_name(list_path[0], output_dir)
    str_bbox=geojson_2_bboxcoordo(path_geojson)
    os.system("gdalbuildvrt -te {} {} {}".format(str_bbox,output_name, list_2_str(list_path)))
    return output_name


def combine_band(list_path_vrt,output_dir):
    """Given a list of all vrt file for a sentinel"""
    output_name = get_name_sent_vrt(list_path_vrt[0],output_dir)
    os.system("gdalbuildvrt -separate {} {}".format(output_name, list_path_vrt))


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
    return ch


def tiling(image_vrt, output_dir):
    os.system("gdal_retile.py {}  -targetDir {} -overlap {} -v -tileIndex {}".format(image_vrt, output_dir, OVERLAP,
                                                                                     "Tiling_fp"))


def get_band_image_name(image_path, output_dir):
    assert output_dir[-1] == "/", "The path of output dir should end with / {}".format(output_dir)
    image_name = image_path.split("/")[-1]
    return output_dir + image_name.split(VAR_NAME)[0] + ".vrt"


def get_name_sent_vrt(band_vrt,output_dir):
    return output_dir+band_vrt.split("/")[-1][3:]


def _argparser():
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('--input_dir', type=str, default="/datastore/dum031/data/dataset1/prepro7/date2/",
                        help="path to input directory ")
    parser.add_argument('--output_dir', type=str, default="/datastore/dum031/data/dataset1/prepro7/date2/",
                        help="path to the output directory")
    parser.add_argument("--bands2", nargs="+", default=None, help="list of all the bands of sentinel 2 format B02,B03")

    parser.add_argument("--bands1", nargs="+", default=None, help="list of all the bands of sentinel1 format vv, vh")
    parser.add_argument("--geojson", default="./confs/train_kangaroo.geojson", help="path to the zone geojson")
    return parser.parse_args()


def main(input_dir, output_dir, list_band2, list_band1,path_geojson):
    if list_band2 is None:
        list_band2 = [b.lower().replace("0", "") for b in
                      LISTE_BANDE[1]]  # liste band of sentinel 2, convert it from B02->b2
    if list_band1 is None:
        list_band1 = LISTE_BANDE[0]
    list_name_band_sent2_vrt = []
    list_name_band_sent1_vrt = []
    for b in list_band2:
        list_image = get_path_tile(b, input_dir)
        output_name = mosaic_image(list_image, output_dir+TEMPORARY_DIR,path_geojson)
        print("The image {} has been created".format(output_name))
        list_name_band_sent2_vrt += [output_name]
    for b in list_band1:
        list_image = get_path_tile(b, input_dir)
        output_name = mosaic_image(list_image, output_dir+TEMPORARY_DIR,path_geojson)
        print("The image {} has been created".format(output_name))
        list_name_band_sent1_vrt += [output_name]
    print("Sentinel 1 {} Sentinel 2 {}".format(list_name_band_sent1_vrt, list_name_band_sent2_vrt))
    combine_band(list_name_band_sent2_vrt + list_name_band_sent1_vrt,output_dir+TILING_DIR)


if __name__ == '__main__':
    args = _argparser()
    main(args.input_dir, args.output_dir, args.bands2,args.bands1,args.geojson)
