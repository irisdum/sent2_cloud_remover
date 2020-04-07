# File which contains all the processing functions , mostly using snap
import argparse
import glob
import os

from gee_constant import VAR_NAME, LISTE_BANDE


def mosaic_image(list_path, output_dir):
    """Given the path to multiple images of the same band create a mosaic"""
    output_name = get_band_image_name(list_path[0], output_dir)
    os.system("gdal_merge.py -o {} {}".format(output_name, list_2_str(list_path)))
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
    return ch


def get_band_image_name(image_path, output_dir):
    assert output_dir[-1] == "/", "The path of output dir should end with / {}".format(output_dir)
    image_name = image_path.split("/")[-1]
    return output_dir + image_name.split(VAR_NAME)[0] + ".tif"


def _argparser():
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('--input_dir', type=str, default="/datastore/dum031/dataset1/prepro6/date2",
                        help="path to input directory ")
    parser.add_argument('--output_dir', type=str, default="/datastore/dum031/dataset1/prepro6/date2",
                        help="path to the output directory")
    parser.add_argument("--bands", nargs="+", default=None, help="list of all the bands")
    return parser.parse_args()


def main(input_dir, output_dir, list_band):
    if list_band is None:
        list_band = [b.lower().replace("0", "") for b in
                     LISTE_BANDE[1]]  # liste band of sentinel 2, convert it from B02->b2

    for b in list_band:
        list_image = get_path_tile(b, input_dir)
        output_name = mosaic_image(list_image, output_dir)
        print("The image {} has been created".format(output_name))


if __name__ == '__main__':
    args = _argparser()
    main(args.input_dir, args.output_dir, args.bands)
