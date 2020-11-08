import argparse
import glob
import os

from constant.gee_constant import TEMPORARY_DIR, DIR_T, XDIR, LABEL_DIR, VAR_NAME, EPSG, LISTE_BANDE
from utils.converter import geojson_2_strcoordo_ul_lr
from utils.storing_data import create_tiling_hierarchy


def _argparser():
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('--input_dir', type=str, default="/srv/osirim/idumeur/data/dataset2/prepro3/",
                        help="path to input directory ")
    parser.add_argument('--output_dir', type=str, default="/srv/osirim/idumeur/data/dataset2/prepro3/build_dataset/",
                        help="path to the output directory")
    parser.add_argument("--bands2", nargs="+", default=None, help="list of all the bands of sentinel 2 format B02,B03")

    parser.add_argument("--bands1", nargs="+", default=None, help="list of all the bands of sentinel1 format vv, vh")
    parser.add_argument("--geojson", default="./confs/train_kangaroo_utm2.geojson", help="path to the zone geojson")

    return parser.parse_args()


def main(input_dir, output_dir, list_band2, list_band1, path_geojson):
    """

    Args:
        input_dir: string, path to the directory which contains the preprocessed image
        output_dir: string, path to the output directory
        list_band2:list of string, contains the band selected for Sentinel 2
        list_band1: list of string contains the band selected for Sentinel 1
        path_geojson: string, path to the geojson file, which gives the Bbox information

    Returns:

    """
    create_tiling_hierarchy(output_dir)
    # Work
    # Sentinel 1 at date 1 :
    process_date_sent(list_band1, 1, input_dir, output_dir, XDIR, path_geojson, 0)
    # Sentinel 1 at date 2 :
    # process_date_sent(list_band1, 1, input_dir, output_dir, XDIR, path_geojson, 1)
    # Sentinel 2 at date 1 :
    process_date_sent(list_band2, 2, input_dir, output_dir, XDIR, path_geojson, 0)
    # Sentinel 2 at date 2 :
    # process_date_sent(list_band2, 2, input_dir, output_dir, LABEL_DIR, path_geojson, 1)


def process_date_sent(list_band, sent, input_dir, output_dir, sub_dir, path_geojson, t):
    """

    Args:
        t:
        list_band: list of string, indicates the band which are going to be used
        sent: int, could be 1 or 2, respectively for sentinel 1 data or sentinel 2 data
        input_dir: string path to the directory which contains the preprocesssed image, we consider it is  type **/prepro3/
        output_dir: string The name of the directory for the dataset
        sub_dir: string, directory name for label of x data. (For the training)
        path_geojson: string, path to the geojson file which gives the coordinates of the Bbox.

    Returns:

    """
    input_dir = input_dir + DIR_T[t]
    assert os.path.isdir(input_dir), "No directory name {}".format(input_dir)
    if list_band is None and sent == 2:
        list_band = [b.replace("0", "") for b in
                     LISTE_BANDE[1]]  # liste band of sentinel 2, convert it from B02->b2
    if list_band is None and sent == 1:
        list_band = LISTE_BANDE[0]

    # Find the band directories contained in the input dir
    list_directory = glob.glob("{}*S{}*.data".format(input_dir, sent))
    assert len(list_directory) > 0, "No directory type {} found".format("{}*S{}*.data".format(input_dir, sent))
    print("[INFO] start combining for {} the bands {}".format(list_directory, list_band))
    # Merge for each image the band together.
    l_output_path = []
    for dir in list_directory:
        list_path_band = find_image_band(dir, list_band=list_band)
        assert len(list_path_band) > 0, "No bands {} found in {}".format(list_band, dir)
        merged_band_image = combine_band(list_path_band, output_dir + sub_dir + TEMPORARY_DIR)
        l_output_path += [merged_band_image]
        print ("[INFO] we combined for images bands {} in dir {} and output it as {}".format(list_path_band, dir,
                                                                                             merged_band_image))
    # Mosaic if multiple images found
    if len(l_output_path) > 1:
        l_output_path = [mosaic_image(l_output_path, output_dir + sub_dir + TEMPORARY_DIR)]

    print("[INFO] we end up with {} as the merged/mosaic_image".format(l_output_path))
    assert len(
        l_output_path) == 1, "Issue with the code should not end up with more than one image in the list {}".format(
        l_output_path)
    # Crop the image
    crop_image_name = crop_image(l_output_path[0], path_geojson,
                                 output_dir + "merged_crop_sent{}_t{}.vrt".format(sent, t))
    return crop_image_name


def mosaic_image(list_path, output_dir):
    """

    Args:
        list_path: list of string, path to the image to mosaic together
        output_dir: string, path to the output directory

    Returns:
        path where the mosaic image has been created

    """
    output_name = get_band_image_name(list_path[0], output_dir)

    os.system("gdalbuildvrt  {} {}".format(output_name, list_2_str(list_path)))
    assert os.path.isfile(output_name), "The file has not been created at {}".format(output_name)
    return output_name


def find_image_band(input_directory, list_band, format="img"):
    """

    Args:
        input_directory: string, path to the directory
        list_band: list of the band to collect

    Returns:
        a list of the path each image band
    """
    l_final = []
    for b in list_band:
        lpath2band = glob.glob("{}*{}*{}".format(input_directory, b, format))
        assert len(lpath2band) == 1, "Error None or Multiple image have been found {}, should be only one command {} ".format(
            lpath2band,"{}*{}*{}".format(input_directory, b, format))
        l_final += lpath2band
    return l_final


def combine_band(list_path_band, output_dir):
    """

    Args:
        list_path_band: list of string, path to the image of the ban
        output_dir: string, path to the output directory

    Returns:
        string, path to the output vrt image, which correspond to the merge of all the input image. Each input image
        is considered as a band in the output image
    """
    output_name = create_name_band(list_path_band[0], output_dir)  # The name of the ouptut image
    print("BAND COMBINATION  : gdalbuildvrt -separate {} {}".format(output_name, list_2_str(list_path_band)))
    os.system("gdalbuildvrt -separate {} {}".format(output_name, list_2_str(list_path_band)))  # Sent2 RGB NIR
    print("AFTER COMBINE ")
    # os.system("gdalinfo {}".format(output_name))
    return output_name


def create_name_band(band_path, output_dir):
    """

    Args:
        band_path: string, path to the image
        output_dir: string, path to the output directory

    Returns:

    """
    return output_dir + band_path.split("/")[-2][:-4]


def get_band_image_name(image_path, output_dir):
    """

    Args:
        image_path: string, path to one image
        output_dir: string, path to an output directory

    Returns: a string, which corresponds to the name (ending with vrt)

    """
    assert output_dir[-1] == "/", "The path of output dir should end with / {}".format(output_dir)
    image_name = image_path.split("/")[-1]
    return output_dir + image_name.split(VAR_NAME)[0] + "merged.vrt"


def crop_image(image_path, path_geojson, output_path):
    """

    Args:
        image_path: string, path to the image which should be croped
        path_geojson: string, path to the geojson which gives the Bbox on which to crop
        output_path: string, path of the output image, recommendation should end up with .vrt

    Returns: a string, path of the cropped image

    """
    assert os.path.isfile(path_geojson), "No path in {}".format(path_geojson)
    # assert os.path.isdir(output_dir),"No dir in {}".format(output_dir)
    str_bbox = geojson_2_strcoordo_ul_lr(path_geojson)

    os.system(
        "gdal_translate {} {} -projwin  {}  -strict ".format(image_path, output_path, str_bbox))
    return output_path


if __name__ == '__main__':
    args = _argparser()
    main(args.input_dir, args.output_dir, args.bands2, args.bands1, args.geojson)


def list_2_str(list):
    """

    Args:
        list: list of string

    Returns: a string, where all the element of the list is a

    """
    ch = ""
    for p in list:
        ch += "{} ".format(p)
    print(ch)
    return ch