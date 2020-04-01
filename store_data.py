# Files with all the functions refering to the file hierarchy, how to deal the storing and searchin
import os
import shutil
import click
from shutil import rmtree
from gee_constant import DIR_T, DIR_SENT, TEMPORARY_DIR, SENT_FORMAT, LISTE_BANDE, DOWNLOAD_PATH
import glob
from datetime import date


def crop_image(image_path, path_shapefile, output_path):
    assert os.path.isfile(path_shapefile), "No path in {}".format(path_shapefile)
    # assert os.path.isdir(output_dir),"No dir in {}".format(output_dir)
    print("gdalwarp -cutline  SHAPE_RESTORE_SHX=YES {} {} {}".format(path_shapefile, image_path, output_path))
    os.system("gdalwarp -cutline   {} {} {}".format(path_shapefile, image_path, output_path))
    return output_path


def list_directory(path_downloads):
    """:returns a list of path to the directory which corresponds to all the unzip folders"""
    list_dir = os.listdir(path_downloads)
    path_image_dir = []
    for dir in list_dir:
        if dir.endswith(".SAFE"):
            path_image_dir += [path_downloads + dir]
    print(path_image_dir)
    return path_image_dir


def list_band(path_image_zip_dir, sent):
    # print(type(sent))
    # print(glob.glob("{}/**/*.{}".format(path_image_zip_dir, SENT_FORMAT[sent - 1]), recursive=True))
    return glob.glob("{}/**/*.{}".format(path_image_zip_dir, SENT_FORMAT[sent - 1]), recursive=True)


def create_hierarchy(output_path):
    """Create an approriate directory directory to download the data"""
    # 1 Check if there is already a directory in output path
    if os.path.isdir(output_path):
        if click.confirm('The directory already exists, it will remove it do you want to continue?', default=True):
            print('Ok remove')
            shutil.rmtree(output_path)
        else:
            return False
    assert output_path[-1] == "/", "wrong output path directory should end with / but is {}".format(output_path)
    print("create {} {}", output_path, output_path + TEMPORARY_DIR)
    os.makedirs(output_path)  # it is the directory which contains all the images for one area
    os.makedirs(output_path + TEMPORARY_DIR)
    mk_sentineldir(output_path + DIR_T[0])
    mk_sentineldir(output_path + DIR_T[1])
    assert os.path.isdir(output_path + TEMPORARY_DIR), "The directory has not been created yet {}".format(
        output_path + TEMPORARY_DIR)


def create_download_dir(download_path):
    if os.path.isdir(download_path):
        if click.confirm('The directory already exists, it will remove it do you want to continue?', default=True):
            print('Ok remove')
            shutil.rmtree(download_path)
        else:
            return False

    os.makedirs(download_path)
    os.makedirs(download_path + DIR_T[0])
    os.makedirs(download_path + DIR_T[1])
    return True


def mk_sentineldir(path_time_dir):
    os.makedirs(path_time_dir)
    os.makedirs(path_time_dir + DIR_SENT[0])
    os.makedirs(path_time_dir + DIR_SENT[1])


def combine_image(lpath, output_path_dir, image_name, sent):
    """Given a list of path to different image bands : combine them to create one final image
    :param sent: 1 or 2
    :returns : str the output paths"""
    print("The bands are going to be combined in the following order from top to bottom {}".format(lpath))
    # Combine all the images and store them in the temporary directory
    if sent == 1:
        assert len(lpath) == 2, "Wrong bands selected should be 4 bands for sentinel 2 if not change de combine_image" \
                                " functions to accept {}".format(lpath)
        os.system("gdal_merge.py -separate -o {} -co PHOTOMETRIC=MINISBLACK {} {}".format(
            output_path_dir + image_name, lpath[0], lpath[1]))
    elif sent == 2:
        assert len(lpath) == 4, "Wrong bands selected should be 4 bands for sentinel 2 if not change de " \
                                "combine_image functions to accept {}".format(lpath)
        os.system("gdal_merge.py -separate -o {}  {} {} {} {}".format(
            output_path_dir + image_name, lpath[0], lpath[1], lpath[2], lpath[3]))
    # -co PHOTOMETRIC=MINISBLACK
    return output_path_dir + image_name


def sent2_convert(list_image_path, output_path_dir):
    """:param list_image_path : list of path to jpg2 images
    :param lband : list of the bands to be selectec to create the final sentinel image"""
    assert type(list_image_path) == type([]), "Wrong input format should be list but is {}".format(list_image_path)
    l_path_tiff = []
    for path_jp2 in list_image_path:
        path_tiff = output_path_dir + jp2name_2_tiffname(path_jp2)
        os.system("gdal_translate {} {}".format(path_jp2, path_tiff))
        l_path_tiff += [path_tiff]

    # save the image in temporary directory
    return l_path_tiff


def jp2name_2_tiffname(path_jp2):
    """Given a jp2 image path return the name of the image with the extension .tiff"""
    l = path_jp2.split("/")
    name = l[-1][:-3] + "tiff"
    return name


def store_image(list_path_band, sent, t, output_path_dir, image_name, path_shapefile):
    """    :param image_name:
    :param path_shapefile:
:param init_image_dir:
    :param output_path_dir:
    :param list_path_band : list path of the bands of the same image
    :param sent : 1 or 2 depending of the satellite used,
    :param t : corresponds to the first date or second date of acquisition"""
    # Select only the requested bands in list_path_band
    list_path_band = sort_sent_band(list_path_band, sent)
    print("DIR ", output_path + TEMPORARY_DIR)
    list_path_band_cropped = [
        crop_image(path_band, path_shapefile, output_path + TEMPORARY_DIR + path_band.split("/")[-1]) for path_band in
        list_path_band]
    if sent == 1:
        final_path = combine_image(list_path_band_cropped, output_path_dir + TEMPORARY_DIR, image_name, sent)
    else:
        list_converted_image_path = sent2_convert(list_path_band_cropped,
                                                  output_path_dir + TEMPORARY_DIR)  # convert all the images into geotiff
        assert len(list_converted_image_path) > 0, "Wrong conversion of {} \n paths : {} \n the output path list is " \
                                                   "empty".format(sent, list_path_band)
        final_path = combine_image(list_converted_image_path, output_path_dir + TEMPORARY_DIR, image_name, sent)

    save_image(final_path, output_path_dir, sent, t)
    # Delete the image from the temporary directory


def save_image(final_path, output_path_dir, sent, t):
    """:param final_path : path to the image
    :param: output_path_dir : path to the main directory
    :param sent : int 1 or 2
    :param t : int 1 or 2"""
    # create a copy of the image in the output_dir_path right folders

    # hierarchy path within the outputpath_dir :
    hierarchy_path = DIR_T[t - 1] + DIR_SENT[sent - 1]
    print("hierarchy path {}".format(hierarchy_path))
    image_name = final_path.split("/")[-1]
    stroring_path = output_path_dir + hierarchy_path + image_name
    os.system("cp {} {}".format(final_path, stroring_path))


def storing_process(output_path_dir, list_dir, t, path_shapefile):
    """    :param t:
    :param list_dir:
    :param output_path_dir : path to output directory
"""

    for dir_image in list_dir:
        # extract sentinel
        sent = path_2_sent(dir_image)
        print("sent : {}".format(sent))
        list_band_path = list_band(dir_image, sent)
        assert len(list_band_path) > 0, "No images band format {} found in {}".format(SENT_FORMAT[sent - 1], dir_image)
        # store the image the images will be converted and band combined to be set in an appropriate directory
        image_name = extract_path_image_name(dir_image)
        store_image(list_band_path, sent, t, output_path_dir, image_name, path_shapefile)


def sort_sent_band(list_sent_band_path, sent):
    """Given the list of images sort them given the order given in the constant file"""
    final_image_path = []
    for band in LISTE_BANDE[sent - 1]:
        for path in list_sent_band_path:
            if band in path:
                final_image_path += [path]
    assert len(final_image_path) > 0, "No image has been selected the bands might not be correct {}".format(
        LISTE_BANDE[sent - 1])
    return final_image_path


def path_2_sent(path_unzipped_dir):
    print(path_unzipped_dir)
    if "S1" in path_unzipped_dir:
        print("sentinel1")
        return 1
    elif "S2" in path_unzipped_dir:
        print("sentinel2")
        return 2
    else:
        return None


def extract_datetime_path(path_dir):
    str_date = path_dir.split("_")[4]
    str_datetime = str_date.split("T")[0]
    str_year = str_datetime[:4]
    str_month = str_datetime[4:6]
    str_day = str_datetime[6:8]
    return date(str_year, str_month, str_day)


def preprocess_all(output_path_dir, unzipped_dwnld_path_dir, path_shapefile):
    # make the hierarchy
    create_hierarchy(output_path_dir)

    # First get the list of all the directory downloads
    list_dir_t1 = list_directory(unzipped_dwnld_path_dir + DIR_T[0])
    list_dir_t2 = list_directory(unzipped_dwnld_path_dir + DIR_T[1])
    storing_process(output_path_dir, list_dir_t1, 1, path_shapefile)
    storing_process(output_path_dir, list_dir_t2, 2, path_shapefile)
    # Delete the temporary directory
    shutil.rmtree(output_path_dir + TEMPORARY_DIR)
    # Delete the downloads directory


def select_dir_t(list_dir, dict_t):
    """Filter list_dir by returning only the directory with their names in dict_t"""
    dir_t = []
    for key_name in dict_t:
        for dir in list_dir:
            if key_name in list_dir:
                dir_t += [dir]
    return dir_t


def extract_path_image_name(path_dir):
    return path_dir.split("/")[-1][:-5] + ".tiff"


def create_summary(output_dir):
    """Create a summary of the data collected name of the image, url and """
    pass


def main(output_path, download_path, path_shapefile):
    preprocess_all(output_path, download_path, path_shapefile)


if __name__ == '__main__':
    download_path = DOWNLOAD_PATH
    output_path = "./test_kangaroo_image/"
    main(output_path, download_path, path_shapefile="./confs/fp_kangaroo.shp")
