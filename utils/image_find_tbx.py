import glob
import os
import shutil
import click

def find_path(sent_dir, image_id):
    """:returns a string which is the path of the image with the id image_id
    """
    assert os.path.isdir(sent_dir),"Wrong path to dir, path  {} does not exist ".format(sent_dir)
    #print(sent_dir + "**/*{}".format(image_id))
    assert ".tif" in image_id, "The id {} is not an image ".format(image_id)
    l = glob.glob(sent_dir + "**/*{}".format(image_id), recursive=True)
    assert len(l) > 0, "No image found with id {} at {}".format(image_id, sent_dir)
    assert os.path.isfile(l[0]), "No file found at {}".format(l[0])
    return l[0]



def extract_relative_path(path_tif):
    """Given the path to an tif tile returns its relative path within the Sentineli_tj directory"""
    l = path_tif.split("/")
    return "/".join(l[-3:-1])


def extract_tile_id(path_tif):
    ch=path_tif.split("/")[-1]
    return  ch.split("_")[4]+"_"+ch.split("_")[5]



def get_all_tiles_path(path_sent_dir):
    """Given the path to Sentineli_tj directory returns a list of all the paths to all the tiles of the image"""
    assert os.path.isdir(path_sent_dir), "The dir {} does not exist".format(path_sent_dir)
    print("research :  {}**/*.tif".format(path_sent_dir))
    l = glob.glob("{}**/*.tif".format(path_sent_dir), recursive=True)
    assert len(l) > 0, "No image found in {}".format(path_sent_dir)
    return l


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


def find_image_indir(path_dir, image_format):
    """

    Args:
        path_dir: string, path to a directory
        image_format: string image format should be in "vrt", "tif", "SAFE/","npy"

    Returns:
        a list of string, the output of glob.glob(path_dir*.format). The order of the element is the same as ls -U
    """
    assert image_format in ["vrt", "tif", "SAFE/",
                            "npy"], "Wrong format should be vrt or tif SAFE/ npy but is {}".format(format)
    assert path_dir[-1] == "/", "path should en with / not {}".format(path_dir)
    return glob.glob("{}*.{}".format(path_dir, image_format))


def find_csv(path_dir,band):
    """:returns a string path to the csv [band]*.csv"""
    path_band_csv = glob.glob("{}*{}*.csv".format(path_dir, band))
    assert len(path_band_csv) > 0, "No csv found at {}*{}*.csv".format(path_dir, band)
    return path_band_csv[0]