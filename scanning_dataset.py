# Python file to scan the dataset and remove the non regular tile
import glob
import argparse

def is_no_data(raster):
    """Given a raster check if there are no data:
    :returns bool"""
    pass

def is_s2_cloud(s2_raster):
    """Given a sentinel 2 raster check if the cloud mask band contains no_data
    :returns bool """
    pass

def extract_relative_path(path_tif):
    """Given the path to an tif tile returns its relative path within the Sentineli_tj directory"""
    l=path_tif.split("/")
    return "/".join(l[-3:-1])

def get_all_tiles_path(path_sent_dir):
    """Given the path to Sentineli_tj directory returns a list of all the paths to all the tiles of the image"""
    l= glob.glob("{}**/*.tif".format(path_sent_dir),recursive=True)
    assert len(l)>0, "No image found in {}".format(path_sent_dir)
    return l


def _argparser():
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('--input_dir', type=str, default="/datastore/dum031/data/dataset1/prepro7/",
                        help="path to input directory ")
    parser.add_argument('--output_dir', type=str, default="/datastore/dum031/data/dataset1/prepro7/build_dataset/",
                        help="path to the output directory")
    parser.add_argument("--bands2", nargs="+", default=None, help="list of all the bands of sentinel 2 format B02,B03")

    parser.add_argument("--bands1", nargs="+", default=None, help="list of all the bands of sentinel1 format vv, vh")
    parser.add_argument("--geojson", default="./confs/train_kangaroo_utm2.geojson", help="path to the zone geojson")
    parser.add_argument("--shp", default="./confs/train_kangaroo.shp", help="path to the esri shapefile")
    return parser.parse_args()


if __name__ == '__main__':
    args = _argparser()