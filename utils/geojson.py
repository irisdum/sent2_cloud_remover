import glob
import json
import os

from constant.gee_constant import EPSG
from constant.storing_constant import XDIR, DICT_ORGA


def create_geojson(path_build_dataset):
    """:param path_build_dataset : a string path to the build_dataset directory
    This functions convert the shp grid from the tiling into a geojson
    :returns a string : path to the geojson"""
    print(path_build_dataset + XDIR + DICT_ORGA[XDIR][0])
    assert os.path.isdir(path_build_dataset + XDIR + DICT_ORGA[XDIR][0]), "No dir exists at {}".format(
        path_build_dataset + XDIR + DICT_ORGA[XDIR][0])
    l_shp = glob.glob("{}/**/*.shp".format(path_build_dataset + XDIR + DICT_ORGA[XDIR][0]))
    assert len(l_shp) > 0, "No shp files found at {}/**/*.shp".format(path_build_dataset + XDIR + DICT_ORGA[XDIR][0])
    geojson_path = path_build_dataset + XDIR + DICT_ORGA[XDIR][0] + l_shp[0].split("/")[-1].split(".")[0] + ".geojson"
    if os.path.isfile(geojson_path):
        print("The file has already been created")
        return geojson_path
    else:
        print("SAVE geojson at {} ".format(geojson_path))
        os.system("ogr2ogr -f GEOJSON -s_srs {}  -t_srs crs:84 {} {} ".format(EPSG, geojson_path, l_shp[0]))
        assert os.path.isfile(geojson_path), "No file has been created at {} with the command \n {}".format(
            geojson_path, "ogr2ogr -f GEOJSON  -t_srs crs:84 {} {} ".format(geojson_path, l_shp[0]))
    return geojson_path


def load_grid_geojson(path_geojson):
    """Open the path to the geojson, returns a list of list [path_image,liste_coordo]"""
    with open(path_geojson) as f:
        data = json.load(f)
    l_result = []
    assert len(data) > 0, "The geojson file {} is empty {}".format(path_geojson, data)
    # print(data["features"])
    for i in range(len(data["features"])):
        path_image = data["features"][i]['properties']["location"]
        image_coordo = data["features"][i]["geometry"]["coordinates"]
        assert type(path_image) == type("u"), "Wrong path information {}".format(path_image)
        assert type(image_coordo) == type([]), "Wrong coordo information {}".format(image_coordo)
        assert len(image_coordo) > 0, "No coordinates has been found {}".format(image_coordo)
        l_result += [[path_image, image_coordo]]

    print("We have collected {} information on the tiles ".format(len(l_result)))
    return l_result