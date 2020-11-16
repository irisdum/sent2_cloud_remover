import argparse

import geojson
import numpy as np
from shapely.geometry import shape
import json

from constant.gee_constant import SCALE_S1, CONVERTOR


def geojson_2_WKT(path_geojson, path_wkt):
    """

    Args:
        path_geojson: string, path to the geojson
        path_wkt: strin,g path to the wkt file which is going to be created

    Returns:

    """
    with open(path_geojson) as f:
        data = json.load(f)
    f = open(path_wkt, "w")
    for polygon in data["features"]:
        str_poly = json.dumps(polygon["geometry"])
        # print(str_poly)
        # print(type(str_poly))
        g = geojson.loads(str_poly)
        # shp=shape(g)
        wkt_str = shape(g).wkt
        f.write(wkt_str + "\n")
        print(wkt_str)
    f.close()




def geojson_2_strcoordo_ul_lr(path_geojson):
    """

    Args:
        path_geojson: string, path to the geojson

    Returns:
        a string, which corresponds to the coordinates of the Bbox of the geometry described in the geosjon
    """
    with open(path_geojson) as f:
        data = json.load(f)
    polygon = data["features"][0]
    print(type(polygon))
    print(polygon)
    str_poly = json.dumps(polygon["geometry"])
    g = geojson.loads(str_poly)
    sh = shape(g)
    coordo = sh.bounds
    print("Minx {}, miny {}, maxx {}, maxy{}".format(coordo[0], coordo[1], coordo[2], coordo[3]))
    return "{} {} {} {}".format(coordo[0], coordo[3], coordo[2], coordo[1])


def add_batch_str_coorodo(coordo_str, add_val): #TODO check its utility
    """
    DO NOT USED ONLY FOR LANDCLASSIF
    Args:
        coordo_str: string
        add_val:

    Returns:
        """
    assert len(add_val) == 4, "Wrong in put param add_val should be len 4 not {}".format(add_val)
    l_coordo = coordo_str.split(" ")
    l_increase = []
    for i, coordo in enumerate(l_coordo):
        l_increase += [str(float(coordo) + float(add_val[i]))]
    print("We have addee {} to coordo str {} \n The new bbox coordo are {}".format(coordo_str, add_val,
                                                                                   " ".join(l_increase)))
    return " ".join(l_increase)


def convert_array(raster_array, scale_s1=SCALE_S1, mode=None):
    """
    NOT RECOMMENDED TO BE USED
    Args:
        raster_array: a numpy array, corresponds to a raster
        scale_s1: float, the values of the numpy array are divided by its value
        mode: None or String, if set to CLOUDS_MASK, deal with the conversion of the array
        otherwise divide the array by its max value
    Returns:


    """
    if raster_array.dtype == np.uint16:  # sentinel 2 data needs to be converted and rescale
        return uin16_2_float32(raster_array)
    elif raster_array.dtype == np.float32:
        return np.divide(raster_array, scale_s1).astype(np.float32)
    elif mode == "CLOUD_MASK":
        np.where(raster_array == 1, 0, raster_array)  # clear land pixel
        np.where(raster_array == 4, 0, raster_array)  # snow (cf http://www.pythonfmask.org/en/latest/fmask_fmask.html)
        np.where(raster_array == 5, 0, raster_array)  # water
        return np.divide(raster_array, 5).astype(np.float32)
    else:
        return np.divide(raster_array, np.max(raster_array))


def uin16_2_float32(raster_array, max_scale=CONVERTOR):
    """

    Args:
        raster_array: a numpy array
        max_scale: int or float

    Returns:
        a float32 array, where all the values of the input raster array have been divided my th max scale number

    """
    scaled_array = np.divide(raster_array, max_scale)
    return scaled_array.astype(np.float32)
