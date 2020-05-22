# Function used to find the minimum and maximum of each tiles
import argparse
import json
import os

import ee
import glob
from find_image import gjson_2_eegeom, get_filter_collection, list_image_name, define_geometry
from constant.gee_constant import DICT_ORGA, XDIR
from scanning_dataset import extract_tile_id
import pandas as pd

ee.Initialize()

def load_grid_geojson(path_geojson):
    """Open the path to the geojson, returns a list of list [location,liste_coordo]"""
    with open(path_geojson) as f:
        data = json.load(f)
    l_result=[]
    assert len(data)>0, "The geojson file {} is empty {}".format(path_geojson,data)
    #print(data["features"])
    for i in range(len(data["features"])):
        path_image=data["features"][i]['properties']["location"]
        image_coordo=data["features"][i]["geometry"]["coordinates"]
        assert type(path_image)==type("u"),"Wrong path information {}".format(path_image)
        assert type(image_coordo)==type([]),"Wrong coordo information {}".format(image_coordo)
        assert len(image_coordo)>0,"No coordinates has been found {}".format(image_coordo)
        l_result+=[[path_image,image_coordo]]

    print("We have collected {} information on the tiles ".format(len(l_result)))
    return l_result

def normalize(image,band,geometry,scale=None):
    """:param image : an ee Image
    :param band a String should correspond exactly to the band as it is written in the GEE S2 images
    : returns image with a new band which is norm_band"""
    maxReducer = ee.Reducer.minMax()
    subBand = image.select(band)
    if scale is None:
        minMax = ee.Image(subBand).reduceRegion(maxReducer, geometry, 1, subBand.projection())
        bmin = minMax.get("{}min".format(band))
        bmax = minMax.get("{}max".format(band))
    else:
        bmin,bmax=scale
    normalize_band=ee.Image(subBand.select(band).subtract(ee.Image.constant(bmin))).divide(
        ee.Image.constant(bmax).subtract(ee.Image.constant(bmin))).rename("{}_norm".format(band))
    return image.addBands(normalize_band)

def apply_ndvi(image):
    valndvi = image.normalizedDifference(['B8_norm', 'B4_norm']).rename('ndvi')
    return image.addBands(valndvi)


def _argparser():
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('--path_bdata',type=str,help="path to the build_dataset directory")
    parser.add_argument('--path_input_data',type=str,help="path to the input directory dataset which conatins,train,val,test where the csv is going to be stored")
    parser.add_argument('--bd', type=str, help="begin date")
    parser.add_argument('--ed', type=str, help="begin date")
    parser.add_argument('--zone', type=str, help="path where the zone coordinates are stored ")
    parser.add_argument('--c', type=int, help="collection")
    return parser.parse_args()


def get_ndvi_minmax_tile(col,roi,scale=None,liste_band=None,vi="ndvi"):
    """:param col : An image collection of all the images in the roi"""
    if liste_band is None:
        liste_band=["B4","B8"]
    #first we normalize
    for b in liste_band:
        col=col.map(lambda img : normalize(img,b,scale))
    #compute the ndvi
    if vi=="ndvi":
        assert "B8" in liste_band, "The band B8 has not been normalized {}".format(liste_band)
        assert "B4" in liste_band, "The band B4 has not been normalized {}".format(liste_band)
        col=col.map(apply_ndvi)
    vi_max=col.select(vi).max()
    maxReducer = ee.Reducer.minMax()
    minMax = ee.Image(vi_max).reduceRegion(maxReducer,roi, 1, vi_max.projection())
    vi_min = minMax.get("{}min".format(vi))
    vi_max = minMax.get("{}max".format(vi))
    print("We found vi {} min : {} max {}".format(vi,vi_min.getInfo(),vi_max.getInfo()))
    return vi_min.getInfo(),vi_max.getInfo()


def create_geojson(path_build_dataset):
    print(path_build_dataset+XDIR+DICT_ORGA[XDIR][0])
    assert os.path.isdir(path_build_dataset+XDIR+DICT_ORGA[XDIR][0]), "No dir exists at {}".format(
        path_build_dataset+XDIR+DICT_ORGA[XDIR][0])
    l_shp=glob.glob("{}/**/*.shp".format(path_build_dataset+XDIR+DICT_ORGA[XDIR][0]))
    assert len(l_shp)>0,"No shp files found at {}/**/*.shp".format(path_build_dataset+XDIR+DICT_ORGA[XDIR][0])
    geojson_path=path_build_dataset+XDIR+DICT_ORGA[XDIR][0]+l_shp[0].split("/")[-1].split(".")[0]+".geojson"
    if os.path.isfile(geojson_path):
        print("The file has already been created")
        return geojson_path
    else:
        print("SAVE geojson at {} ".format(geojson_path))
        os.system("ogr2ogr -f GEOJSON  -t_srs crs:84 {} {} ".format(geojson_path,l_shp[0]))
        assert os.path.isfile(geojson_path),"No file has been created at {} with the command \n {}".format(geojson_path,"ogr2ogr -f GEOJSON  -t_srs crs:84 {} {} ".format(geojson_path,l_shp[0]))
    return geojson_path


def all_minmax(path_build_dataset, input_dataset,begin_date, ending_date):
    """:param path_build_dataset string path to the build dataset (path where all the tiles are stored)
    :param input_dataset path to the dataset which is used to train, test and val the model
    :param output_name path of the name of the csv files we are going to create for the input_dataset with, for each image, tile_id and ndvi min and ndvi max"""
    geojson_path=create_geojson(path_build_dataset) #path where the geojson of the grid of all the tiles is stored
    l_grid_info=load_grid_geojson(geojson_path) #list of list with path to the image, and liste of coordo
    df=pd.DataFrame(columns=["tile_id","vi_min","vi_max"])
    print(l_grid_info[0:10])
    #go over all the tiles
    for tile in l_grid_info:
        #print(tile)
        path_tile=tile[0]
        coordo_tile=tile[1]
        #print(coordo_tile)
        tile_id=extract_tile_id(path_tile)
        zone=define_geometry(coordo_tile)
        print(type(zone))
        print("We are going to collect the image of the area {}".format(zone.area(0.001)))
        collection=get_filter_collection(begin_date, ending_date, zone, 2)
        #get_ndvi_minmax_tile(collection, zone)
        vi_min,vi_max=get_ndvi_minmax_tile(collection,zone)
        df=df.append(dict(zip(["tile_id","vi_min","vi_max"],[tile_id,vi_min,vi_max])))
    df.head(10)
    return df

def main(path_build_dataset, input_dataset,begin_date, ending_date):
    all_minmax(path_build_dataset, input_dataset,begin_date, ending_date)
    #name = ee.Image(collection.first()).get("PRODUCT_ID")
        #.getInfo()
    #print(ee.String(name).getInfo())
    #print(list_image_name(collection, sent))

if __name__ == '__main__':
    args = _argparser()
    main(args.path_bdata,args.path_input_data,args.bd, args.ed)