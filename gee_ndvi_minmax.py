# Function used to find the minimum and maximum of each tiles
import argparse
import json
import os
import ee
import glob
from find_image import gjson_2_eegeom, get_filter_collection, list_image_name, define_geometry
from constant.gee_constant import DICT_ORGA, XDIR, EPSG, DICT_EVI_PARAM, GEE_S2_BAND, GEE_DRIVE_FOLDER
from scanning_dataset import extract_tile_id
import pandas as pd

ee.Initialize()


def load_grid_geojson(path_geojson):
    """Open the path to the geojson, returns a list of list [location,liste_coordo]"""
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


def normalize(image, band, geometry, scale=None):
    """:param image : an ee Image
    :param band a String should correspond exactly to the band as it is written in the GEE S2 images
    : returns image with a new band which is norm_band"""
    maxReducer = ee.Reducer.minMax()
    subBand = image.select(band)
    if scale is None:
        print("{} Compute our own norm".format(band))
        minMax = ee.Image(subBand).reduceRegion(maxReducer, geometry, 1, subBand.projection())
        bmin = minMax.get("{}_min".format(band))
        bmax = minMax.get("{}_max".format(band))
        print(type(bmin),type(bmax))
    else:
        bmin, bmax = scale
    print("bmin {} bmax {}".format(bmin.getInfo(), bmax.getInfo()))
    normalize_band = ee.Image(subBand.select(band).subtract(ee.Image.constant(bmin))).divide(
        ee.Image.constant(bmax).subtract(ee.Image.constant(bmin))).rename("{}_norm".format(band))
    return image.addBands(normalize_band)


def apply_ndvi(image):
    valndvi = image.normalizedDifference(['B8_norm', 'B4_norm']).rename('ndvi')
    return image.addBands(valndvi)


def apply_evi(image, param=None):
    if param is None:
        param = DICT_EVI_PARAM
    evi = image.expression(
        '{} * ((NIR-RED) / (NIR +{} * RED - {} * BLUE + {}))'.format(param["G"], param["C1"], param["C2"], param["L"]),
        {
            'NIR': image.select('B8'),
            'RED': image.select('B4'),
            'BLUE': image.select('B2')
        })
    return image.addBands(evi.rename("evi"))


def _argparser():
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('--path_bdata', type=str, help="path to the build_dataset directory")
    parser.add_argument('--path_input_data', type=str,
                        help="path to the input directory dataset which conatins,train,val,test where the csv is going to be stored")
    parser.add_argument('--bd', type=str, help="begin date")
    parser.add_argument('--ed', type=str, help="begin date")
    parser.add_argument('--zone', type=str, help="path where the zone coordinates are stored ")
    parser.add_argument('--c', type=int, help="collection")
    parser.add_argument('--vi', default="ndvi", type=str, help="vegetation index could be  ndvi")
    parser.add_argument('--export', default="GEE", type=str,
                        help="Type of export could be GEE (google earth engine dirve) or loac "
                             "a csv via pandas lib")
    return parser.parse_args()


def get_ndvi_minmax_tile(col, roi, scale=None, liste_band=None, vi="ndvi"):
    """:param col : An image collection of all the images in the roi"""
    if liste_band is None:
        liste_band = ["B4", "B8"]
    # first we normalize
    for b in liste_band:
        col = col.map(lambda img: normalize(img, b, scale))
    # compute the ndvi
    if vi == "ndvi":
        assert "B8" in liste_band, "The band B8 has not been normalized {}".format(liste_band)
        assert "B4" in liste_band, "The band B4 has not been normalized {}".format(liste_band)
        col = col.map(apply_ndvi)
    vi_max = col.select(vi).max()
    maxReducer2 = ee.Reducer.minMax()
    minMax = ee.Image(vi_max).reduceRegion(maxReducer2, roi, 1, vi_max.projection())
    vi_min = minMax.get("{}_min".format(vi))
    vi_max = minMax.get("{}_max".format(vi))
    # print("We found vi {} min : {} max {}".format(vi,vi_min.getInfo(),vi_max.getInfo()))
    return vi_min, vi_max


def one_band_max(image_band, band, zone):
    maxReducer = ee.Reducer.minMax()
    minMax = ee.Image(image_band).reduceRegion(maxReducer, zone, 1, image_band.projection())
    return minMax.get("{}_min".format(band)), minMax.get("{}_max".format(band))


def create_geojson(path_build_dataset):
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


def band_min_max(col, zone, lband=None, export="GEE"):
    "The aim of this function is to save a csv with th min max value of Sentinel 2 and use them to normalize the data"
    if lband is None:  #
        lband = GEE_S2_BAND
    dict_band_minmax = {}
    print("TYPE ROI {}".format(type(zone)))
    for band in lband:
        print(band)
        _, band_max = one_band_max(col.select(band).max(), band, zone)
        band_min, _ = one_band_max(col.select(band).min(), band, zone)
        if export == "GEE":
            dict_band_minmax.update(
                {"{}_min".format(band): band_min, "{}_max".format(band): band_max})
        else:
            dict_band_minmax.update(
                {"{}_min".format(band): band_min.getInfo(), "{}_max".format(band): band_max.getInfo()})

    return dict_band_minmax


def all_minmax(path_build_dataset, input_dataset, begin_date, ending_date, vi, export="GEE"):
    """:param path_build_dataset string path to the build dataset (path where all the tiles are stored)
    :param input_dataset path to the dataset which is used to train, test and val the model
    :param output_name path of the name of the csv files we are going to create for the input_dataset with, for each image, tile_id and ndvi min and ndvi max"""
    geojson_path = create_geojson(path_build_dataset)  # path where the geojson of the grid of all the tiles is stored
    l_grid_info = load_grid_geojson(geojson_path)  # list of list with path to the image, and liste of coordo
    df = pd.DataFrame(columns=["tile_id", "vi_min", "vi_max"])
    print(l_grid_info[0:10])
    features = []
    # go over all the tiles
    for i, tile in enumerate(l_grid_info):

        path_tile = tile[0]
        coordo_tile = tile[1]
        print("INFO ite {} nber of coordo {} ,name of tile {}".format(i, len(coordo_tile), path_tile))
        print(i, len(coordo_tile))
        # print(coordo_tile)
        tile_id = extract_tile_id(path_tile)
        zone = define_geometry(coordo_tile)
        print("TYPE ROI {}".format(type(zone)))
        collection = get_filter_collection(begin_date, ending_date, zone, 2)
        # get_ndvi_minmax_tile(collection, zone)
        print("We have collection")  # TODO combien the two functions and see if it works
        vi_min, vi_max = get_ndvi_minmax_tile(collection, zone)
        if export == "GEE":
            new_feat = ee.Feature(None, {"name": tile_id, "vi_min": vi_min, "vi_max": vi_max})
            features += [new_feat]
        else:
            print("We are going to collect the image of the area {}".format(zone.area(0.001).getInfo()))
            vi_min_val = vi_min.getInfo()
            print("MIN {}".format(vi_min_val))
            vi_max_val = vi_max.getInfo()
            print("MAX {}".format(vi_max))
            df = df.append(dict(zip(["tile_id", "vi_min", "vi_max"], [tile_id, vi_min_val, vi_max_val])),
                           ignore_index=True)
            print(df)
    if export == "GEE":
        fromList = ee.FeatureCollection(features)
        task = ee.batch.Export.table.toDrive(collection=fromList, description="export_{}".format(vi),
                                             folder=GEE_DRIVE_FOLDER,
                                             fileNamePrefix="{}-{}".format(begin_date, ending_date), fileFormat="CSV")
        print(type(task))
        print("Export of the CSV file in your Drive folder {}".format(GEE_DRIVE_FOLDER))
        task.start()
    else:
        df.head(10)
        df.to_csv(path_build_dataset + "{}_min_mx.csv".format(vi), sep=",")


def get_minmax_fromcsv(path_im, path_csv):
    """:path_im : str path to the npy image
    :path_csv : str path to the csv which contains the minmax
    :returns a dictionnary for each band giving the min_max"""
    assert os.path.isfile(path_csv), "No file found at {}".format(path_csv)
    df = pd.read_csv(path_csv, header=True)
    df.head(3)
    tile_id = extract_tile_id(path_im).replace("npy", "tif")
    subf_df = df[df["tile_id"] == tile_id]
    assert subf_df.shape[0] == 1, "Wrong number of image found {}".format(subf_df)
    dict_res = subf_df.iloc[0].to_dict()
    return dict_res


def get_band_s2_min_max(path_build_dataset, begin_date, ending_date, lband=None, save_name="s2_bands_min_max",
                        export="GEE"):
    """:param export a string if export in GEE save the file into a csv Drive if local export via pandas into the local computer
    """
    geojson_path = create_geojson(path_build_dataset)  # path where the geojson of the grid of all the tiles is stored
    l_grid_info = load_grid_geojson(geojson_path)  # list of list with path to the image, and liste of coordo
    df = pd.DataFrame()
    features = []
    for i, tile in enumerate(l_grid_info):
        path_tile = tile[0]
        coordo_tile = tile[1]
        print("ite {} image {}".format(i, path_tile))
        # print(coordo_tile)
        tile_id = extract_tile_id(path_tile)
        zone = define_geometry(coordo_tile)
        collection = get_filter_collection(begin_date, ending_date, zone, 2)
        dic_band_min_max = band_min_max(collection, zone, lband=lband, export=export)
        dic_band_min_max.update({"tile_id": tile_id})
        print(dic_band_min_max.keys())
        if export == "GEE":
            new_feat = ee.Feature(None, dic_band_min_max)
            features += [new_feat]
        else:
            df = df.append(dic_band_min_max, ignore_index=True)
            print(df)
    if export == "GEE":
        fromList = ee.FeatureCollection(features)
        task = ee.batch.Export.table.toDrive(collection=fromList, description="export_s2", folder=GEE_DRIVE_FOLDER,
                                             fileNamePrefix="{}-{}".format(begin_date, ending_date), fileFormat="CSV")
        print("Export of the CSV file in your Drive folder {}".format(GEE_DRIVE_FOLDER))
        print(task.status())
        # print(task.task_type)
        print(task.list())
        task.start()
        print(task.status())
    else:
        df.to_csv(path_build_dataset + "{}.csv".format(save_name), sep=",")


def main(path_build_dataset, input_dataset, begin_date, ending_date, vi, export):
    if vi in ["evi", "ndvi"]:
        all_minmax(path_build_dataset, input_dataset, begin_date, ending_date, vi,
                   export=export)  # TODO adapt the script so the normalization of the data when computing the vi is the global constant
    else:
        print("We take care of the the S2 min max")
        get_band_s2_min_max(path_build_dataset, begin_date, ending_date, export=export)
    # name = ee.Image(collection.first()).get("PRODUCT_ID")
    # .getInfo()
    # print(ee.String(name).getInfo())
    # print(list_image_name(collection, sent))


if __name__ == '__main__':
    args = _argparser()
    main(args.path_bdata, args.path_input_data, args.bd, args.ed, args.vi, args.export)
