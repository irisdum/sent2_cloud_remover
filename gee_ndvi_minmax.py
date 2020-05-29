# Function used to find the minimum and maximum of each tiles
import argparse
import ee
from find_image import get_filter_collection, define_geometry
from constant.gee_constant import DICT_EVI_PARAM, GEE_S2_BAND, GEE_DRIVE_FOLDER, EVI_BAND, \
    NDVI_BAND, DICT_TRANSLATE_BAND, NB_VI_CSV, CONVERTOR
from scanning_dataset import extract_tile_id
import pandas as pd

from utils.geojson import create_geojson, load_grid_geojson
from utils.normalize import stat_from_csv

ee.Initialize()


def normalize(image, band, geometry, scale=None):
    """:param image : an ee Image
    :param band a String should correspond exactly to the band as it is written in the GEE S2 images
    :param scale a tuple (min,max)
    :param geometry : an ee.Geometry
    : returns image with a new band which is norm_band"""
    subBand = image.select(band)
    if scale is None:  # TODO : correct and find where the bugs come from
        # We want to compute local max and min within the tile but it does not work ...
        # print("{} Compute our own norm".format(band))
        assert True, "This feature does not work, error Image.constant when exporting to Earth Engine"

        bmin, bmax = one_band_max(subBand, band, geometry)
    else:
        # print("Max and min Given {}".format(scale))
        bmin = ee.Number(scale[0] * CONVERTOR)
        bmax = ee.Number(scale[1] * CONVERTOR)
    # to change

    normalize_band = ee.Image(subBand.select(band).subtract(ee.Image.constant(bmin))).divide(
        ee.Image.constant(bmax).subtract(ee.Image.constant(bmin))).rename("{}_norm".format(band))

    return image.addBands(normalize_band)


def apply_ndvi(image):
    """:param image : an ee.Image
    add a band ndvi"""
    valndvi = image.normalizedDifference(['B8_norm', 'B4_norm']).rename('ndvi')
    return image.addBands(valndvi)


def apply_evi(image, param=None):
    """:param image : an ee.Image
    :param param a dictionnary with the value of the constant used for evi computation"""
    if param is None:
        param = DICT_EVI_PARAM
    evi = image.expression(
        '{} * ((NIR-RED) / (NIR +{} * RED - {} * BLUE + {}))'.format(param["G"], param["C1"], param["C2"], param["L"]),
        {
            'NIR': image.select('B8_norm'),
            'RED': image.select('B4_norm'),
            'BLUE': image.select('B2_norm')
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
    parser.add_argument('--path_csv', default=None, type=str,
                        help="path to the directory where the csv are stored")
    return parser.parse_args()


def get_ndvi_minmax_tile(col, roi, dict_scale=None, liste_band=None, vi="ndvi", export="GEE"):
    """    :param dict_scale: dict of tuple keys are the bands R,G,B,NIR ex {R:(min,max}}
    :param col : An image collection of all the images in the roi
    :param roi : an ee.Geometry, corresponds to a tile footprint
    :param liste_band : list of the band to use in order to get the ndvi should be in sentinel 2 format ie B2,B3,B4
    :param vi : a string which corresponds to the name of the vegetation index we are interested in could be vi, or
    :returns a dictionnary : {vi_min : value_min,vi_max : value_max }"""
    assert vi in ["ndvi",
                  "evi"], "The extraction of min max for this vegetation index {} is undefined,please modify gee_ndvi_minmax.py".format(
        vi)

    if liste_band is None:
        if vi == "ndvi":
            liste_band = NDVI_BAND
        elif vi == "evi":
            liste_band = EVI_BAND
    # first we normalize
    # print("The dict is {}".format(dict_scale))
    for i, b in enumerate(liste_band):
        # print("We use band {}".format(dict_scale[DICT_TRANSLATE_BAND[b]]))
        col = col.map(lambda img: normalize(img, b, roi, scale=dict_scale[DICT_TRANSLATE_BAND[b]]))
    # cast the value
    pixel_val = ee.PixelType('float', ee.Number(0), ee.Number(1))
    # print(type(pixel_val))
    liste_band_norm = ["{}_norm".format(b) for b in liste_band]
    # print(liste_band_norm)
    col = col.select(liste_band_norm).cast(dict(zip(liste_band_norm, [pixel_val for i in range(len(liste_band_norm))])),
                                           liste_band_norm)
    # compute the ndvi
    # test_min,test_max=one_band_max(col.first(),band="{}_norm".format(b),zone=roi)
    # print("test min {} max {}".format(test_min.getInfo(),test_max.getInfo()))
    if vi == "ndvi":
        assert "B8" in liste_band, "The band B8 has not been normalized {}".format(liste_band)
        assert "B4" in liste_band, "The band B4 has not been normalized {}".format(liste_band)
        col = col.map(apply_ndvi)
    if vi == "evi":
        assert "B8" in liste_band, "The band B8 has not been normalized {}".format(liste_band)
        assert "B4" in liste_band, "The band B4 has not been normalized {}".format(liste_band)
        assert "B2" in liste_band, "The band B2 has not been normalized {}".format(liste_band)
        col = col.map(lambda image: apply_evi(image, param=None))
    # print("Band {} created".format(vi))
    # print(type(roi))
    # vi_min,vi_max=one_band_max(col.select(vi).max(),vi,zone=roi)

    return band_min_max(col, roi, lband=[vi], export=export)


def one_band_max(image_band, band, zone):
    """:param image_band an ee.Image
    :param band a string, should be S2 official band name i.e B2,B3,B4,B8 ...
    :zone an ee.Geometry, the footprint
    :returns the min and the max on this image """
    image_band = ee.Image(image_band.select([band]))
    maxReducer = ee.Reducer.minMax()
    minMax = ee.Image(image_band).reduceRegion(maxReducer, zone, 1, image_band.projection())
    return minMax.get("{}_min".format(band)), minMax.get("{}_max".format(band))


def band_min_max(col, zone, lband=None, export="GEE"):
    """:param col : an ee.Collection
    :param zone: an ee.Geometry
    :param lband : a liste of the band on which min and max are going to be computed
    :param export : a string if export=GEE the output is ee.Number otherwise string
    :returns a dict ex : {band1_min: val, band2_min: val ...}
     """
    if lband is None:  #
        lband = GEE_S2_BAND
    dict_band_minmax = {}
    print("TYPE ROI {}".format(type(zone)))
    for band in lband:
        print(band)
        _, band_max = one_band_max(col.select([band]).max(), band,
                                   zone)  # correct by adding putting list as an input of select
        band_min, _ = one_band_max(col.select([band]).min(), band, zone)
        if export == "GEE":
            print("{}_min".format(band))
            dict_band_minmax.update(
                {"{}_min".format(band): band_min, "{}_max".format(band): band_max})
        else:
            print("BAND MIN {} MAX {}".format(band_min.getInfo(), band_max.getInfo()))
            dict_band_minmax.update(
                {"{}_min".format(band): band_min.getInfo(), "{}_max".format(band): band_max.getInfo()})

    return dict_band_minmax


def all_minmax(path_build_dataset, begin_date, ending_date, vi, export="GEE", path_csv=None):
    """ :param path_csv: path where the csv containing the S2 min max value for band is going to be stored, if None a
    the normalization is done using the min and max value from the image
    :param vi: string, indicates the type of vegetation index which is going to be use ex : vi, evi
    :param export: string could be GEE or local
    :param ending_date: string ending date to collect Image for the collection
    :param begin_date: string end date to collect Image for the collection
:   :param path_build_dataset string path to the build dataset (path where all the tiles are stored)
    Depending on export, if export is GEE the Download into google drive of the results for all the tiles of the grid
    otherwise download occurs using pandas to_csv"""
    geojson_path = create_geojson(path_build_dataset)  # path where the geojson of the grid of all the tiles is stored
    l_grid_info = load_grid_geojson(geojson_path)  # list of list with path to the image, and liste of coordo
    df = pd.DataFrame()
    print(l_grid_info[0:10])
    features = []
    # go over all the tiles
    for i, tile in enumerate(l_grid_info):
        path_tile = tile[0]
        coordo_tile = tile[1]
        print("INFO ite {} nber of coordo {},name of tile {}".format(i, len(coordo_tile), path_tile))
        print(i, len(coordo_tile))
        # print(coordo_tile)
        tile_id = extract_tile_id(path_tile)
        zone = define_geometry(coordo_tile)
        if path_csv is not None:  # we are going to be
            dict_tile_stat = stat_from_csv(path_tile, path_csv,
                                           dict_translate_band=None)  # get a dict of the stats of the tile
        else:
            dict_tile_stat = None
        print("TYPE ROI {}".format(type(zone)))
        collection = get_filter_collection(begin_date, ending_date, zone, 2)
        # get_ndvi_minmax_tile(collection, zone)
        print("We have collection")  # TODO combien the two functions and see if it works
        dic_band_min_max = get_ndvi_minmax_tile(collection, zone, vi=vi, dict_scale=dict_tile_stat, export=export)
        dic_band_min_max.update({"name": tile_id})
        if export == "GEE":
            new_feat = ee.Feature(None, dic_band_min_max)
            features += [new_feat]
        else:
            df = df.append(dic_band_min_max, ignore_index=True)
            print(df)
    if export == "GEE":
        nb_csv = NB_VI_CSV  # To avoid broken pipe we divide the export in many csv files
        tot = len(l_grid_info)
        for i in range(0, nb_csv):
            fromList = ee.FeatureCollection(features[i * (tot // nb_csv):(i + 1) * (tot // nb_csv)])
            print("Iter {} on {}, nb_element {}".format(i, nb_csv,
                                                        len(features[i * (tot // nb_csv):(i + 1) * (tot // nb_csv)])))
            task = ee.batch.Export.table.toDrive(collection=fromList, description="export_{}_n{}".format(vi, i),
                                                 folder=GEE_DRIVE_FOLDER,
                                                 fileNamePrefix="export_{}_n{}_d{}_{}".format(vi, i, begin_date,
                                                                                              ending_date),
                                                 fileFormat="CSV")
            task.start()
            print(type(task))
        print("Export of the CSV file in your Drive folder {}".format(GEE_DRIVE_FOLDER))
    else:
        df.head(10)
        df.to_csv(path_build_dataset + "{}_min_mx.csv".format(vi), sep=",")


def get_band_s2_min_max(path_build_dataset, begin_date, ending_date, lband=None, save_name="s2_bands_min_max",
                        export="GEE"):
    """    :param path_build_dataset: path to the build_dataset
    :param lband : is a list of string which indicates the band on which we are going to compute the min and the max
     ex [B2,B3,B4]
    :param export: string could be GEE or local
    :param ending_date: string ending date to collect Image for the collection
    :param begin_date: string end date to collect Image for the collection
:   :param path_build_dataset string path to the build dataset (path where all the tiles are stored)
    Depending on export, if export is GEE the Download into google drive of the results for all the tiles of the grid
    otherwise download occurs using pandas to_csv
    """
    if lband is None:  #
        lband = GEE_S2_BAND
    geojson_path = create_geojson(path_build_dataset)  # path where the geojson of the grid of all the tiles is stored
    l_grid_info = load_grid_geojson(geojson_path)  # list of list with path to the image, and liste of coordo
    for band in lband:
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
            print(band)
            dic_band_min_max = band_min_max(collection, zone, lband=[band], export=export)
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
            task = ee.batch.Export.table.toDrive(collection=fromList, description="{}_export_minmax".format(band),
                                                 folder=GEE_DRIVE_FOLDER + band,
                                                 fileNamePrefix="{}_export_minmax_d{}_{}".format(band, begin_date,
                                                                                                 ending_date),
                                                 fileFormat="CSV")
            task.start()
            print("Export of the CSV file in your Drive folder {}".format(GEE_DRIVE_FOLDER))

        else:
            df.to_csv(path_build_dataset + "{}_{}.csv".format(band, save_name), sep=",")


def main(path_build_dataset, begin_date, ending_date, vi, export, path_csv):
    if vi in ["evi", "ndvi"]:
        all_minmax(path_build_dataset, begin_date, ending_date, vi, export=export,
                   path_csv=path_csv)  # TODO adapt the script so the normalization of the data when computing the vi is the global constant
    else:
        print("We take care of the the S2 min max")
        get_band_s2_min_max(path_build_dataset, begin_date, ending_date, export=export)


if __name__ == '__main__':
    args = _argparser()
    main(args.path_bdata, args.bd, args.ed, args.vi, args.export, args.path_csv)
