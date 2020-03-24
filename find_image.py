import ee
from gee_constant import dict_collection
import argparse
import json

ee.Initialize()


def gjson_2_eegeom(path_geojson):
    """Given the oath to the goejson returns an ee geometry polygon"""
    with open(path_geojson) as f:
        data = json.load(f)
    assert len(data["features"]) == 1, "More than one feature is stored {}".format(data["features"])
    list_coords = data["features"][0]["geometry"]["coordinates"]
    return define_geometry(list_coords)


def define_geometry(list_coordinates):
    """list_coordinates is a list of list. [[x1,y1],[x2,y2]..]
    :returns an ee.Geometry"""
    geometry = ee.Geometry.Polygon(
        list_coordinates, None, False);
    return geometry


def display_search(begin_date, ending_date, zone, collection):
    # print("[INFO] searching images from collection {} \n beginning {} ending {} \n"
    #       "in zone {}".format(dict_collection[collection], begin_date, ending_date, zone)
    #       )
    pass


def get_filter_collection(begin_date, ending_date, zone, sent=1, opt_param={}):
    """:param collection sent1 or sent2 collections
    :param zone : path to the geojson file which contains the zone geometry
    :return an Image collection"""
    display_search(begin_date, ending_date, zone, sent)
    collection = ee.ImageCollection(dict_collection[sent])
    collection = collection.filterDate(begin_date, ending_date).filterBounds(zone)
    if sent == 2:
        return collection
    else:
        return opt_filter(collection, opt_param, sent)


def opt_filter(collection, opt_param, sent):
    """Function used to add new filters
    :param collection:
    :param opt_param:
    :param sent:
    """
    # print("sent {}".format(sent))
    if sent == 1:
        if "sensorMode" in opt_param:
            # print("sensorMode {}".format(opt_param["sensorMode"]))
            collection = collection.filter(ee.Filter.eq('instrumentMode', opt_param["sensorMode"]))
        if "polarisation" in opt_param:
            for polar in opt_param["polarisation"].split(","):
                # print("Filter by polarisation {}".format(polar))
                collection = collection.filter(ee.Filter.listContains('transmitterReceiverPolarisation', polar))
        if "orbitDirection" in opt_param:
            # print("Filter by orbit direction {}".format(opt_param["orbitDirection"].upper()))
            collection = collection.filter(ee.Filter.eq('orbitProperties_pass', opt_param["orbitDirection"].upper()))
    else:  # sentinel2
        # print("Sentinel 2 default mode are MSI and Level 1C !!! To change that change the constant parameters !!")
        # print(opt_param)
        assert "ccp" in opt_param, "Wrong param for sentinel 2 {} should only be the cloud coverage percentage".format(
            opt_param)
        # print("Filter values with less than {} percentage of cloud pixels type {}".format(opt_param["ccp"],type(opt_param["ccp"])))
        collection = collection.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', opt_param["ccp"]))
        # print("Filter values with less than {} percentage of cloud pixels".format(opt_param["ccp"]))

    return collection


def _argparser():
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('--bd', type=str, help="begin date")
    parser.add_argument('--ed', type=str, help="begin date")
    parser.add_argument('--zone', type=str, help="path where the zone coordinates are stored ")
    parser.add_argument('--c', type=int, help="collection")
    return parser.parse_args()


def extract_fp(image):
    """Given an ee.Image extract its fp"""
    pass


def list_image_name(image_collection, sent):
    """    :param sent:
    :param image_collection : an ee image collection
    :returns a list of the name of the image collection"""
    # get the len of the image collection
    n = image_collection.toList(1000).length().getInfo()
    list_name = []
    list_image_collection = image_collection.toList(n)
    for i in range(n):
        if sent == 1:
            name = ee.Image(list_image_collection.get(i)).get("system:id").getInfo()
        else:
            name = ee.Image(list_image_collection.get(i)).get("PRODUCT_ID").getInfo()

        list_name += [name]
    return list_name


def main(begin_date, ending_date, zone, sent):
    collection = get_filter_collection(begin_date, ending_date, zone, sent)
    print(list_image_name(collection, sent))


if __name__ == '__main__':
    args = _argparser()
    main(args.bd, args.ed, args.zone, args.c)
