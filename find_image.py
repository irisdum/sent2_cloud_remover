import ee
from constant.gee_constant import DICT_COLLECTION
import argparse
from datetime import date, timedelta
import json
from utils.cloud_filters import filter_clouds

ee.Initialize()


# l

def eedate_2_string(date):
    """

    Args:
        date:

    Returns:

    """
    str_day = convert_int(str(date.get("day").format().getInfo()))
    str_month = convert_int(str(date.get("month").format().getInfo()))
    str_year = convert_int(str(date.get("year").format().getInfo()))
    return "{}-{}-{}".format(str_year, str_month, str_day)


def convert_int(str_value):
    if type(str_value) == type(1):
        str_value = str(str_value)

    if len(str_value) == 1:
        return "0" + str_value
    else:
        return str_value


def string_2_datetime(str_date):
    """

    Args:
        str_date: string which describe the date year-month-day

    Returns:
        an ee.Date

    """
    list_date = str_date.split("-")
    new_date = date(int(list_date[0]), int(list_date[1]), int(list_date[2]))
    return new_date


def next_string_date(str_date, i):
    """

    Args:
        str_date: string, date year-month-day
        i: int

    Returns:
        a string, which is is the input date +i

    """
    old_date = string_2_datetime(str_date)
    old_date = old_date + timedelta(days=i)
    return datetime_2_string(old_date)


def datetime_2_string(ex_date):
    """

    Args:
        ex_date: ee.Date

    Returns:
        string, the ee.Date converted into "year-month-day" format
    """
    return "-".join([convert_int(ex_date.year), convert_int(ex_date.month), convert_int(ex_date.day)])


def next_day(str_date, add=1):
    str_day = str_date.split("-")[-1]
    str_next_day = str(int(str_day) + add)
    str_next_date = "-".join(str_date.split("-")[:-1] + [str_next_day])
    # print(str_next_day)
    return str_next_date


def gjson_2_eegeom(path_geojson):
    """

    Args:
        path_geojson: path to a Geojson Polygon File (NOT MULTIPOLUGON!!)

    Returns:
        the ee.Geometry which is decribed in the geojson
    """
    with open(path_geojson) as f:
        data = json.load(f)
    assert len(data["features"]) == 1, "More than one feature is stored {}".format(data["features"])
    list_coords = data["features"][0]["geometry"]["coordinates"]
    print(list_coords)
    return define_geometry(list_coords)


def define_geometry(list_coordinates):
    """

    Args:
        list_coordinates: list of coordinates ex : [[[x1,y1],...[xn,yn]]]

    Returns:
        the ee.Geometry.Polygon defined by the list of coordinates
    """

    geometry = ee.Geometry.Polygon(
        list_coordinates, None, False)
    return geometry


def get_filter_collection(begin_date, ending_date, zone, sent=1, opt_param=None, name_s2=None):
    """

    Args:
        begin_date: ee.Date
        ending_date: ee.Date
        zone: ee.Geometry, the image of the collection searched should cover a part of the zone
        sent: int, could be 1 or 2, respectively indicating Sentinel 1 Collection or Sentinel 2
        opt_param: dictionnary
        name_s2: string or None

    Returns:
    an ee.ImageCollection
    """

    # print("begin {} ending {}".format(begin_date,ending_date))
    if opt_param is None:
        opt_param = {}
    if type(begin_date) != type("u"):
        print("begin {} ending {}".format(begin_date.format().getInfo(), ending_date.format().getInfo()))

    collection = ee.ImageCollection(DICT_COLLECTION[sent])
    collection = collection.filterDate(begin_date, ending_date).filterBounds(zone)
    # print("Collection sent {} filter len {}".format(sent, collection.toList(100).length().getInfo()))
    print(type(collection))
    if sent == 2:
        if name_s2 is not None:
            return collection.filter(ee.Filter.eq("PRODUCT_ID", name_s2))
        else:
            return filter_clouds(collection, zone)
    else:
        return opt_filter(collection, opt_param, sent)


def opt_filter(collection, opt_param, sent):
    """

    Args:
        collection: an ee.Image Collection
        opt_param: a dictionnary, contains optional filters to apply on S1 image
        sent: int, could be one or 2

    Returns:
        collection : an ee.ImageCollection, corresponds to the input ImageCollection on which we have applied different filters
    """
    # print("sent {}".format(sent))
    if collection.toList(100).length().getInfo() == 0:
        return collection
    else:
        if sent == 1:
            if "sensorMode" in opt_param:
                # print("sensorMode {}".format(opt_param["sensorMode"]))
                collection = collection.filter(ee.Filter.eq('instrumentMode', opt_param["sensorMode"]))
            if "polarisation" in opt_param:
                for polar in opt_param["polarisation"].split(","):
                    # print("Filter by polarisation {}".format(polar))
                    collection = collection.filter(ee.Filter.listContains('transmitterReceiverPolarisation', polar))
            if True:
                # print("Filter by orbit direction {}".format(opt_param["orbitDirection"].upper()))
                collection = collection.filter(
                    ee.Filter.eq('orbitProperties_pass', "DESCENDING"))  # ee.Filter.eq('orbitProperties_pass', "")) #

        else:
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


def list_image_name(image_collection, sent):
    """

    Args:
        image_collection: ee.ImageCollection
        sent: int could be 1 or 2

    Returns:list of string, returns a list of the Image_id of the Images contained in the input ImageCollection

    """

    # get the len of the image collection
    print(type(image_collection))
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


def main(begin_date, ending_date, path_zone, sent):
    zone = gjson_2_eegeom(path_zone)
    collection = get_filter_collection(begin_date, ending_date, zone, sent)
    print("get the collection")
    # name = ee.Image(collection.first()).get("PRODUCT_ID")
    # .getInfo()
    # print(ee.String(name).getInfo())
    print(list_image_name(collection, sent))


if __name__ == '__main__':
    args = _argparser()
    main(args.bd, args.ed, args.zone, args.c)
