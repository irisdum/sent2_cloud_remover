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
    print("[INFO] searching images from collection {} \n beginning {} ending {} \n"
          "in zone {}".format(dict_collection[collection],begin_date,ending_date,zone)
          )


def get_image(begin_date, ending_date, zone, collection="sent1", opt_param={}):
    """:param collection sent1 or sent2 collections
    :param zone a list of list
    :return an Image collection"""
    display_search(begin_date, ending_date, zone, collection)
    collection = ee.ImageCollection(dict_collection[collection])
    collection = collection.filterDate(begin_date, ending_date).filterBounds(gjson_2_eegeom(zone))
    if len(opt_param) > 0:
        for key_param in opt_param:
            collection = opt_filter(collection, key_param)
    return collection


def opt_filter(collection, param):
    """Function used to add new filters"""
    print("No defined yet")
    return collection


def _argparser():
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('--bd', type=str, help="begin date")
    parser.add_argument('--ed', type=str, help="begin date")
    parser.add_argument('--zone', type=str, help="path where the zone coordinates are stored ")
    parser.add_argument('--c', type=str, help="collection")
    return parser.parse_args()

def list_image_name(image_collection):
    """:param image_collection : an ee image collection
    :returns a list of the name of the image collection"""
    #get the len of the image collection
    n=image_collection.toList(1000).length().getInfo()
    list_name=[]
    list_image_collection=image_collection.toList(n)
    for i in range(n):
        name=ee.Image(list_image_collection.get(i)).get("system:id").getInfo()
        list_name+=[name]
    return list_name

def main(begin_date, ending_date, zone, collection):
    collection = get_image(begin_date, ending_date, zone, collection)
    #print(collection)
    dict_collection=collection.toDictionary()
    #print(type(dict_collection))
    #print(dict_collection)
    #print(collection.aggregate_array("id"))
    print(list_image_name(collection))

if __name__ == '__main__':
    args = _argparser()
    main(args.bd, args.ed, args.zone, args.c)
