# File where all the functions linked with sent 2 footprints are sentinel 1 footprints are defined

import ee
import json
from find_image import next_day, eedate_2_string
from gee_constant import orbit_id


def get_day_date(image):
    """:param image
    :returns an int with the day of the year"""
    return image.date()


def get_property_collection(collection, sent):
    """    :param sent:
:param collection : an ee.ImageCollection
    :returns a list of all the different tiles for one acquisition date, by default the first acquisition date of the zone"""

    small_collection=collection
    assert small_collection.toList(100).length().getInfo() > 0, "Wrong small collection "
    # TODO deal with area with multiples fp
    list_small = small_collection.toList(1000)
    n = list_small.length().getInfo()
    print("Collection property {}".format(n))
    list_tiles = []
    for i in range(n):
        #print(ee.Image(list_small.get(i)).propertyNames().getInfo())
        list_tiles += [get_orbit_id(ee.Image(list_small.get(i)),sent)]
    list_tiles = list(set(list_tiles))  # keep distinct the tiles id
    print(list_tiles)
    return list_tiles

def get_orbit_id(image,sent):
    if sent ==2:
        return ee.String(image.get(orbit_id[sent]).getInfo())
    else:
        return image.get(orbit_id[sent]).getInfo()

def sub_collection_tiles(collection, sent=2):
    """:param collection an ee.ImageCollection
    : returns a list which contains sub collections of the collection. The initial collection is divided using the tile_id,
     one tile_id corresponds to a sub collection"""
    list_tiles = get_property_collection(collection, sent)
    list_subcollection = []
    # list_tiles=["56HKG", "55HGB"]
    for tile_id in list_tiles:
        # print(collection.propertyNames().getInfo()) print(collection.first().propertyNames().getInfo()) print(
        # collection.first().get("MGRS_TILE").getInfo()) assert collection.first().get("MGRS_TILE").getInfo() in
        # list_tiles, "{} not in {}".format(collection.first().get("MGRS_TILE").getInfo(), list_tiles) assert type(
        # tile_id)==type("55HGB"),"Wrong tile id format should be {} not {}".format(type(tile_id),type("55HGB"))
        print(tile_id)
        sub_collection = collection.filter(ee.Filter.eq(orbit_id[sent], tile_id))  # filter by the tilesid
        list_subcollection += [sub_collection]
        assert sub_collection.toList(100).length().getInfo() > 0, "Subcollection  {} is empty  ...".format(tile_id)
    assert len(list_subcollection) > 0, "Wrong size of the list subcollection {}".format(len(list_subcollection))
    return list_subcollection

def sort_sentinel1_images(collection):
    list_subcollection=sub_collection_tiles(collection,1) # This list contains a collections of Image from the same date
    # We only want a collection from one date, otherwise it will be repetitive
    # Then we want to download th
    pass



def extract_fp(image):
    """:returns an ee.Geometry"""

    return ee.Geometry(image.get("system:footprint"))
