# File where all the functions linked with sent 2 footprints are sentinel 1 footprints are defined

import ee


def get_day_date(image):
    """:param image
    :returns an int with the day of the year"""
    return image.date().get("day")


def get_zone_collection(collection):
    """:param collection : an ee.ImageCollection
    :returns a list of all the different tiles for one acquisition date, by default the first acquisition date of the zone"""
    day_one = get_day_date(collection.first())
    small_collection = collection.filter(
        ee.Filter.calendarRange(day_one, day_one.get("day")))  # Only the image taken from the same date
    list_small = small_collection.toList(1000)
    n = list_small.length()
    list_tiles = []
    for i in range(n):
        list_tiles = ee.Image(list_small.get(i)).get("MRGS_TILE")
    return list_tiles

def sub_collection_tiles(collection):
    """:param collection an ee.ImageCollection
    : returns a list which contains sub collections of the collection. The initial collection is divided using the tile_id,
     one tile_id corresponds to a sub collection"""
    list_tiles=get_zone_collection(collection)
    list_subcollection=[]
    for tile_id in list_tiles:
        sub_collection=collection.filter(ee.Filter.eq("MRGS_TILE",tile_id))
        list_subcollection+=[sub_collection]
    return list_subcollection

def extract_fp(image):
    """:returns an ee.Geometry"""
    return  ee.Geometry(image.get("system:footprint")).bounds()

