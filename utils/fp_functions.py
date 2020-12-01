# File where all the functions linked with sent 2 footprints are sentinel 1 footprints are defined

import ee
from typing import Tuple

from constant.gee_constant import ORBIT_ID, FACTEUR_AREA


def is_contained(zone, image_fp):
    """:param zone : ee.geometry check if is fully contained in image_fp
    :param image_fp : ee.geometry
    :returns bool True if zone is contained in image_fp"""
    print("Test contains")
    answer = zone.containedIn(image_fp, 0.001)  # check if one geometry is contained in another
    print("Result {} area zone {} area fp {}".format(answer.getInfo(), zone.area(0.001).getInfo(),
                                                     image_fp.area(0.001).getInfo()))
    assert type(answer.getInfo()) == type(True), "Wrong type de answer {}".format(type(answer.getInfo()))
    if answer.getInfo():  # one geometry is contained in another
        print("contains")
        # check wether the geometry contained in zone
        if zone.area(1).getInfo() <= image_fp.area(0.001).getInfo():  # the geometry contained is zone
            return True
        else:  # the geometry contained is image_fp
            return False
    else:
        return False


def get_day_date(image):
    """:param image: ee.Image
    :returns an int with the day of the year"""
    return image.date()


def get_property_collection(collection, sent, zone):
    """    :param sent:
    :param collection : an ee.ImageCollection
    :returns a list of all the different tiles for one acquisition date
    and if there are two tiles that entirely recovers the zone : returns only one"""
    small_collection = collection
    assert small_collection.toList(100).length().getInfo() > 0, "Wrong small collection "
    # TODO deal with area with multiples fp
    list_small = small_collection.toList(1000)
    n = list_small.length().getInfo()
    print("Collection property {}".format(n))
    list_tiles = []
    # list_tile_contain=[] #list of the tiles that contains entirely the area
    for i in range(n):
        # print(ee.Image(list_small.get(i)).propertyNames().getInfo())
        list_tiles += [get_orbit_id(ee.Image(list_small.get(i)), sent)]
        # fp_i=extract_fp(ee.Image(list_small.get(i)))
        # if is_contained(zone, fp_i): #the zone is completly contained in the image fp
    #     if False:
    #         print("{} contains entirely the zone selected".format(get_orbit_id(ee.Image(list_small.get(i)), sent)))
    #         list_tile_contain+=[get_orbit_id(ee.Image(list_small.get(i)), sent)]
    # if len(list_tile_contain)>0:
    #     return [list_tile_contain[0]] #only need one tile
    if True:
        # print("No tiles contains entirely the area")
        list_tiles = list(set(list_tiles))  # keep distinct the tiles id
        print(list_tiles)
        return list_tiles


def get_orbit_id(image, sent):
    if sent == 2:
        return ee.String(image.get(ORBIT_ID[sent]).getInfo())
    else:
        return image.get(ORBIT_ID[sent]).getInfo()


def sub_collection_tiles(collection, zone, sent=2):
    """:param collection an ee.ImageCollection
    : returns a list which contains sub collections of the collection. The initial collection is divided using the tile_id,
     one tile_id corresponds to a sub collection"""
    assert type(sent) == type(1), "Wrong param sent format is {}".format(type(sent))
    # print(type(zone))
    list_tiles = get_property_collection(collection, sent, zone)
    list_subcollection = []
    # list_tiles=["56HKG", "55HGB"]
    for tile_id in list_tiles:
        # print(collection.propertyNames().getInfo()) print(collection.first().propertyNames().getInfo()) print(
        # collection.first().get("MGRS_TILE").getInfo()) assert collection.first().get("MGRS_TILE").getInfo() in
        # list_tiles, "{} not in {}".format(collection.first().get("MGRS_TILE").getInfo(), list_tiles) assert type(
        # tile_id)==type("55HGB"),"Wrong tile id format should be {} not {}".format(type(tile_id),type("55HGB"))
        # print(tile_id.getInfo())
        # print(type(sent))
        # print("SENTfor tile {}".format(tile_id.getInfo()))
        # assert collection.filter(ee.Filter.eq(ORBIT_ID[sent], tile_id)).toList(10).length().getInfo()>0,"Subcollection tile {}filter does not work list empty ".format(tile_id.getInfo())
        image_fp = extract_fp(collection.filter(ee.Filter.eq(ORBIT_ID[sent], tile_id)).first(), sent)
        if is_contained(zone, image_fp):
            print("SENT{} tile contains entirely the zone ".format(sent))
            return [collection.filter(ee.Filter.eq(ORBIT_ID[sent], tile_id))]
        sub_collection = collection.filter(ee.Filter.eq(ORBIT_ID[sent], tile_id))  # filter by the tilesid
        list_subcollection += [sub_collection]

        assert sub_collection.toList(100).length().getInfo() > 0, "Subcollection is empty  ..."
    assert len(list_subcollection) > 0, "Wrong size of the list subcollection {}".format(len(list_subcollection))
    return list_subcollection


def extract_fp(image, sent=0):
    """    :param sent:
:returns an ee.Geometry
    """
    fp = ee.Image(image).get("system:footprint")
    coordo = ee.Geometry(fp).coordinates()
    return ee.Geometry.Polygon(coordo)
    # print(type(fp))


def check_clip_area(zone, zone_sent2):
    ar_zone = zone.area(0.001).getInfo()
    ar_zone_sent2 = zone_sent2.area(0.001).getInfo()
    if ar_zone < ar_zone_sent2:
        print("There is an issue the area of the zone to dwnld is smaller than the area of the intersection ")
        print("zone : {} sent2: {}".format(ar_zone, ar_zone_sent2))
        return ee.Geometry(zone.intersection(zone_sent2, 0.001))
    else:
        print("Everything normal")
        return ee.Geometry(zone_sent2)
    pass


def get_biggest_s1_image(zone: ee.Geometry, ImageCollection: ee.ImageCollection) -> Tuple[bool, ee.ImageCollection]:
    """

    Args:
        zone: a ee.Geometry
        ImageCollection: an ee.ImageCollection

    Returns:
        - a boolean wether or not one image in the Imagecollection has a footprint that cover FACTEUR AREA proportion of the
        input zone
        - an ee.ImageCollection of len 1, which contains the  image that covers the maximum amount of the zone
    """

    list_image = ee.List(ImageCollection.toList(100))
    n = list_image.length().getInfo()
    max_area_geom = extract_fp(ee.Image(list_image.get(0)))
    final_image = ee.Image(list_image.get(0))
    if n == 0:
        return False,ImageCollection
    else:
        for i in range(1, n):
            image = ee.Image(list_image.get(i))
            geo = extract_fp(image)
            if geo.area(0.001).getInfo() > max_area_geom.area(0.001).getInfo():
                max_area_geom = geo
                final_image = image
        if max_area_geom.area(0.001).getInfo() >= FACTEUR_AREA * zone.area(0.001).getInfo():
            print("The geometries of the Image Collection contains the geometry")
            return True, ee.ImageCollection(final_image)
        else:
            return False, ee.ImageCollection(final_image)


def zone_in_images(zone, ImageCollection):
    """

    Args:
        zone: ee.Geometry
        ImageCollection: an ee.ImageCollection

    Returns:
        bool wether or not the Images in the image collection represent the area
    """

    list_inter = []  # list of intersection of the fp of all the images with the zone (ee.Geometry)
    list_image = ee.List(ImageCollection.toList(100))
    n = list_image.length().getInfo()
    if n == 0:
        return False
    else:
        for i in range(n):
            image = ee.Image(list_image.get(i))
            geo = extract_fp(image)
            geo_inter = geo.intersection(zone)
            list_inter = add_distinct_geom(list_inter, geo_inter)
        # compute the area
        area_union = get_list_area(list_inter)
        if area_union > FACTEUR_AREA * zone.area(0.001).getInfo():
            print("The geometries of the Image Collection contains the geometry")
            return True
        else:
            return False


def add_distinct_geom(list_geom, new_geom):
    """Given a list of geometry that does not intersect returns a list of the geometry that does not instersect with
    the union of new_geom """
    add_geom = True  # boolean True if the new_geom has no intersection with the geom of the list
    print("The list is len {}".format(len(list_geom)))
    if len(list_geom) == 1:
        if new_geom.intersects(list_geom[0]):
            print("There is a union between the two geometry ")
            # there is an intersection between the two element : return in a list the union
            return [new_geom.union(list_geom[0])]
        else:
            print("There is no union")
            return list_geom + [new_geom]
    elif len(list_geom) == 0:
        return [new_geom]
    else:
        for i, geom in enumerate(list_geom):
            if new_geom.intersects(geom):
                add_geom = False
                list_geom.pop(i)
                return add_distinct_geom(list_geom, new_geom.union(geom))
        if add_geom is True:
            print("There is no intersection between new geom and all the elements of the list")
            return list_geom + [new_geom]


def get_list_area(list_geom):
    """Given a list of geometry returns the total area"""
    total_area = 0
    for geom in list_geom:
        total_area += geom.area(0.001).getInfo()
    return total_area


def merge_image_collection(collection1, collection2):
    """Function which check if one the fp of one collection is include into another one, if yes keep the one with the biggest intersection area"""
    pass

# TODO do a function which check if the initial zone is include on both fp is it is keep only the images.clip(ini_zone) où il y
##TODO a le moins d'image. Pareil pour sentinel 1 : prendre la zone
