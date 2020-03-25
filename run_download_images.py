# File where the search of the image name in GEE scripts and the download procedure of auscohup are linked
import json
import ee
import argparse

from find_image import get_filter_collection, list_image_name, opt_filter, gjson_2_eegeom
from fp_functions import sub_collection_tiles, extract_fp


def _argparser():
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('--bd1', type=str, help="begin date where we are lookinf for a sentinel 2 cloud free image")
    parser.add_argument('--ed1', type=str,
                        help="ending date where we are looking for the sentinel 2  clouf free images")
    parser.add_argument("--bd2", type=str,
                        help="t2 corresponds to the second date for cloud free sent2 images"
                             "acquisition")
    parser.add_argument('--ed2', type=str,
                        help="ending date t2 corresponds to the second date for cloud free sent2 images"
                             "acquisition")
    parser.add_argument('--zone', type=str, help="path where the zone coordinates are stored ")
    parser.add_argument("--sent2criteria", type=str, default="lessclouds",
                        help="sent 2 criteria to select the image begin or"
                             " end returns the image closer to the begin or "
                             "end date or could be lessclouds to returns the "
                             "sentinel 2 image between bd1 and bd2 with less clouds")
    parser.add_argument("--optparam1", type=str, default=None, help="optional parameters to filter the data of "
                                                                    "sentinel 1 ")
    parser.add_argument("--ccp", type=int, default=20, help="Percentage of cloud allowed in the image")

    return parser.parse_args()


def default_param(collection):
    if collection == 1:
        return {"productType": "GRD", "sensorMode": "IW", "instrument": "C-SAR", "polarisation": "VV,VH"}
    elif collection == 2:
        pass
    else:
        assert True, "Wrong collection name : should be 1 or 2 but is {}".format(collection)
    # TODO see the parameters for sentinel 2 !


def collection_lenght(collection):
    """:param collection: an ee.ImageCollection
    :returns an int"""
    return collection.toList(1000).length().getInfo()


def sent_image_search(date_t, zone, sent, optparam1, i, opt_search):
    """:return the length of the sentinel  1 collection at date +-i from the given date depending of the opt_search"""
    collection_sent1_t1_before = get_filter_collection(date_t.advance(-i, "day"), date_t, zone, sent,
                                                       optparam1)  # get the sentinel 1 collection
    collection_sent1_t1_after = get_filter_collection(date_t, date_t.advance(i, "day"), zone, sent, optparam1)
    if opt_search == "both":
        print("both")
        total_collection = collection_sent1_t1_after.merge(collection_sent1_t1_before)
        len_collection = collection_lenght(total_collection)
        return len_collection, total_collection
    elif opt_search == "before":
        print(opt_search)
        return collection_lenght(collection_sent1_t1_before), collection_sent1_t1_before
    else:
        print(opt_search)
        assert opt_search == "after", "Wrong parameter opt_search should be after but is {}".format(opt_search)
        return collection_lenght(collection_sent1_t1_after), collection_sent1_t1_after


def get_sentinel1_image(date_t, zone, optparam1, opt_search="both", sent=1):
    """    :param sent:
    :param opt_search:
    :param date_t : an ee.Date from which we want to find the closest sentinel 1 image
    :param zone : correspond to the path to the geojson
    :param optparam1 : corresponds to seninel1 optional filters
    :return the name of the sentinel 1 images closer to date_t1 and an ee.Date which corresponds to the acquisition time
    """
    print("Test day +- {} from {}".format(0, date_t.format().getInfo()))
    i = 1
    total_len, total_collection = sent_image_search(date_t, zone, sent, optparam1, i, opt_search)
    print(type(zone), zone.getInfo())
    while total_len < 1:  # iterate until a sentinel 1 image is found
        print("Test day +- {} from {}".format(i, date_t.format().getInfo()))
        i += 1
        total_len, total_collection = sent_image_search(date_t, zone, sent, optparam1, i, opt_search)
    print("Number of image found of sent 1  found {} at {} days from sentinel 2 ".format(total_len,i))

    final_list = list_image_name(total_collection, sent)
    assert len(final_list) > 0, "Pb the list is empty {}".format(final_list)

    list_subcol_sent1 = sub_collection_tiles(total_collection, sent)  # Get subcollections list
    list_name_sent1 = list_image_name(list_subcol_sent1[0], sent)
    #for subcol in list_subcol_sent1:
        #name, date_coll, _ = extract_name_date_first(subcol, 1)
        #print("Sentinel 1 collected at {} for sent2 collected at {}".format(date_coll.format().getInfo(),
                                                                           # date_t.format().getInfo()))
        #list_name_sent1 += [name]  # collect the name of the sent1 images

    return list_name_sent1


def sent2_filter_clouds(collection, sent2criteria, ccp):
    """ Given a ee.ImageCollection returns the name of the image with cloud pixel coverage below than ccp and that fit
     sent2criteria"""
    collection = opt_filter(collection, {"ccp": ccp}, 2)
    assert collection.toList(100).length().getInfo() > 0, "No sentinel 2 image found with the ccp {}".format(ccp)
    # Sort all these images, choose the one with less clouds or the image the closer to bd1 ed1
    if sent2criteria == "begin":
        collection = collection.sort("system:time_start")
    elif sent2criteria == "end":
        collection = collection.sort("system:time_start", False)
    else:
        assert sent2criteria == "lessclouds", "Wrong parameter sent2criteria  {} should be in begin,end,lessclouds" \
            .format(sent2criteria)
        collection = collection.sort('CLOUDY_PIXEL_PERCENTAGE')
    # assert collection.toList(100).length().getInfo()>0, "No sentinel 2 image found with the ccp {}".format(ccp)
    return extract_name_date_first(collection, 2)


def extract_name_date_first(collection, sent):
    """Extract the name and the date of the first image of the collection"""
    date_coll = collection.first().date()
    if sent == 1:
        name = collection.first().get("system:id").getInfo()
    else:
        #print("sent2")
        name = ee.Image(collection.first()).get("PRODUCT_ID").getInfo()
        #print("here")
    zone = extract_fp(collection.first())

    return name, date_coll, zone  # TODO take care of the zone format read


def download_sent2_sent1(bd, ed, zone, sent2criteria, optparam1, ccp):
    """    :param ccp:
    :param optparam1:
    :param sent2criteria:
:param zone : ee.Geometry
    """

    list_sent1_sent2_name = []
    # Extract the Image collection of sentinel 2 between the range dates
    global_collection_sent2_t1 = get_filter_collection(bd, ed, zone, 2, opt_param={
        "ccp": ccp})
    # Extract the List of subcollection with one subcollection = image between the range date
    # at one special tile
    list_subcol_sent2_t1 = sub_collection_tiles(global_collection_sent2_t1, 2)
    list_name_sent2 = []  # Will contains the name, date and fp of the required sentinel 2 Images
    list_name_sent1 = []
    assert len(list_subcol_sent2_t1) > 0, "No sentinel 2 list of subcollection has been created"
    for sub_col in list_subcol_sent2_t1:  # Go over all the different subcollection

        name, date1_sent2_subcol, zone_sent2 = sent2_filter_clouds(sub_col, sent2criteria,
                                                                   ccp)  # filter the image with no many clouds
        # print("Zone {}".format(zone_sent2.coordinates().getInfo()))
        list_name_sent2 += [name]  # save the name of the sent2 image at t1 to download
        # we extract the footprint of sentinel 2 : we extract now all the sentinel 1 images which can reproduce this
        # image
        list_name_sent1 = get_sentinel1_image(date1_sent2_subcol, zone_sent2, optparam1, "both")

    list_sent1_sent2_name += list_name_sent2 + list_name_sent1  # collect all the names

    return list_sent1_sent2_name


def main(bd, ed, bd2,ed2, path_zone, sent2criteria, optparam1, ccp):
    """
    :param bd2:
    :param ccp: cloud pixel coverage percentage
    :param path_zone: path to the geojson which contains the zone description
    :param ed2:
    :param sent2criteria: used to choose wether sent2 images cloud free chosern should be closer to the begin date or
    ending date
    :param optparam1: None or a str which contains optional filter to be applied to the satellite data
    :param bd : string begin date from where we are looking for sentinel 2
    :param ed : string ending date
    """
    if optparam1 is None:
        optparam1 = default_param(1)
    else:
        optparam1 = json.loads(optparam1)
    # First we are looking for the first image with with less than ccp percentage of clouds
    # print("CCP val : {} type :{}".format(ccp,type(ccp)))
    # print({"ccp": ccp})
    zone_sent2_init = gjson_2_eegeom(path_zone)
    list_name_t1 = download_sent2_sent1(bd, ed, zone_sent2_init, sent2criteria, optparam1, ccp)
    print("{} {}".format(bd2,ed2))
    list_name_t2 = download_sent2_sent1(bd2, ed2, zone_sent2_init, sent2criteria, optparam1, ccp)
    list_total=list_name_t1+list_name_t2
    print(list_total)


if __name__ == '__main__':
    args = _argparser()
    main(args.bd1, args.ed1, args.bd2, args.ed2, args.zone, args.sent2criteria, args.optparam1, int(args.ccp))
