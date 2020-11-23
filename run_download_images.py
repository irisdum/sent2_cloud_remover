# File where the search of the image name in GEE scripts and the download procedure of auscohup are linked
import json

import ee
import argparse
import sys

from utils.download_images import download_all, create_download_dir
from find_image import get_filter_collection, list_image_name, opt_filter, gjson_2_eegeom, eedate_2_string
from utils.fp_functions import sub_collection_tiles, extract_fp, check_clip_area, zone_in_images
from constant.gee_constant import S1_OPTPARAM
from constant.storing_constant import DIR_T, OPT_DWND_IMAGE


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
    parser.add_argument("--ccp", type=int, default=20, help="Percentage of cloud allowed in the image, not really used anymore the roi_cpp defined in the constant file is more restrictive")
    parser.add_argument("--save", default=True, help="wether or not we are going to store the images")
    parser.add_argument("--output_path", default=True, help="where the image preprocess and ordered are going "
                                                            "to be stored")
    parser.add_argument("--s2_t0",default=None,help="If known name of the S2 image at t0 to download")
    parser.add_argument("--s2_t1",default=None,help="If known name of the S2 image at t0 to download")

    return parser.parse_args()


def default_param(collection):
    if collection == 1:
        return S1_OPTPARAM
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
    """

    Args:
        date_t: ee.Date
        zone: an ee.Geometry
        sent: int could be 1 or 2
        optparam1: dictionnary, sentinel 1 optional parameters
        i: nber of days to look at before and after the date_t
        opt_search: string, could be "before","after" or "both" respectively looks for image [date-i, date] or [date,date+i]
        or [date-i,date+i]

    Returns:
         length : int length of the ImageCollection of S1 image
         ImageCollection : The Sentinel 1 ImageCollection which respect the criteria of seach (input args)

    """
    """:return the length of the sentinel  1 collection at date +-i from the given date depending of the opt_search"""
    collection_sent1_t1_before = get_filter_collection(date_t.advance(-i, "day"), date_t, zone, sent,
                                                       optparam1)  # get the sentinel 1 collection
    collection_sent1_t1_after = get_filter_collection(date_t, date_t.advance(i, "day"), zone, sent, optparam1)
    if opt_search == "both":
        # print("both")
        total_collection = collection_sent1_t1_after.merge(collection_sent1_t1_before)
        len_collection = collection_lenght(total_collection)
        return len_collection, total_collection
    elif opt_search == "before":
        # print(opt_search)
        return collection_lenght(collection_sent1_t1_before), collection_sent1_t1_before
    else:
        # print(opt_search)
        assert opt_search == "after", "Wrong parameter opt_search should be after but is {}".format(opt_search)
        return collection_lenght(collection_sent1_t1_after), collection_sent1_t1_after


def get_sentinel1_image(date_t, zone, optparam1, opt_search="both", sent=1):
    """

    Args:
        date_t: ee.Date, the acquisition date of the Sentinel 2 image
        zone:  ee.Geometry, the S1 images downloaded should cover this area
        optparam1: dictionnary, Sentinel 1 optional filter
        opt_search:string, could be "before","after" or "both" respectively looks for image [date-i, date] or [date,date+i]
        sent:

    Returns:
    list_name_sent1: list of string, which corresponds to the id of the images contained in the final ImageCollection.
    Image Collection found from searching S1 images closest to Sentinel 2
     list_date_sent1: list of ee.Date
    """

    print("Test day +- {} from {}".format(0, date_t.format().getInfo()))
    i = 1
    total_len, dayli_collection = sent_image_search(date_t, zone, sent, optparam1, i, opt_search)
    all_found= zone_in_images(zone,dayli_collection) #boolean wether or not all the good images have been selected
    print(type(zone), zone.getInfo())
    final_collection=dayli_collection
    while all_found is False:  # iterate until a sentinel 1 image is found
        print("Test day +- {} from {}".format(i, date_t.format().getInfo()))
        i += 1
        total_len, dayli_collection = sent_image_search(date_t, zone, sent, optparam1, i, opt_search)
        final_collection=final_collection.merge(dayli_collection)
        #TODO : make a function which merge the collection : if one geometry of the image is include into another one : remove the includede image
        all_found=zone_in_images(zone,final_collection)
    print("Number of image found of sent 1  found {} at {} days from sentinel 2 ".format(total_len, i))

    final_list = list_image_name(final_collection, sent)
    assert len(final_list) > 0, "Pb the list is empty {}".format(final_list)

    list_subcol_sent1 = sub_collection_tiles(final_collection, zone, sent)  # Get subcollections list
    list_name_sent1=[]
    list_date_sent1=[]
    for sub_list in list_subcol_sent1:
        list_name_sent1 += list_image_name(sub_list, sent)
        list_date_sent1 += [sub_list.first().date() for image in list_image_name(sub_list, sent)]
    # for subcol in list_subcol_sent1:
    # name, date_coll, _ = extract_name_date_first(subcol, 1)
    # print("Sentinel 1 collected at {} for sent2 collected at {}".format(date_coll.format().getInfo(),
    # date_t.format().getInfo()))
    # list_name_sent1 += [name]  # collect the name of the sent1 images

    return list_name_sent1, list_date_sent1


def clip_on_geometry(geometry):
    """

    Args:
        geometry: an ee.Geometry

    Returns: a function which corresponds to clipping an ee.Image along the input geometry

    """
    def clip0(image):
        return image.clip(geometry)

    return clip0


def sent2_filter_clouds(collection, sent2criteria, ccp, zone):
    """

    Args:
        collection: a ee.ImageCollection
        sent2criteria: string should be begin, end, lessclouds :  respectively the image the closest to the begin date,
        end date or the less clouds
        ccp: int maximum cloud percentage accepted on the S2 image
        zone: an ee.Geometry, define the BBOX

    Returns:
        Three argument which corresponds to a string name of the image of the first image of the collection, an ee.Date which
        is the date of the first the Image of the output collection, the footprint of the first image of the collection
    """
   #TODO check the use of this function, used the new one which claculate the ccp on the area !!!
    print("before clipping length collection = {}".format(collection.toList(100).length().getInfo()))

    collection_zone = collection.map(clip_on_geometry(zone))
    assert collection_zone.toList(100).length().getInfo() > 0, "The clip function does not work"
    print("after clipping length = {}".format(collection_zone.toList(100).length().getInfo()))
    collection_zone = opt_filter(collection_zone, {"ccp": ccp}, 2)
    assert collection_zone.toList(100).length().getInfo() > 0, "No sentinel 2 image found with the ccp {}".format(ccp)
    # Sort all these images, choose the one with less clouds or the image the closer to bd1 ed1
    if sent2criteria == "begin":
        collection_zone = collection_zone.sort("system:time_start")
    elif sent2criteria == "end":
        collection_zone = collection_zone.sort("system:time_start", False)
    else:
        assert sent2criteria == "lessclouds", "Wrong parameter sent2criteria  {} should be in begin,end,lessclouds" \
            .format(sent2criteria)
        collection_zone = collection_zone.sort('CLOUDY_PERCENTAGE_ROI')
    # The image are sorted along one criteria, we returns the information for the first image of the sorted collection
    return extract_name_date_first(collection_zone, 2)


def extract_name_date_first(collection, sent):
    """

    Args:
        collection: an ee.ImageCollection (ee stands for earth engine library)
        sent: int could be 1 or 2

    Returns:
        name : string name of the image of the first image of the collection
        date_coll: an ee.Date, which is the date of acquisition of the first image of the collection
        zone: a ee.Geometry which is the foortprint of the first image of the collectoion

    """
    date_coll = collection.first().date()
    if sent == 1:
        name = collection.first().get("system:id").getInfo()
    else:
        # print("sent2")
        name = ee.Image(collection.first()).get("PRODUCT_ID").getInfo()
        # print("here")
    zone = extract_fp(collection.first(), sent)

    return name, date_coll, zone  # TODO take care of the zone format read


def download_sent2_sent1(bd, ed, zone, sent2criteria, optparam1, ccp,name_s2):
    """

    Args:
        bd: string, begin date
        ed: string, end date
        zone: ee.Geometry
        sent2criteria: string should be "begin", "end" or "lessclouds" : respectively  select s2  image the closest to
        the begin date,end date or the less clouds
        optparam1: dictionnary or None, optional filter top apply on s1 image
        ccp: int, maximum cloud percentage in the image accepted, used only if name_s2 is not defined
        name_s2: string or None, name_id of the S2 image

    Returns:
        Two dictionnaries respectively for sentinel 1 and two. The keys of these dictionnaries are the image id and the
        values are the date of the image

    """

    dict_image_dwnld1 = {}
    dict_image_dwnld2 = {}
    list_sent1_sent2_name = []
    # Extract the Image collection of sentinel 2 between the range dates
    print("With {} - {} looking for {}".format(bd,ed,name_s2))
    global_collection_sent2_t1 = get_filter_collection(bd, ed, zone, 2, opt_param={
        "ccp": ccp},name_s2=name_s2)
    # Extract the List of subcollection with one subcollection = image between the range date
    # at one special tile
    print(type(global_collection_sent2_t1))
    print(global_collection_sent2_t1.toList(100).length().getInfo())
    list_subcol_sent2_t1 = sub_collection_tiles(global_collection_sent2_t1, zone, 2)
    list_name_sent2 = []  # Will contains the name, date and fp of the required sentinel 2 Images
    list_name_sent1 = []
    assert len(list_subcol_sent2_t1) > 0, "No sentinel 2 list of subcollection has been created"
    for sub_col in list_subcol_sent2_t1:  # Go over all the different subcollection

        name, date1_sent2_subcol, zone_sent2 = sent2_filter_clouds(sub_col, sent2criteria,
                                                                   ccp, zone)  # returns the image with less clouds
        # on the specific zone which is the intersection of the two
        #print("zone {}".format(type(zone)))
        #print("zone  sent2 {}".format(type(zone)))
        new_zone = check_clip_area(zone,
                                   zone_sent2)  # corresponds to the intersection of the sent2 fp and the zone to download

        # print("Zone {}".format(zone_sent2.coordinates().getInfo()))
        list_name_sent2 += [name]  # save the name of the sent2 image at t1 to download
        # we extract the footprint of sentinel 2 : we extract now all the sentinel 1 images which can reproduce this
        # image
        dict_image_dwnld2.update({name: eedate_2_string(date1_sent2_subcol)})
        list_name_sent1, list_date_sent1 = get_sentinel1_image(date1_sent2_subcol, new_zone, optparam1, "both")
        dict_image_dwnld1.update(
            dict(zip(list_name_sent1, [eedate_2_string(date) for date in list_date_sent1])))
    list_sent1_sent2_name += list_name_sent2 + list_name_sent1  # collect all the names
    return dict_image_dwnld1, dict_image_dwnld2


def main(bd, ed, bd2, ed2, path_zone, sent2criteria, optparam1, ccp, save, output_path, s2_t0=None,s2_t1=None):
    """

    Args:
        bd: string, begin date 1
        ed: string, end date 1
        bd2: string, begin date 2
        ed2: string, end date 2
        path_zone: string, path to a Polygon geometry geojson
        sent2criteria: string, schoos if the sent2 cloud free images should be closer to the begin date or end date
        optparam1: None or dictionnary, contains filter for Sentinel 1 ImageCollection
        ccp: int, maximum cloud percentage accpeted for an s2 image (used if s2_t0 or s2_t1 are undefined)
        save: bool, if set to True the image are saved
        output_path: string, path to the directory where the images are going to be saved
        s2_t0: string or None, sentinel 2 image id for date 1
        s2_t1: string or None, sentinel 2 image id for date 2

    Returns:

    """


    assert create_download_dir(output_path), "Download directory has not been well created"

    if optparam1 is None:
        optparam1 = default_param(1)
    else:
        optparam1 = json.loads(optparam1)
    # First we are looking for the first image with with less than ccp percentage of clouds
    # print("CCP val : {} type :{}".format(ccp,type(ccp)))
    # print({"ccp": ccp})
    zone_sent2_init = gjson_2_eegeom(path_zone)
    print(type(zone_sent2_init))
    dic_name_t1_sent1, dic_name_t1_sent2 = download_sent2_sent1(bd, ed, zone_sent2_init, sent2criteria, optparam1, ccp,s2_t0)
    print("{} {}".format(bd2, ed2))
    print(dic_name_t1_sent1, dic_name_t1_sent2, )
    dic_name_t2_sent1, dic_name_t2_sent2 = download_sent2_sent1(bd2, ed2, zone_sent2_init, sent2criteria, optparam1,
                                                                ccp,s2_t1)
    print(dic_name_t2_sent1, dic_name_t2_sent2, )

    if save:
        download_all(dic_name_t2_sent1, sent=1,output_path= output_path+DIR_T[1],opt=OPT_DWND_IMAGE)
        download_all(dic_name_t1_sent1, sent=1, output_path=output_path+DIR_T[0],opt=OPT_DWND_IMAGE)
        download_all(dic_name_t2_sent2, sent=2,output_path= output_path+DIR_T[1],opt=OPT_DWND_IMAGE)
        download_all(dic_name_t1_sent2, sent=2,output_path= output_path+DIR_T[0],opt=OPT_DWND_IMAGE)

    else:
        return True

if __name__ == '__main__':

    sys.path.append("./")
    args = _argparser()
    main(args.bd1, args.ed1, args.bd2, args.ed2, args.zone, args.sent2criteria, args.optparam1, int(args.ccp),
         args.save,
         args.output_path,args.s2_t0,args.s2_t1)
