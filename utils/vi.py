# A file which contains all the information concerning the vegetation index functions
from constant.fire_severity_constant import THR_VI_UNBURNED
from constant.gee_constant import DICT_BAND_LABEL,DICT_BAND_X,DICT_EVI_PARAM
import numpy as np

from utils.normalize import get_minmax_fromcsv, normalization


def extract_red_nir(image,dict_band):
    band_red = image[:, :, dict_band["R"][0]]
    band_nir = image[:, :, dict_band["NIR"][0]]
    return band_red,band_nir



def compute_ndvi(image,dict_band=None,path_csv=None,image_id=None):
    """:param image an array
    :param dict_band a dict with the position of the band within the array
    :param path_csv path to the csv which contains global max or min of the array
    :param image_id string, needs to be set if path_csv not None
    :returns the ndvi as an array"""
    if dict_band is None:
        print("We consider it is a predicted image with R G B NIR only ")
        dict_band=DICT_BAND_LABEL
    assert len(image.shape)==3,"Wrong dimension of the image should be 3 not {}".format(image.shape)
    assert image.shape[-1]<image.shape[0],"Check the dimension of the image should be channel last. {}".format(image.shape)
    band_red,band_nir=extract_red_nir(image,dict_band)
    #print("NDVI")
    mask = (band_nir+band_red)==0
    ndvi = np.zeros(band_nir.shape)
    ndvi[  mask ] = 0
    print(np.count_nonzero(ndvi))
    ndvi[ ~mask ] = ((band_nir-band_red)/(band_nir+band_red))[ ~mask ]
    if path_csv is not None: #the normalization is going to occur
        assert image_id is not None,"Normalized NDVI requested BUT image_id not given"
        tile_min,tile_max=get_minmax_fromcsv(image_id,path_csv,"ndvi",1)
        return normalization(ndvi,tile_min,tile_max)
    else:
        return ndvi

def compute_bai(image,dict_band=None):
    if dict_band is None:
        print("We consider it is a predicted image with R G B NIR only ")
        dict_band=DICT_BAND_LABEL
    band_red, band_nir = extract_red_nir(image, dict_band)
    terme1=np.square(0.1-band_red)
    terme2=np.square(0.06-band_nir)
    return np.divide(1,terme1*terme2)

def compute_msavi(image,dict_band=None):
    if dict_band is None:
        print("We consider it is a predicted image with R G B NIR only ")
        dict_band=DICT_BAND_LABEL
    band_red, band_nir = extract_red_nir(image, dict_band)
    sqrt_terme=np.square(2*band_nir+1)-8*(band_nir-band_red)
    return np.divide(2*band_nir+1-np.sqrt(sqrt_terme),2)

def compute_vi(image,vi,dict_band=None,param=None,path_csv=None,image_id=None):
    """vi a string of the vegetation index"""
    if vi!= "identity":
        assert len(image.shape)==3, "Image shape (n,n,channel) only accepted not {} ".format(image.shape)
    if dict_band is None:
        print("We consider it is a predicted image with R G B NIR only ")
        dict_band=DICT_BAND_LABEL
    assert vi in ["msavi","bai","ndvi","identity","evi"], "The vegetation index {} has no function defined. please define a function in utils.vi".format(vi)
    if vi=="ndvi":
        return compute_ndvi(image,dict_band,path_csv=path_csv,image_id=image_id)
    if vi=="bai":
        return compute_bai(image,dict_band)
    if vi=="msavi":
        return compute_msavi(image,dict_band)
    if vi=="identity":
        return image
    if vi=="evi":
        return compute_evi(image,dict_band,param)

def compute_evi(image,dict_band,param=None):
    if param is None:
        param=DICT_EVI_PARAM
    red=image[:,:,dict_band["R"]]
    nir=image[:,:,dict_band["NIR"]]
    blue=image[:, :, dict_band["B"]]
    evi_res= param["G"]*np.divide(nir-red,nir+param["C1"]*red-param["C2"]*blue+param["L"])
    return np.resize(evi_res,(image.shape[0],image.shape[1]))


def diff_metric(image_pre, image_post, vi, dict_band_pre=None, dict_band_post=None, image_id=None, path_csv=None):
    """:param image_id:
    :param path_csv:
    :param image_pre the image before the event
    :param image post the image post transformation
    :vi a vegetation index could be msavu,bai or ndvi"""
    if dict_band_pre is None:
        dict_band_pre=DICT_BAND_X
    if dict_band_post is None:
        dict_band_post=DICT_BAND_LABEL
    pre_vi=compute_vi(image_pre,vi,dict_band_pre,image_id=image_id,path_csv=path_csv)
    post_vi=compute_vi(image_post,vi,dict_band_post,image_id=image_id,path_csv=path_csv)
    return pre_vi-post_vi

def diff_relative_metric(image_pre,image_post,vi,dict_band_pre=None,dict_band_post=None, image_id=None, path_csv=None):
    """:param image_pre the image before the event
        :param image post the image post transformation
        :vi a vegetation index could be msavu,bai or ndvi"""
    if dict_band_pre is None:
        dict_band_pre = DICT_BAND_X
    if dict_band_post is None:
        dict_band_post = DICT_BAND_LABEL
    pre_vi = compute_vi(image_pre, vi, dict_band_pre,image_id=image_id,path_csv=path_csv)
    post_vi = compute_vi(image_post, vi, dict_band_post,image_id=image_id,path_csv=path_csv)
    return np.divide(pre_vi-post_vi,np.sqrt(np.abs(pre_vi)))


def is_change(image_pre,image_post,vi,dict_band_pre=None,dict_band_post=None,path_csv=None,image_id=None,thr_vi=None):
    """:returns a boolean True if burned has occured on this tile between image_pre, image_post"""
    if thr_vi is None:
        thr_vi=THR_VI_UNBURNED
    dvi=diff_metric(image_pre, image_post, vi, dict_band_pre=dict_band_pre, dict_band_post=dict_band_post,
                    image_id=image_id, path_csv=path_csv)
    if np.median(dvi)<thr_vi:
        return False
    else:
        return True


def clean_changed_batch(batch_pre, batch_gt, batch_pred, vi, dict_band_pre=None, dict_band_post=None, path_csv=None, l_image_id=None, thr_vi=None):
    assert batch_pre.shape[0] == batch_gt.shape[0], "Wrong input shape post {} gt : {}".format(batch_pre.shape, batch_gt.shape)
    l_out_pre=[]
    l_out_gt=[]
    l_out_pred=[]
    if l_image_id is None:
        l_image_id=[None]*(batch_pre.shape[0])
    for i in range(batch_gt.shape[0]):
        if is_change(batch_pre[i, :, :, :],batch_gt[i,:,:,:], vi, dict_band_pre=dict_band_pre, dict_band_post=dict_band_post
                         , path_csv=path_csv, image_id=l_image_id, thr_vi=thr_vi):
            l_out_gt+=[batch_gt[i,:,:,:]]
            l_out_pre+=[batch_pre[i, :, :, :]]
            l_out_pred+=[batch_pred[i,:,:,:]]
    return np.array(l_out_pre),np.array(l_out_gt),np.array(l_out_pred)

