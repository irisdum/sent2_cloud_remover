# A file which contains all the information concerning the vegetation index functions
from constant.gee_constant import DICT_BAND_LABEL,DICT_BAND_X
import numpy as np

def extract_red_nir(image,dict_band):
    band_red = image[:, :, dict_band["R"][0]]
    band_nir = image[:, :, dict_band["NIR"][0]]

    return band_red,band_nir

def compute_ndvi(image,dict_band=None):
    if dict_band is None:
        print("We consider it is a predicted image with R G B NIR only ")
        dict_band=DICT_BAND_LABEL
    assert len(image.shape)==3,"Wrong dimension of the image should be 3 not {}".format(image.shape)
    assert image.shape[-1]<image.shape[0],"Check the dimension of the image should be channel last. {}".format(image.shape)
    band_red,band_nir=extract_red_nir(image,dict_band)
    return np.divide(band_nir-band_red,band_nir+band_red)

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

def compute_vi(image,vi,dict_band=None):
    """vi a string of the vegetation index"""
    if dict_band is None:
        print("We consider it is a predicted image with R G B NIR only ")
        dict_band=DICT_BAND_LABEL
    assert vi in ["msavi","bai","ndvi","identity"], "The vegetation index {} has no function defined. please define a function in utils.vi".format(vi)
    if vi=="ndvi":
        return compute_ndvi(image,dict_band)
    if vi=="bai":
        return compute_bai(image,dict_band)
    if vi=="msavi":
        return compute_msavi(image,dict_band)
    if vi=="identity":
        return image

def diff_metric(image_pre,image_post,vi,dict_band_pre=None,dict_band_post=None):
    """:param image_pre the image before the event
    :param image post the image post transformation
    :vi a vegetation index could be msavu,bai or ndvi"""
    if dict_band_pre is None:
        dict_band_pre=DICT_BAND_X
    if dict_band_post is None:
        dict_band_post=DICT_BAND_LABEL
    pre_vi=compute_vi(image_pre,vi,dict_band_pre)
    post_vi=compute_vi(image_post,vi,dict_band_post)
    return pre_vi-post_vi

def diff_relative_metric(image_pre,image_post,vi,dict_band_pre=None,dict_band_post=None):
    """:param image_pre the image before the event
        :param image post the image post transformation
        :vi a vegetation index could be msavu,bai or ndvi"""
    if dict_band_pre is None:
        dict_band_pre = DICT_BAND_X
    if dict_band_post is None:
        dict_band_post = DICT_BAND_LABEL
    pre_vi = compute_vi(image_pre, vi, dict_band_pre)
    post_vi = compute_vi(image_post, vi, dict_band_post)
    return np.divide(pre_vi-post_vi,np.sqrt(np.abs(pre_vi)))
