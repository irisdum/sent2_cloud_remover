from constant.fire_severity_constant import DICT_FIRE_SEV_CLASS
import numpy as np


def get_fire_severity(array_dndvi, dict_classe):
    """:param array_dndvi : numpy array:
     :returns an array with value from 1 to len(dict_class)"""
    print("The min {} the max {}".format(np.min(array_dndvi), np.max(array_dndvi)))
    thr_dndvi = np.ones(array_dndvi.shape)
    if dict_classe is None:
        dict_classe = DICT_FIRE_SEV_CLASS
    for i in array_dndvi.shape[0]:
        for j in array_dndvi.shape[1]:
            pix_class = get_pixels_class(array_dndvi[i, j], dict_classe)
            thr_dndvi[i, j] = pix_class
    assert np.max(thr_dndvi) < 10000, "A pixel value is not in the classes defined"
    return thr_dndvi


def get_pixels_class(pixels, dict_class):
    """:param pixels a float
    :param dict_class a dictionnary {"name classe": (min,max) ...}"""
    for i, tuple_val in enumerate(dict_class.values()):
        if pixels in range(tuple_val[0], tuple_val[1]):
            return i
    return 10000
