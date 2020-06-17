from constant.fire_severity_constant import DICT_FIRE_SEV_CLASS
import numpy as np

from constant.gee_constant import DICT_BAND_LABEL
from utils.vi import diff_metric
from sklearn.metrics import confusion_matrix


def batch_map_fire_sev(batch_x, batch_predict, batch_gt, max_im=100, vi="ndvi", dict_burned_class=None):
    if dict_burned_class is None:
        dict_burned_class = DICT_FIRE_SEV_CLASS
    n = batch_predict.shape[0]
    if n < max_im:
        max_im = n
    output_shape = (batch_gt.shape[0], batch_gt.shape[1], batch_gt.shape[1])
    batch_output_sev = np.ones(output_shape)
    batch_pred_sev = np.ones(output_shape)
    for i in range(max_im):
        image_pre_fire = batch_x[i, :, :, :]
        image_post = batch_gt[i, :, :, :]
        image_pred = batch_predict[i, :, :, ]

        # print_array_stat(image_pred)
        gt_dvi = diff_metric(image_pre_fire, image_post, vi, dict_band_pre={"R": [4], "NIR": [7]},
                             dict_band_post=DICT_BAND_LABEL)
        # print('PRED DIF ')
        pred_dvi = diff_metric(image_pre_fire, image_pred, vi, dict_band_pre={"R": [4], "NIR": [7]},
                               dict_band_post=DICT_BAND_LABEL)
        fire_sev_pred = get_fire_severity(pred_dvi, dict_burned_class)
        fire_sev_gt = get_fire_severity(gt_dvi, dict_burned_class)
        batch_output_sev[i, :, :] = fire_sev_gt
        batch_pred_sev[i, :, :] = fire_sev_pred

    return batch_output_sev, batch_pred_sev


def get_fire_severity(array_dndvi, dict_classe=None):
    """:param array_dndvi : numpy array:
     :returns an array with value from 1 to len(dict_class)"""
    print("The min {} the max {}".format(np.min(array_dndvi), np.max(array_dndvi)))
    thr_dndvi = np.ones(array_dndvi.shape)
    if dict_classe is None:
        dict_classe = DICT_FIRE_SEV_CLASS
    for i in range(array_dndvi.shape[0]):
        for j in range(array_dndvi.shape[1]):
            pix_class = get_pixels_class(array_dndvi[i, j], dict_classe)
            thr_dndvi[i, j] = pix_class
    # assert np.max(thr_dndvi) < 10000, "A pixel value is not in the classes defined"
    return thr_dndvi


def is_into(val, val_min, val_max):
    # print(round(val,4),val_max)
    if round(val, 4) >= val_min and round(val, 4) < val_max:
        return True
    else:
        return False


def get_pixels_class(pixels, dict_class):
    """:param pixels a float
    :param dict_class a dictionnary {"name classe": (min,max) ...}"""
    # print(dict_class.values())
    for i, tuple_val in enumerate(dict_class.values()):
        # print(i,tuple_val)
        if is_into(pixels, tuple_val[0], tuple_val[1]):
            # print(pixels,tuple_val[0], tuple_val[1])
            return i
    return -1


def confusion_seg_map(y, y_pred, plot=False):
    cf_matrix = confusion_matrix(y.ravel(), y_pred.ravel())
    return cf_matrix


def normalize_cf(cf, axis=0):
    vect_sum=cf.astype(np.float).sum(axis=axis)[np.newaxis]
    if axis==0:
        return cf / vect_sum
    else:
        return cf / vect_sum.T