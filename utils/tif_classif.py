# Function to analyze the behavior of the Landclassification the slope and the aspect during the classification
import buzzard as buzz
import os
import numpy as np

from constant.dem_constant import DICT_CLASSE_SLOPE, DICT_ORIENTATION_PRECISE
from constant.fire_severity_constant import DICT_FIRE_SEV_CLASS
from constant.gee_constant import XDIR, DICT_ORGA
from constant.landclass_constant import LISTE_LAND_CLASS
from utils.display_image import histo_val
from utils.image_find_tbx import find_path
import pandas as pd



def load_tile_classif(input_dataset,list_id_tile,path_classif_tif,max_im):
    """:param input_dataset : str path to the  dataset which contains .tif
    :param list_id_tile : list of tile id
    :param path_classif_tif : str path to the landclass rejorj tiff, be careful that the projections match with the tiles tiff
    :param max_im an int : if < len tile id reduced the size of the output to max_im
    :returns a list of array which are land classification for each tile"""
    assert os.path.isfile(path_classif_tif),"No tif at {}".format(path_classif_tif)
    if max_im<len(list_id_tile):
        list_id_tile=list_id_tile[:max_im]
    batch_landclass=[]
    ds = buzz.Dataset(allow_interpolation=True)
    ds_tile=buzz.Dataset()
    ds.open_raster('land_class',path_classif_tif)
    for i,image_id in enumerate(list_id_tile):
        path=find_path(input_dataset + XDIR + DICT_ORGA[XDIR][0], image_id)
        with ds_tile.open_raster('tile',path).close:
            fp_tile=ds_tile.tile.fp
            tile_land_class=ds.land_class.get_data(fp=fp_tile)
            batch_landclass+=[tile_land_class]
    return batch_landclass


def get_vegeconfusion(batch_landclass, batch_bool):
    assert batch_landclass.shape == batch_bool.shape, "Input should have the same dim landclass {} bool {}".format(batch_landclass.shape,batch_bool.shape)
    b_conf_landclass=batch_landclass[batch_bool]
    return b_conf_landclass



def compute_land_class_stats(array_lc):
    """:param array_lc a numpy array
    :returns a dictionnary with for all classes {class_vege : stat}"""
    dim_tot=array_lc.size
    final_dic={}
    for i,classe_vege in enumerate(LISTE_LAND_CLASS):
        stat=np.count_nonzero(array_lc==i)/dim_tot
        final_dic.update({classe_vege:stat})
    return final_dic

def compute_batch_land_class_stat(list_arr_lc,path_tileid):
    """:param list_arr_lc
    :returns a pd dataframe with for all tiles the stats"""
    df=pd.DataFrame()
    assert len(list_arr_lc)==len(path_tileid),"The list of array lc len {} does not have the same lenght as list path tile {}".format(len(list_arr_lc),len(path_tileid))
    for i in range(len(list_arr_lc)):
        dic_stat=compute_land_class_stats(list_arr_lc[i])
        dic_stat.update({"tile_id":path_tileid[i]})
        df=df.append(dic_stat,ignore_index=True)
    return df

def get_vege_confusion(batch_landclass, batch_bool,cst=24):
    assert batch_landclass.shape == batch_bool.shape, "Input should have the same dim landclass {} bool {}".format(batch_landclass.shape,batch_bool.shape)
    output=np.copy(batch_landclass)
    output[batch_bool]=cst
    return output

def get_confusion(label_class,pred_class,thr=1):
    print(label_class.shape)
    conf=np.abs(label_class-pred_class)
    return conf<thr

def get_precise_conf(label_class, pred_class, nb_classe_label, nb_class_pred):
    """:param label_class a rray which contains fire sev class for label
    :param pred_class array with fire sev
    :param nb_class_pred an int
    :param nb_classe_label an int
    :returns a boolean where False is at the position where the two confusion between the two classes has been made"""
    print("We are looking at class nber {} from pred and nb {} from label".format(nb_class_pred, nb_classe_label))
    cond_1=(label_class != nb_classe_label)
    cond_2=(pred_class != nb_class_pred)
    return np.array(cond_1+cond_2,dtype=bool)

def slope_thr(batch_slope,dict_slope=None,return_dict_class=False):
    thr_batch_slope=np.ones(batch_slope.shape)
    if dict_slope is None:
        dict_slope=DICT_CLASSE_SLOPE
    for i,slope_class in enumerate(dict_slope):
        print(dict_slope[slope_class])
        val_min,val_max=dict_slope[slope_class]
        thr_batch_slope[(batch_slope>=val_min)&(batch_slope<val_max)]=i
    if return_dict_class:
        return thr_batch_slope,dict(zip(list(dict_slope.keys()),[i for i in range(len(dict_slope))]))
    else:
        return thr_batch_slope

def aspect_thr(batch_aspect,dict_r_aspect=None,return_dict_class=False):
    thr_batch_aspect=np.ones(batch_aspect.shape)
    if dict_r_aspect is None:
        dict_r_aspect=DICT_ORIENTATION_PRECISE
    for i,tuple_val in enumerate(dict_r_aspect):
        if len(tuple_val)==1: #only one val is given (-1 case)
            thr_batch_aspect[batch_aspect == tuple_val[0]]=i
        else:
            val_min, val_max=tuple_val
            thr_batch_aspect[(batch_aspect >= val_min) & (batch_aspect< val_max)] = i
    if return_dict_class:
        return thr_batch_aspect,dict(zip(list(dict_r_aspect.values),[i for i in range(len(dict_r_aspect))]))
    else:
        return thr_batch_aspect


def proba_wc_vege(batch_classif, batch_confusion, plot=True, N_tot=24, all_val=True, list_class=None):
    unique, counts = np.unique(batch_classif, return_counts=True)
    print("TOTAL CLASSE {} ".format(dict(zip(unique,counts))))
    freq = list(counts / np.sum(counts))
    dic_count_glob = dict(zip(unique, counts))
    unique_conf, count_conf = np.unique(batch_confusion, return_counts=True)
    print("Classe with confusion {} {}".format(unique_conf,count_conf))
    if N_tot in unique_conf:
        unique_conf, count_conf = unique_conf[:-1], count_conf[:-1]
    else:
        print("error N should be in unique {} {}".format(N_tot,unique_conf))
    print("Classe with confusion after removing confusion {} {}".format(unique_conf,count_conf))
    dic_temp = dict(zip(unique_conf, count_conf))
    freq_final = []
    #sum_wc = np.sum(count_conf)
    sum_wc=np.sum(counts)
    #proba_conf=np.sum(count_conf)/np.sum(counts)
    #print(unique, unique_inter)
    print(dict(zip(unique_conf, count_conf/sum_wc)))
    for i in range(0, N_tot):
        if i not in unique_conf:
            freq_final += [0]
        else:
            #freq_inter = dic_temp[i] / np.sum(count_conf)*proba_conf
            if i not in dic_count_glob.keys():
                freq_final += [0]
            else:
                freq_final += [dic_temp[i]/ dic_count_glob[i]]
    print("Init {} Wrongly classified {}".format(np.sum(counts), np.sum(count_conf)))
    print("Conf percentage {}".format((np.sum(count_conf)/np.sum(counts))))
    unique_conf, count_conf = np.array(unique_conf), np.array(count_conf)
    dic_final = dict(zip([i for i in range(N_tot)], freq_final))
    if plot:
        print(dic_count_glob)
        histo_val(dic_count_glob,list_class=list_class)
        print(dict(zip(unique_conf, count_conf / sum_wc)))
        #histo_val(dict(zip(unique_inter, count_conf / sum_wc),list_class=list_class))
        histo_val(dic_final,list_class=list_class)
    if all_val:
        return dic_final, dic_count_glob, sum_wc, np.sum(counts)
    else:
        return dic_final