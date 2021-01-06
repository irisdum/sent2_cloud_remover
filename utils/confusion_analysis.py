import numpy as np

from utils.old_dispay_vi import histo_val


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