# Some functions useful for jupyter notebooks used to study the models
from constant.storing_constant import XDIR
from models import clean_gan
import glob
import os

from utils.display_image import plot_all_compar
from utils.image_find_tbx import find_image_indir
from utils.open_yaml import open_yaml

def predict_iter_on_val(path_model, training_nber, select_weight=100, save=True, dataset=None,prefix_save="val",path_csv=None):
    """Run a prediction of the model and save the images if required and plot them too
    :param dataset: 
    """
    path_model_yaml, path_train_yaml=get_important_path(path_model,training_nber)
    gan = clean_gan.GAN(open_yaml(path_model_yaml), open_yaml(path_train_yaml))
    l_weight = glob.glob("{}*h5".format(gan.checkpoint_dir))
    if dataset is None:
        path_val=gan.val_directory
        val_dataX, val_dataY = gan.val_X, gan.val_Y
    else:
        assert os.path.isdir(dataset),"No dataset found at {}".format(dataset)
        print("We predict on {}".format(dataset))
        path_val=dataset
        val_dataX=dataset
    l_image_name=find_image_indir(path_val+XDIR, "npy")
    print("The val image founded are {}".format(l_image_name))
    assert len(l_image_name)>0, "No image found in val dir {}".format(path_val)
    path_weight,founded=find_weight_path(l_weight,select_weight)
    assert founded is True,"No path weight nb {} founded in {}".format(select_weight,l_weight)
    gan_gen = gan.generator.load_weights(path_weight)
    if save:
        path_save=path_model + "training_{}/image_{}_iter_{}/".format(training_nber,prefix_save,select_weight)
        print("saving image at {}".format(path_save))
    else:
        path_save=None
    bath_res= gan.predict_on_iter(val_dataX, path_save, l_image_id=l_image_name,path_csv=path_csv,un_rescale=True)

    return bath_res,gan


def get_important_path(path_model,training_nber):
    path_model_yaml = path_model + "model.yaml"
    assert os.path.isfile(path_model_yaml), "Wrong path model yaml {}".format(path_model_yaml)
    path_training_dir = path_model + "training_{}/".format(training_nber)
    lpath_train_yaml = glob.glob("{}*.yaml".format(path_training_dir))
    assert len(lpath_train_yaml) == 1, "Wrong selection train.yaml {}".format(lpath_train_yaml)
    path_train_yaml = lpath_train_yaml[0]
    return path_model_yaml,path_train_yaml


def find_weight_path(l_w,nb_w):
    """:param l_w : a list of str which are path to weights
    :param nb_w : another string which corresponds to the iteration we want
    :returns the path to weight of the training iteration nb_w"""
    for i,path_w in enumerate(l_w):
        if nb_w in path_w:
            return path_w,True
    else:
        return None,False