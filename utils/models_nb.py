# Some functions useful for jupyter notebooks used to study the models

from models import clean_gan
import glob
import os

from utils.display_image import find_image_indir, plot_all_compar


def predict_iter_on_val(path_model,training_nber,select_weight=100,save=True,plot=True):
    """Run a prediction of the model and save the images if required and plot them too"""
    path_model_yaml, path_train_yaml=get_important_path(path_model,training_nber)
    gan = clean_gan.GAN(path_model_yaml, path_train_yaml)
    l_weight = glob.glob("{}*h5".format(gan.checkpoint_dir))
    path_val=gan.val_directory
    l_image_name=find_image_indir(path_val, "npy")
    path_weight,founded=find_weight_path(l_weight,select_weight)
    assert founded is True,"No path weight nb {} founded in {}".format(select_weight,l_weight)
    gan_gen = gan.generator.load_weights(path_weight)
    if save:
        path_save=path_model + "training_{}/image_val_iter_{}/".format(training_nber,select_weight)
    else:
        path_save=None
    val_dataX, val_dataY = gan.val_X, gan.val_Y
    bath_res=gan.predict_on_iter(val_dataX,path_save,l_image_id=l_image_name)
    if plot:
        plot_all_compar(bath_res,val_dataY)
    return bath_res


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