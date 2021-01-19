import argparse

from utils.models_nb import predict_iter_on_val, get_important_path, find_weight_path
from keras.models import model_from_yaml
import glob

def _argparser():
    parser = argparse.ArgumentParser(description="Argument GAN train")
    parser.add_argument('--model_path', type=str, default="/datastore/dum031/trainings/K0_GAN_DBN_SIG_LRELU/",
                        help="path to yaml model ")
    parser.add_argument("--tr_nber", type=int, default=15)
    parser.add_argument("--weights", nargs="+", required=True,help="List of the weight on which to predict")
    parser.add_argument("--dataset",type=str,default=None, help="List of the weight on which to predict")
    parser.add_argument("--pref", type=str,default="val", help="prefix of the saved image")
    parser.add_argument("--path_csv", type=str, default=None, help="path to the xls with normalization cst")
    return parser.parse_args()

def load_generator(path_yaml, path_weight):
    # load YAML and create model
    yaml_file = open(path_yaml, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights(path_weight)
    print("Loaded model from disk")
    return loaded_model

def main(path_model,training_nber,l_weight,dataset=None,pref="val",path_csv=None):
    print(type(l_weight))
    assert len(l_weight) > 0, "No prediction will be made as no weights given "
    path_training_dir = path_model + "training_{}/".format(training_nber)
    path_checkpoints="{}checkpoints/".format(path_training_dir)
    l = glob.glob("{}*model_generator.yaml".format(path_checkpoints)) #load the generator archi
    assert len(l) == 1, "Wrong nber file found {} at {}".format(l, "{}*model_generator.yaml".format(path_checkpoints))

    for w in l_weight:
        #lw = glob.glob("{}*gene*".format(path_checkpoints))
        #path_weight, founded = find_weight_path(lw,w)
        #assert founded is True, "No path weight nb {} founded in {}".format(w, lw)
        #generator=load_generator(l[0], path_weight) # laod the weight
        predict_iter_on_val(path_model, training_nber, select_weight=w, save=True,dataset=dataset,prefix_save=pref,
                            path_csv=path_csv,generator=None)

if __name__ == '__main__':
    parser = _argparser()
    main(parser.model_path,parser.tr_nber,parser.weights,parser.dataset,parser.pref,parser.path_csv)

