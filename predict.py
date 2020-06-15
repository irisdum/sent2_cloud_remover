import argparse

from utils.models_nb import predict_iter_on_val


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


def main(path_model,training_nber,l_weight,dataset=None,pref="val",path_csv=None):
    print(type(l_weight))
    assert len(l_weight) > 0, "No prediction will be made as no weights given "
    for w in l_weight:
        predict_iter_on_val(path_model, training_nber, select_weight=w, save=True,dataset=dataset,prefix_save=pref,
                            path_csv=path_csv)

if __name__ == '__main__':
    parser = _argparser()
    main(parser.model_path,parser.tr_nber,parser.weights,parser.dataset,parser.pref,parser.path_csv)

