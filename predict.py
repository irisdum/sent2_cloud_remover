import argparse

from utils.models_nb import predict_iter_on_val


def _argparser():
    parser = argparse.ArgumentParser(description="Argument GAN train")
    parser.add_argument('--model_path', type=str, default="/datastore/dum031/trainings/K0_GAN_DBN_SIG_LRELU/",
                        help="path to yaml model ")
    parser.add_argument("--tr_nber", type=int, default=15)
    parser.add_argument("--weights", nargs="+", required=True,help="List of the weight on which to predict")
    return parser.parse_args()


def main(path_model,training_nber,l_weight):
    print(type(l_weight))
    assert len(l_weight) > 0, "No prediction will be made as no weights given "
    for w in l_weight:
        predict_iter_on_val(path_model, training_nber, select_weight=w, save=True, plot=False)

if __name__ == '__main__':
    parser = _argparser()
    main(parser.model_path,parser.tr_nber,parser.weights)

