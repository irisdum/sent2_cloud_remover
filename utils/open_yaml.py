from ruamel import yaml
import os

def saving_yaml(path_yaml, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.system("cp {} {}".format(path_yaml, output_dir))


def open_yaml(path_yaml):
    with open(path_yaml) as f:
        return yaml.load(f)
