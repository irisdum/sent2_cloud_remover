from constant.storing_constant import TEMPORARY_DIR, XDIR, LABEL_DIR, URL_FILE
from utils.image_find_tbx import create_safe_directory


def create_tiling_hierarchy(output_dir: str):
    """
    It will create a directory tree : with output dir:{XDIR:{.,TEMPORARY_DIR},LABEL_DIR:{.,TEMPORARY_DIR}}
    Args:
        output_dir: path to the directory

    Returns:

    """
    create_safe_directory(output_dir)
    for cst in [XDIR, LABEL_DIR]:
        #     print("BUILDING DATA {}".format(cst))
        create_safe_directory(output_dir + cst)
        create_safe_directory(output_dir + cst + TEMPORARY_DIR)

def save_url(output_dir,url):
    f=open(output_dir+URL_FILE,"a")
    f.write("{} \n".format(url))
    f.close()