from constant.storing_constant import TEMPORARY_DIR, XDIR, LABEL_DIR
from utils.image_find_tbx import create_safe_directory


def create_tiling_hierarchy(output_dir):
    create_safe_directory(output_dir)
    for cst in [XDIR, LABEL_DIR]:
        #     print("BUILDING DATA {}".format(cst))
        create_safe_directory(output_dir + cst)
        create_safe_directory(output_dir + cst + TEMPORARY_DIR)