# Python file to scan the dataset and remove the non regular tile


def is_no_data(raster):
    """Given a raster check if there are no data:
    :returns bool"""
    pass

def is_s2_cloud(s2_raster):
    """Given a sentinel 2 raster check if the cloud mask band contains no_data
    :returns bool """
    pass

def extract_relative_path(path_tif):
    """Given the path to an tif tile returns its relative path within the Sentineli_tj directory"""
    l=path_tif.split("/")
    return "/".join(l[-3:-1])

def get_all_tiles_path(path_sent_dir):
    """Given the path to all """