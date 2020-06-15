import buzzard as buzz

def open_tif(path_tif):
    ds = buzz.Dataset()
    r = ds.open_raster("raster", path_tif)
    ds = buzz.Dataset(allow_interpolation=True)
    ds.open_raster('sent', path_tif)
    total_fp=ds.sent.fp
