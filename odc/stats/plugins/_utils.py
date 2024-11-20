import dask
from osgeo import gdal, ogr, osr


def rasterize_vector_mask(
    shape_file, transform, dst_shape, filter_expression=None, threshold=None
):
    source_ds = ogr.Open(shape_file)
    source_layer = source_ds.GetLayer()

    if filter_expression is not None:
        source_layer.SetAttributeFilter(filter_expression)

    yt, xt = dst_shape[1:]
    no_data = 0
    albers = osr.SpatialReference()
    albers.ImportFromEPSG(3577)

    geotransform = (
        transform.c,
        transform.a,
        transform.b,
        transform.f,
        transform.d,
        transform.e,
    )
    target_ds = gdal.GetDriverByName("MEM").Create("", xt, yt, gdal.GDT_Byte)
    target_ds.SetGeoTransform(geotransform)
    target_ds.SetProjection(albers.ExportToWkt())
    mask = target_ds.GetRasterBand(1)
    mask.SetNoDataValue(no_data)
    gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[1])

    mask = mask.ReadAsArray()

    # used by landcover level3 urban
    # if valid area >= threshold
    # then the whole tile is valid

    if threshold is not None:
        if mask.sum() > mask.size * threshold:
            return dask.array.ones(dst_shape, name=False)

    return dask.array.from_array(mask.reshape(dst_shape), name=False)
