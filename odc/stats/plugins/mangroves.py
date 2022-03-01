"""
Mangroves canopy cover classes
"""
from functools import partial
from itertools import product
from typing import Optional, Tuple

import numpy as np
import xarray as xr
import dask
import os
from odc.algo import keep_good_only, erase_bad
from odc.algo._masking import _fuse_mean_np, _fuse_or_np, _or_fuser, _xr_fuse
from odc.algo._percentile import xr_quantile_bands
from osgeo import gdal, ogr, osr

from ._registry import StatsPluginInterface, register

NODATA = 255


class Mangroves(StatsPluginInterface):

    NAME = "mangroves"
    SHORT_NAME = NAME
    VERSION = "0.0.1"
    PRODUCT_FAMILY = "mangroves"

    def __init__(
        self,
        pv_thresholds=[14, 38, 60],
        tcw_threshold=-1850,
        **kwargs,
    ):
        self.mangroves_extent = kwargs.pop('mangroves_extent', None)
        self.pv_thresholds = pv_thresholds
        self.tcw_threshold = tcw_threshold
        super().__init__(input_bands=["pv_pc_10", "qa", "wet_pc_10"], **kwargs)

    @property
    def measurements(self) -> Tuple[str, ...]:
        _measurements = [
            "canopy_cover_class"
        ]
        return _measurements
    
    def rasterize_mangroves_extent(self, shape_file, array_shape, orig_coords, resolution=(30, -30)):
        source_ds = ogr.Open(shape_file)
        source_layer = source_ds.GetLayer()

        yt, xt = array_shape[1:]
        xres, yres = resolution 
        no_data = 0
        xcoord, ycoord = orig_coords

        geotransform = (xcoord - (xres * 0.5), xres, 0, ycoord - (yres * 0.5), 0, yres)

        target_ds = gdal.GetDriverByName("MEM").Create("", xt, yt, gdal.GDT_Byte)
        target_ds.SetGeoTransform(geotransform)
        albers = osr.SpatialReference()
        albers.ImportFromEPSG(3577)
        target_ds.SetProjection(albers.ExportToWkt())
        band = target_ds.GetRasterBand(1)
        band.SetNoDataValue(no_data)

        gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[1])
        return dask.array.from_array(band.ReadAsArray().reshape(array_shape), name=False)

    def fuser(self, xx):
        """
            no fuse required for mangroves since group by none
            return loaded data
        """
        return xx

    def reduce(self, xx: xr.Dataset) -> xr.Dataset:
        """
            mangroves computation here
            it is not a 'reduce' though
        """
        if self.mangroves_extent:
            if not os.path.exists(self.mangroves_extent):
                raise FileNotFoundError(f"{self.mangroves_extent} not found")
            extent_mask = self.rasterize_mangroves_extent(self.mangroves_extent, xx.pv_pc_10.shape,
                                                     (xx.coords["x"].min(), xx.coords["y"].max()))
        else:
            extent_mask = dask.array.ones(xx.pv_pc_10.shape)
        good_data = (extent_mask == 1)
        good_data &= (xx.wet_pc_10 > self.tcw_threshold)
        good_data &= (xx.pv_pc_10 > self.pv_thresholds[0]) & (xx.qa == 2) | (xx.qa == 1)
       
        notsure_mask = (xx.qa == 1)
    
        cover_type = xx.pv_pc_10.copy(True)
        cover_type.data = dask.array.zeros_like(cover_type.data)
        for s_t in self.pv_thresholds:
            cover_type.data += (xx.pv_pc_10.data > s_t).astype(np.uint8)

        cover_type = erase_bad(cover_type, notsure_mask, nodata=0)
        cover_type = keep_good_only(cover_type, good_data, nodata=NODATA)
        cover_type.attrs['nodata'] = NODATA

        cover_type = cover_type.to_dataset(name="canopy_cover_class")
        # don't want the dimension spec from input but keep the info in case
        if "spec" in cover_type.dims:
            cover_type = cover_type.squeeze(dim=["spec"])
        return cover_type 


register("mangroves", Mangroves)
