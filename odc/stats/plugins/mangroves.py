"""
Mangroves canopy cover classes
"""
from typing import Tuple

import numpy as np
import xarray as xr
import dask
import os
from odc.algo import keep_good_only, erase_bad
import fiona
from rasterio import features

from ._registry import StatsPluginInterface, register

NODATA = 255


class Mangroves(StatsPluginInterface):

    NAME = "mangroves"
    SHORT_NAME = NAME
    VERSION = "0.0.2"
    PRODUCT_FAMILY = "mangroves"

    def __init__(
        self,
        pv_thresholds=[14, 38, 60],
        tcw_threshold=-1850,
        **kwargs,
    ):
        self.mangroves_extent = kwargs.pop("mangroves_extent", None)
        self.pv_thresholds = pv_thresholds
        self.tcw_threshold = tcw_threshold
        super().__init__(input_bands=["pv_pc_10", "qa", "wet_pc_10"], **kwargs)

    @property
    def measurements(self) -> Tuple[str, ...]:
        _measurements = ["canopy_cover_class"]
        return _measurements

    def rasterize_mangroves_extent(self, shape_file, transform, dst_shape):
        with fiona.open(shape_file) as source_ds:
            geoms = [s["geometry"] for s in source_ds]

        mangrove_extent = features.rasterize(
            geoms,
            transform=transform,
            out_shape=dst_shape[1:],
            all_touched=False,
            fill=0,
            default_value=1,
            dtype="uint8",
        )

        return dask.array.from_array(mangrove_extent.reshape(dst_shape), name=False)

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
            extent_mask = self.rasterize_mangroves_extent(
                self.mangroves_extent, xx.geobox.transform, xx.pv_pc_10.shape
            )
        else:
            extent_mask = dask.array.ones(xx.pv_pc_10.shape)
        good_data = extent_mask == 1
        good_data &= xx.wet_pc_10 > self.tcw_threshold
        good_data &= (xx.pv_pc_10 > self.pv_thresholds[0]) & (xx.qa == 2) | (xx.qa == 1)

        notsure_mask = xx.qa == 1

        cover_type = xx.pv_pc_10.copy(True)
        cover_type.data = dask.array.zeros_like(cover_type.data)
        for s_t in self.pv_thresholds:
            cover_type.data += (xx.pv_pc_10.data > s_t).astype(np.uint8)

        cover_type = erase_bad(cover_type, notsure_mask, nodata=0)
        cover_type = keep_good_only(cover_type, good_data, nodata=NODATA)
        cover_type.attrs["nodata"] = NODATA

        cover_type = cover_type.to_dataset(name="canopy_cover_class")
        # don't want the dimension spec from input but keep the info in case
        if "spec" in cover_type.dims:
            cover_type = cover_type.squeeze(dim=["spec"])
        return cover_type


register("mangroves", Mangroves)
