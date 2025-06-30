"""
Long-term percentiles of S2Cloudless probabilities.

Useful for locating regions persistently misclassified as
cloud by S2Cloudless, which is known to have a high false 
positive rate.

"""

from functools import partial
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Mapping

import datacube
import numpy as np
import pandas as pd
import xarray as xr
from datacube.model import Dataset
from odc.geo.xr import assign_crs
from odc.geo.geobox import GeoBox
from odc.algo.io import load_with_native_transform
from odc.algo._percentile import xr_quantile_bands
from odc.stats.plugins._registry import register, StatsPluginInterface
from odc.algo._masking import (
    erase_bad,
    enum_to_bool
)

class S2Cloudless_percentiles(StatsPluginInterface):
    NAME = "S2Cloudless_percentiles"
    SHORT_NAME = NAME
    VERSION = "1.0.0"
    PRODUCT_FAMILY = "percentiles"

    def __init__(
        self,
        resampling: str = "cubic",
        bands: Sequence[str] = ["oa_s2cloudless_prob"],
        output_bands: Sequence[str] = ['oa_s2cloudless_prob_pc_5', 'oa_s2cloudless_prob_pc_10','oa_s2cloudless_prob_pc_25'],
        mask_band: str = "oa_s2cloudless_mask",
        chunks: Mapping[str, int] = {"y": 512, "x": 512},
        group_by: str = "solar_day",
        nodata_classes: Sequence[str] = ["nodata"], 
        output_dtype: str = "float32",
        **kwargs,
    ):
        
        self.resampling=resampling
        self.bands = bands
        self.output_bands = output_bands
        self.mask_band = mask_band
        self.chunks = chunks
        self.group_by = group_by
        self.resampling = resampling
        self.nodata_classes= nodata_classes
        self.output_dtype = np.dtype(output_dtype)
        self.output_nodata = np.nan

        super().__init__(
            input_bands=tuple(bands),
            resampling=resampling,
            chunks=chunks,
            **kwargs
        )

    @property
    def measurements(self) -> Tuple[str, ...]:
        return (self.output_bands)

    def native_transform(self, xx: xr.Dataset) -> xr.Dataset:
        """
        erases nodata
        """

        # step 1-----------------
        if self.mask_band not in xx.data_vars:
            return xx

        # Erase Data Pixels for which mask == nodata
        mask = xx[self.mask_band]
        bad = enum_to_bool(mask, self.nodata_classes)

        #drop mask band
        xx = xx.drop_vars([self.mask_band])

        # apply the masks
        xx = erase_bad(xx, bad)

        return xx
         

    def reduce(self, xx: xr.Dataset) -> xr.Dataset:
        """
        Calculate the percentiles of long-term cloud probabilities

        """

        #  Compute the percentiles of long-term cloud probabilities.
        yy = xr_quantile_bands(xx, [0.05, 0.10, 0.25], nodata=np.nan)

        return yy

register("s2_gm_tools.S2Cloudless_percentiles", S2Cloudless_percentiles)
