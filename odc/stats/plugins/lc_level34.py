"""
Plugin of Module A3 in LandCover PipeLine
"""

from typing import Optional, List

import numpy as np
import xarray as xr

from ._registry import StatsPluginInterface, register

from .l34_utils import (
    l4_water_persistence,
    l4_veg_cover,
    lc_level3,
    l4_cultivated,
    l4_natural_veg,
    l4_natural_aquatic,
    l4_surface,
    l4_bare_gradation,
    l4_water,
)


NODATA = 255


class StatsLccsLevel4(StatsPluginInterface):
    NAME = "ga_ls_lccs_Level34"
    SHORT_NAME = NAME
    VERSION = "0.0.1"
    PRODUCT_FAMILY = "lccs"

    def __init__(
        self,
        veg_threshold: Optional[List] = None,
        bare_threshold: Optional[List] = None,
        watper_threshold: Optional[List] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.veg_threshold = (
            veg_threshold if veg_threshold is not None else [1, 4, 15, 40, 65, 100]
        )
        self.bare_threshold = bare_threshold if bare_threshold is not None else [20, 60]
        self.watper_threshold = (
            watper_threshold if watper_threshold is not None else [1, 4, 7, 10]
        )

    def fuser(self, xx):
        return xx

    def reduce(self, xx: xr.Dataset) -> xr.Dataset:

        # Water persistence
        water_persistence = l4_water_persistence.water_persistence(
            xx, self.watper_threshold
        )

        # #TODO WATER (99-104)
        l4 = l4_water.water_classification(xx, water_persistence)

        # Generate Level3 classes
        level3 = lc_level3.lc_level3(xx)

        # Vegetation cover
        veg_cover = l4_veg_cover.canopyco_veg_con(xx, self.veg_threshold)

        # Apply cultivated Level-4 classes (1-18)
        l4 = l4_cultivated.lc_l4_cultivated(l4, level3, xx.woody, veg_cover)

        # Apply terrestrial vegetation classes [19-36]
        l4 = l4_natural_veg.lc_l4_natural_veg(l4, level3, xx.woody, veg_cover)

        # Bare gradation
        bare_gradation = l4_bare_gradation.bare_gradation(
            xx, self.bare_threshold, veg_cover
        )

        l4 = l4_natural_aquatic.natural_auquatic_veg(l4, veg_cover, xx.water_season)

        level4 = l4_surface.lc_l4_surface(l4, level3, bare_gradation)

        level3 = level3.astype(np.uint8)
        level4 = level4.astype(np.uint8)

        attrs = xx.attrs.copy()
        attrs["nodata"] = NODATA
        dims = xx.level_3_4.dims[1:]
        data_vars = {
            k: xr.DataArray(v, dims=dims, attrs=attrs)
            for k, v in zip(self.measurements, [level3.squeeze(), level4.squeeze()])
        }

        coords = dict((dim, xx.coords[dim]) for dim in dims)
        leve34 = xr.Dataset(data_vars=data_vars, coords=coords, attrs=xx.attrs)
        return leve34


register("lc_l3_l4", StatsLccsLevel4)
