"""
Plugin of Module A1 in LandCover PipeLine
"""

from typing import Tuple, Dict, Iterable, Optional

import numpy as np
import xarray as xr
from odc.stats._algebra import expr_eval

from ._registry import StatsPluginInterface, register

NODATA = 255


class StatsDem(StatsPluginInterface):
    NAME = "ga_ls_lccs_dem"
    SHORT_NAME = NAME
    VERSION = "0.0.1"
    PRODUCT_FAMILY = "lccs"

    def __init__(
        self,
        ue_threshold: Optional[int] = None,
        cloud_filters: Dict[str, Iterable[Tuple[str, int]]] = None,
        **kwargs,
    ):
        super().__init__(
            input_bands=["dem_h"], chunks={"latitude": -1, "longitude": -1}, **kwargs
        )

    @property
    def measurements(self) -> Tuple[str, ...]:
        _measurements = ["dem_h"]
        return _measurements

    def native_transform(self, xx):
        # reproject cannot work with nodata being int for float
        # hence convert to nan

        nodata = float(xx["dem_h"].attrs["nodata"])
        data = expr_eval(
            "where(a>=b, a, _nan)",
            {"a": xx.dem_h.data},
            name="convert_nodata",
            dtype="float32",
            **{"_nan": np.nan, "b": nodata},
        )
        xx["dem_h"].data = data
        xx["dem_h"].attrs["nodata"] = np.nan
        return xx

    def fuser(self, xx):
        return xx

    def reduce(self, xx: xr.Dataset) -> xr.Dataset:
        return xx


class StatsVegClassL1(StatsPluginInterface):
    NAME = "ga_ls_lccs_veg_class_a1"
    SHORT_NAME = NAME
    VERSION = "0.0.1"
    PRODUCT_FAMILY = "lccs"

    def __init__(
        self,
        ue_threshold: Optional[int] = None,
        cloud_filters: Dict[str, Iterable[Tuple[str, int]]] = None,
        **kwargs,
    ):
        super().__init__(
            input_bands=[
                "nbart_red",
                "nbart_green",
                "nbart_blue",
                "dem_h",
            ],
            **kwargs,
        )

        self.ue_threshold = ue_threshold if ue_threshold is not None else 30
        self.cloud_filters = cloud_filters if cloud_filters is not None else {}

    @property
    def measurements(self) -> Tuple[str, ...]:
        _measurements = ["veg_class_l1"]
        return _measurements

    def native_transform(self, xx):

        print(xx)
        return xx

    def fuser(self, xx):
        return xx

    def reduce(self, xx: xr.Dataset) -> xr.Dataset:
        return xx


register("veg_class_l1", StatsVegClassL1)
register("dem_in_stats", StatsDem)
