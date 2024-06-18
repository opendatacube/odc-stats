"""
Plugin of Module A1 in LandCover PipeLine
"""

from typing import Tuple, Optional

import numpy as np
import xarray as xr
from odc.stats._algebra import expr_eval

from ._registry import StatsPluginInterface, register

NODATA = 255


class StatsDem(StatsPluginInterface):
    NAME = "ga_ls_lccs_(ni)dem"
    SHORT_NAME = NAME
    VERSION = "0.0.1"
    PRODUCT_FAMILY = "lccs"

    def __init__(
        self,
        measurements: Optional[Tuple[str, ...]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._measurements = (
            measurements if measurements is not None else self.input_bands
        )

    @property
    def measurements(self) -> Tuple[str, ...]:
        return self._measurements

    def native_transform(self, xx):
        # reproject cannot work with nodata being int for float
        # hence convert to nan

        for var in self.input_bands:
            nodata = float(xx[var].attrs["nodata"])
            data = expr_eval(
                "where(a>b, a, _nan)",
                {"a": xx[var].data},
                name="convert_nodata",
                dtype="float32",
                **{"_nan": np.nan, "b": nodata},
            )
            xx[var].data = data
            xx[var].attrs["nodata"] = np.nan

        return xx

    def fuser(self, xx):
        return xx

    def reduce(self, xx: xr.Dataset) -> xr.Dataset:
        if self.measurements != self.input_bands:
            xx = xx.rename(dict(zip(self.input_bands, self.measurements)))
        return xx


class StatsVegClassL1(StatsPluginInterface):
    NAME = "ga_ls_lccs_veg_class_a1"
    SHORT_NAME = NAME
    VERSION = "0.0.1"
    PRODUCT_FAMILY = "lccs"

    def __init__(
        self,
        dem_threshold: Optional[int] = None,
        mudflat_threshold: Optional[int] = None,
        saltpan_threshold: Optional[int] = None,
        water_threshold: Optional[float] = None,
        veg_threshold: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dem_threshold = dem_threshold if dem_threshold is not None else 6
        self.mudflat_threshold = (
            mudflat_threshold if mudflat_threshold is not None else 1000
        )
        self.saltpan_threshold = (
            saltpan_threshold if saltpan_threshold is not None else 1500
        )
        self.water_threshold = water_threshold if water_threshold is not None else 0.2
        self.veg_threshold = veg_threshold if veg_threshold is not None else 2

    @property
    def measurements(self) -> Tuple[str, ...]:
        _measurements = ["veg_class_l1"]
        return _measurements

    def native_transform(self, xx):
        return xx

    def fuser(self, xx):
        return xx

    def veg_class(self, xx: xr.Dataset):
        data = expr_eval(
            "where(a>nodata, a*b/c, _nan)",
            {"a": xx.nbart_blue.data, "b": xx.nbart_red.data, "c": xx.nbart_green.data},
            name="caculate_si5",
            dtype="float32",
            **{"_nan": np.nan, "nodata": xx.nbart_blue.attrs["nodata"]},
        )

        # non veg (0): (si5 > 1000 & dem <=6 ) | (si5 > 1500) | (water_freq > 0.2) | (veg_freq < 2)
        veg_mask = expr_eval(
            "where((a>mt)&(b<=dt)|(a>st)|(c>wt)|(d<vt), 0, 1)",
            {
                "a": data,
                "b": xx.dem_h.data,
                "c": xx["frequency"].data,
                "d": xx.veg_frequency.data,
            },
            name="caculate_veg",
            dtype="bool",
            **{
                "mt": self.mudflat_threshold,
                "dt": self.dem_threshold,
                "st": self.saltpan_threshold,
                "wt": self.water_threshold,
                "vt": self.veg_threshold,
            },
        )

        for b in self.optional_bands:
            if (b in xx.data_vars) & (b == "mangroves_cover_extent"):
                # veg: (mangroves > 0) & (mangroves != nodata)
                veg_mask = expr_eval(
                    "where((a>0)&(a!=nodata), 1|b, b)",
                    {"a": xx[b].data, "b": veg_mask},
                    name="calculate_mangroves",
                    dtype="bool",
                    **{"nodata": NODATA},
                )

        # mark nodata if any source is nodata
        # veg 1 := 100
        # no veg 2:= 200
        # issues:
        # - hardcoded output is not ideal
        # - nodata information from non-indexed datasets missing

        veg_mask = expr_eval(
            "where((a!=a)|(b==nodata)|(c!=c)|(d!=d), nodata, 200-e*100)",
            {
                "a": data,
                "b": xx.veg_frequency.data,
                "c": xx["frequency"].data,
                "d": xx.dem_h.data,
                "e": veg_mask,
            },
            name="mark_nodata",
            dtype="uint8",
            **{"nodata": NODATA},
        )
        return veg_mask

    def reduce(self, xx: xr.Dataset) -> xr.Dataset:
        veg_mask = self.veg_class(xx)

        attrs = xx.attrs.copy()
        attrs["nodata"] = int(NODATA)
        data_vars = {
            "veg_class_l1": xr.DataArray(
                veg_mask[0], dims=xx["veg_frequency"].dims[1:], attrs=attrs
            )
        }
        coords = dict((dim, xx.coords[dim]) for dim in xx["veg_frequency"].dims[1:])
        return xr.Dataset(data_vars=data_vars, coords=coords, attrs=xx.attrs)


register("veg_class_l1", StatsVegClassL1)
register("dem_in_stats", StatsDem)
