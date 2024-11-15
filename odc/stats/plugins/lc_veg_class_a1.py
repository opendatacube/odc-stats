"""
Plugin of Module A1 in LandCover PipeLine
"""

from typing import Optional, Dict

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
        output_classes: Dict,
        dem_threshold: Optional[int] = None,
        mudflat_threshold: Optional[int] = None,
        saltpan_threshold: Optional[int] = None,
        water_threshold: Optional[float] = None,
        veg_threshold: Optional[int] = None,
        water_seasonality_threshold: Optional[float] = None,
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
        self.water_seasonality_threshold = (
            water_seasonality_threshold if water_seasonality_threshold else 0.25
        )
        self.output_classes = output_classes

    def fuser(self, xx):
        return xx

    def l3_class(self, xx: xr.Dataset):
        si5 = expr_eval(
            "where(a>nodata, a*b/c, _nan)",
            {"a": xx.nbart_blue.data, "b": xx.nbart_red.data, "c": xx.nbart_green.data},
            name="caculate_si5",
            dtype="float32",
            **{"_nan": np.nan, "nodata": xx.nbart_blue.attrs["nodata"]},
        )

        # water  (water_freq >= 0.2)

        l3_mask = expr_eval(
            "where((a>=wt), m, 0)",
            {"a": xx["frequency"].data},
            name="mark_water",
            dtype="uint8",
            **{
                "wt": self.water_threshold,
                "m": self.output_classes["water"],
            },
        )

        # surface: (si5 > 1000 & dem <=6 ) | (si5 > 1500) | (veg_freq < 2) & !water
        # rest:  aquatic/terretrial veg
        l3_mask = expr_eval(
            "where(((a>mt)&(b<=dt)|(a>st)|(d<vt))&(c<=0), m, c)",
            {
                "a": si5,
                "b": xx.dem_h.data,
                "d": xx.veg_frequency.data,
                "c": l3_mask,
            },
            name="mark_surface",
            dtype="uint8",
            **{
                "mt": self.mudflat_threshold,
                "dt": self.dem_threshold,
                "st": self.saltpan_threshold,
                "vt": self.veg_threshold,
                "m": self.output_classes["surface"],
            },
        )

        # if its mangrove or coast region
        for b in self.optional_bands:
            if b in xx.data_vars:
                if b == "elevation":
                    # intertidal: water | surface & elevation
                    # aquatic_veg: veg & elevation
                    data = expr_eval(
                        "where((a==a), 1, 0)",
                        {
                            "a": xx[b].data,
                        },
                        name="mark_intertidal",
                        dtype="bool",
                    )

                    l3_mask = expr_eval(
                        "where(a&(b>0), m, b)",
                        {"a": data, "b": l3_mask},
                        name="intertidal_water",
                        dtype="uint8",
                        **{"m": self.output_classes["intertidal"]},
                    )

                    l3_mask = expr_eval(
                        "where(a&(b<=0), m, b)",
                        {"a": data, "b": l3_mask},
                        name="intertidal_veg",
                        dtype="uint8",
                        **{"m": self.output_classes["aquatic_veg_herb"]},
                    )
                elif b == "canopy_cover_class":
                    # aquatic_veg: (mangroves > 0) & (mangroves != nodata)
                    # mangroves.nodata = 255 or nan
                    l3_mask = expr_eval(
                        "where((a>0)&((a<nodata)|(nodata!=nodata)), m, b)",
                        {"a": xx[b].data, "b": l3_mask},
                        name="mark_mangroves",
                        dtype="uint8",
                        **{
                            "nodata": xx[b].attrs["nodata"],
                            "m": self.output_classes["aquatic_veg_wood"],
                        },
                    )

        # all unmarked values (0) is terretrial veg

        l3_mask = expr_eval(
            "where(a<=0, m, a)",
            {"a": l3_mask},
            name="mark_veg",
            dtype="uint8",
            **{"m": self.output_classes["terrestrial_veg"]},
        )

        # mark nodata if any source is nodata
        # issues:
        # - nodata information from non-indexed datasets missing

        # Mask nans with NODATA
        l3_mask = expr_eval(
            "where((a!=a), nodata, e)",
            {
                "a": si5,
                "e": l3_mask,
            },
            name="mark_nodata",
            dtype="uint8",
            **{"nodata": NODATA},
        )

        # Now add the water frequency
        # Divide water frequency into following classes:
        # 0 --> 0
        # (0,0.25] --> 1
        # (0.25,1] --> 2

        water_seasonality = expr_eval(
            "where((a > 0) & (a <= wt), 1, a)",
            {"a": xx["frequency"].data},
            name="mark_wo_fq",
            dtype="float32",
            **{"wt": self.water_seasonality_threshold},
        )

        water_seasonality = expr_eval(
            "where((a > wt) & (a <= 1), 2, b)",
            {"a": xx["frequency"].data, "b": water_seasonality},
            name="mark_wo_fq",
            dtype="float32",
            **{"wt": self.water_seasonality_threshold},
        )

        water_seasonality = expr_eval(
            "where((a != a), nodata, a)",
            {
                "a": water_seasonality,
            },
            name="mark_nodata",
            dtype="uint8",
            **{"nodata": NODATA},
        )

        return l3_mask, water_seasonality

    def reduce(self, xx: xr.Dataset) -> xr.Dataset:
        l3_mask, water_seasonality = self.l3_class(xx)
        attrs = xx.attrs.copy()
        attrs["nodata"] = int(NODATA)
        data_vars = {
            k: xr.DataArray(v, dims=xx["veg_frequency"].dims[1:], attrs=attrs)
            for k, v in zip(
                self.measurements, [l3_mask.squeeze(0), water_seasonality.squeeze(0)]
            )
        }
        coords = dict((dim, xx.coords[dim]) for dim in xx["veg_frequency"].dims[1:])
        return xr.Dataset(data_vars=data_vars, coords=coords, attrs=xx.attrs)


register("veg_class_l1", StatsVegClassL1)
register("dem_in_stats", StatsDem)
