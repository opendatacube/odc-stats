"""
Land Cover Level3 classification
"""

from typing import Tuple
import xarray as xr
from odc.stats._algebra import expr_eval
from ._registry import StatsPluginInterface, register

NODATA = 255


class StatsLccsLevel3(StatsPluginInterface):
    NAME = "ga_ls_lccs_level3"
    SHORT_NAME = NAME
    VERSION = "0.0.1"
    PRODUCT_FAMILY = "lccs"

    @property
    def measurements(self) -> Tuple[str, ...]:
        _measurements = ["level3_class"]
        return _measurements

    def reduce(self, xx: xr.Dataset) -> xr.Dataset:

        # Cultivated pipeline applies a mask which feeds only terrestrial veg (110) to the model
        # Just exclude no data (255) and apply the cultivated results
        res = expr_eval(
            "where(a<nodata, a, b)",
            {"a": xx.cultivated_class.data, "b": xx.classes_l3_l4.data},
            name="mask_cultivated",
            dtype="float32",
            **{"nodata": NODATA},
        )

        # Mask urban results with bare sfc (210)

        res = expr_eval(
            "where(a==_u, b, a)",
            {
                "a": res,
                "b": xx.urban_classes.data,
            },
            name="mark_urban",
            dtype="uint8",
            **{"_u": 210},
        )

        attrs = xx.attrs.copy()
        attrs["nodata"] = NODATA
        dims = xx.classes_l3_l4.dims[1:]

        data_vars = {
            "level3_class": xr.DataArray(res.squeeze(), dims=dims, attrs=attrs)
        }

        coords = dict((dim, xx.coords[dim]) for dim in dims)
        level3 = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        return level3


register("lccs_level3", StatsLccsLevel3)
