"""
Land Cover Level3 classification
"""

from typing import Tuple
import xarray as xr
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

        l34_dss = xx.classes_l3_l4
        urban_dss = xx.urban_classes
        cultivated_dss = xx.cultivated_class

        # Cultivated pipeline applies a mask which feeds only terrestrial veg (110) to the model
        # Just exclude no data (255) and apply the cultivated results
        cultivated_mask = cultivated_dss != int(NODATA)
        l34_cultivated_masked = xr.where(cultivated_mask, cultivated_dss, l34_dss)

        # Urban is classified on l3/4 surface output (210)
        urban_mask = l34_dss == 210
        l34_urban_cultivated_masked = xr.where(
            urban_mask, urban_dss, l34_cultivated_masked
        )

        attrs = xx.attrs.copy()
        attrs["nodata"] = NODATA
        l34_urban_cultivated_masked = l34_urban_cultivated_masked.squeeze(dim=["spec"])
        dims = l34_urban_cultivated_masked.dims

        data_vars = {
            "level3_class": xr.DataArray(
                l34_urban_cultivated_masked.data, dims=dims, attrs=attrs
            )
        }

        coords = dict((dim, xx.coords[dim]) for dim in dims)
        level3 = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        return level3


register("lccs_level3", StatsLccsLevel3)
