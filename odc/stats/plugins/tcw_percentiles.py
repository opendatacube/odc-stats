"""
Fractional Cover Percentiles
"""
from functools import partial
from typing import Optional, Sequence, Tuple, Dict
import xarray as xr
import numpy as np
from odc.algo import keep_good_only
from odc.algo._percentile import xr_quantile_bands
from odc.algo._masking import _xr_fuse, _fuse_mean_np, enum_to_bool
from ._registry import StatsPluginInterface, register

NODATA = -9999 # output NODATA


class StatsTCWPC(StatsPluginInterface):

    NAME = "ga_tcw_percentiles"
    SHORT_NAME = NAME
    VERSION = "0.0.1"
    PRODUCT_FAMILY = "twc_percentiles"

    def __init__(
        self,
        coefficients: Dict[str, Dict[str, float]] = {
            "wet": {"blue": 0.0315, "green": 0.2021, "red": 0.3102, "nir": 0.1594,
                    "swir1": -0.6806, "swir2": -0.6109},
            "bright": {"blue": 0.2043, "green": 0.4158, "red": 0.5524, "nir": 0.5741,
                       "swir1": 0.3124, "swir2": 0.2303},
            "green": {"blue": -0.1603, "green": -0.2819, "red": -0.4934, "nir": 0.7940,
                      "swir1": -0.0002, "swir2": -0.1446},
            },
        input_bands: Sequence[str] = ["blue", "green", "red", "nir", "swir1", "swir2", "fmask", "nbart_contiguity"],
        output_bands: Sequence[str] = ["wet"],
        **kwargs
    ):
        super().__init__(input_bands=input_bands, **kwargs)
        self.coefficients = coefficients

    @property
    def measurements(self) -> Tuple[str, ...]:
        _measurments = []
        for band in output_bands:
            _measurments += [f"{band}_pc_10", f"{band}_pc_50", f"{band}_pc_90"]
        return _measurments

    def native_transform(self, xx):
        """
        Loads data in its native projection.
        """
        bad = enum_to_bool(xx["fmask"], ("nodata", "cloud", "shadow")) # a pixel is bad if any of the cloud, shadow, or no-data value
        bad |= xx["nbart_contiguity"] == 0 # or the nbart contiguity bit is 0
        xx = xx.drop_vars(["fmask", "nbart_contiguity"])
        
        for band in xx.data_vars.keys():
            bad = bad | (xx[band] == -999)

        for m in output_bands:
            xx[m] = sum(coeff * xx[band] for band, coeff in self.coefficients[m].items()).astype(np.int16)
            xx[m].attrs = xx.blue.attrs
            xx[m].attrs["nodata"] = NODATA

        xx = xx.drop_vars(input_bands)
        xx = keep_good_only(xx, ~bad, nodata=NODATA)
        return xx

    @staticmethod
    def fuser(xx):
        xx = _xr_fuse(xx, partial(_fuse_mean_np, nodata=NODATA), "")

        return xx
    
    @staticmethod
    def reduce(xx: xr.Dataset) -> xr.Dataset:
        yy = xr_quantile_bands(xx, [0.1, 0.5, 0.9], nodata=NODATA)
        return yy


register("tcw-percentiles", StatsTCWPC)
