"""
Fractional Cover Percentiles
"""
from functools import partial
from itertools import product
from typing import Optional, Tuple

import numpy as np
import xarray as xr
from odc.algo import keep_good_only
from odc.algo._masking import _fuse_mean_np, _fuse_or_np, _or_fuser, _xr_fuse
from odc.algo._percentile import xr_quantile_bands

from ._registry import StatsPluginInterface, register

NODATA = 255


class StatsFCP(StatsPluginInterface):

    NAME = "ga_fc_percentiles"
    SHORT_NAME = NAME
    VERSION = "0.0.2"
    PRODUCT_FAMILY = "fc_percentiles"

    def __init__(
        self,
        max_sum_limit: Optional[int] = None,
        clip_range: Optional[Tuple] = None,
        ue_threshold: Optional[int] = None,
        count_valid: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(input_bands=["water", "pv", "bs", "npv", "ue"], **kwargs)

        self.max_sum_limit = max_sum_limit
        self.clip_range = clip_range
        self.ue_threshold = ue_threshold
        self.count_valid = count_valid

    @property
    def measurements(self) -> Tuple[str, ...]:
        _measurements = [
            f"{b}_pc_{p}" for b, p in product(["pv", "bs", "npv"], ["10", "50", "90"])
        ]
        _measurements.append("qa")
        if self.count_valid:
            _measurements.append("count_valid")
        return _measurements

    def native_transform(self, xx):
        """
        Loads data in its native projection. It performs the following:

        1. Load all fc and WOfS bands
        2. Set the high terrain slope flag to 0
        3. Set all pixels that are not clear and dry to NODATA
        4. Calculate the clear wet pixels
        5. Drop the WOfS band
        """

        water = xx.water & 0b1110_1111
        xx = xx.drop_vars(["water"])

        # Pick out the dry pixels
        dry = water == 0

        # Incrementally add to the valid band
        valid = dry

        # Pick out the pixels that have an unmixing error of less than the threshold
        if self.ue_threshold is not None:
            # No QA
            unmixing_error_lt_30 = xx.ue < self.ue_threshold
            valid = valid & unmixing_error_lt_30
        xx = xx.drop_vars(["ue"])

        # If there's a sum limit or clip range, implement these
        if self.max_sum_limit is not None or self.clip_range is not None:
            sum_bands = None
            for band in xx.data_vars.keys():
                attributes = xx[band].attrs
                mask = xx[band] == NODATA
                band_data = keep_good_only(xx[band], ~mask, nodata=0)

                if self.max_sum_limit is not None:
                    if sum_bands is None:
                        sum_bands = band_data
                    else:
                        sum_bands = sum_bands + band_data

                if self.clip_range is not None:
                    # No QA
                    limit_min, limit_max = self.clip_range
                    clipped = np.clip(xx[band], limit_min, limit_max)
                    # Set masked values back to NODATA
                    xx[band] = clipped.where(~mask, NODATA)
                    xx[band].attrs = attributes

            if self.max_sum_limit is not None:
                sum_lt_limit = sum_bands < self.max_sum_limit
                valid = valid & sum_lt_limit

        xx = keep_good_only(xx, valid, nodata=NODATA)

        xx["wet"] = water == 128
        xx["valid"] = valid

        return xx

    def fuser(self, xx):
        wet = xx["wet"]
        valid = xx["valid"]

        xx = _xr_fuse(
            xx.drop_vars(["wet", "valid"]), partial(_fuse_mean_np, nodata=NODATA), ""
        )

        xx["wet"] = _xr_fuse(wet, _fuse_or_np, wet.name)
        xx["valid"] = _xr_fuse(valid, _fuse_or_np, valid.name)

        return xx

    def reduce(self, xx: xr.Dataset) -> xr.Dataset:
        # (!all_bands_valid) & is_ever_wet => 0
        # (!all_bands_valid) & (!is_ever_wet) => 1
        # all_bands_valid => 2

        wet = xx["wet"]
        valid = xx["valid"]
        xx = xx.drop_vars(["wet", "valid"])

        yy = xr_quantile_bands(xx, [0.1, 0.5, 0.9], nodata=NODATA)
        is_ever_wet = _or_fuser(wet).squeeze(wet.dims[0], drop=True)

        band, *bands = yy.data_vars.keys()
        all_bands_valid = yy[band] != NODATA
        for band in bands:
            all_bands_valid &= yy[band] != NODATA

        all_bands_valid = all_bands_valid.astype(np.uint8)
        is_ever_wet = is_ever_wet.astype(np.uint8)
        yy["qa"] = 1 + all_bands_valid - is_ever_wet * (1 - all_bands_valid)

        if self.count_valid:
            yy["count_valid"] = valid.sum(axis=0, dtype="int16")

        return yy


register("fc-percentiles", StatsFCP)
