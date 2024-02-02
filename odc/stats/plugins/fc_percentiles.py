"""
Fractional Cover Percentiles
"""

from functools import partial
from itertools import product
from typing import Tuple, Dict, Iterable, Optional

import numpy as np
import xarray as xr
from odc.algo import keep_good_only
from odc.algo._masking import _fuse_mean_np, _fuse_or_np, _xr_fuse, mask_cleanup
from odc.algo._percentile import xr_quantile_bands

from ._registry import StatsPluginInterface, register

NODATA = 255


class StatsFCP(StatsPluginInterface):
    NAME = "ga_fc_percentiles"
    SHORT_NAME = NAME
    VERSION = "0.0.3"
    PRODUCT_FAMILY = "fc_percentiles"

    BAD_BITS_MASK = dict(cloud=(1 << 6), cloud_shadow=(1 << 5), terrain_shadow=(1 << 3))

    def __init__(
        self,
        max_sum_limit: Optional[int] = None,
        clip_range: Optional[Tuple] = None,
        ue_threshold: Optional[int] = None,
        count_valid: Optional[bool] = False,
        cloud_filters: Dict[str, Iterable[Tuple[str, int]]] = None,
        **kwargs,
    ):
        super().__init__(input_bands=["water", "pv", "bs", "npv", "ue"], **kwargs)

        self.max_sum_limit = max_sum_limit
        self.clip_range = clip_range
        self.ue_threshold = ue_threshold
        self.count_valid = count_valid
        self.cloud_filters = cloud_filters if cloud_filters is not None else {}

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

        # not mask against bit 4: terrain high slope
        # Pick out the dry and wet pixels
        valid = (xx["water"] & ~(1 << 4)) == 0
        wet = (xx["water"] & ~(1 << 4)) == 128

        # dilate both 'valid' and 'water'
        for key, val in self.BAD_BITS_MASK.items():
            if self.cloud_filters.get(key) is not None:
                raw_mask = (xx["water"] & val) > 0
                raw_mask = mask_cleanup(
                    raw_mask, mask_filters=self.cloud_filters.get(key)
                )
                valid &= ~raw_mask
                wet &= ~raw_mask

        xx = xx.drop_vars(["water"])

        # Pick out the pixels that have an unmixing error of less than the threshold
        if self.ue_threshold is not None:
            # No QA
            valid &= xx.ue < self.ue_threshold
        xx = xx.drop_vars(["ue"])

        # If there's a sum limit or clip range, implement these
        if self.max_sum_limit is not None or self.clip_range is not None:
            sum_bands = 0
            for band in xx.data_vars.keys():
                attributes = xx[band].attrs
                mask = xx[band] == NODATA
                band_data = keep_good_only(xx[band], ~mask, nodata=0)

                if self.max_sum_limit is not None:
                    sum_bands = sum_bands + band_data

                if self.clip_range is not None:
                    # No QA
                    clipped = np.clip(xx[band], self.clip_range[0], self.clip_range[1])
                    # Set masked values back to NODATA
                    xx[band] = clipped.where(~mask, NODATA)
                    xx[band].attrs = attributes

            if self.max_sum_limit is not None:
                valid &= sum_bands < self.max_sum_limit

        xx = keep_good_only(xx, valid, nodata=NODATA)
        xx["wet"] = wet
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
        # wet and valid(dry) are mutually exclusive, here we keep good data
        wet &= ~valid
        is_ever_wet = (wet.sum(axis=0, dtype="int16") > 0).astype("uint8")

        all_bands_valid = True
        for band in yy.data_vars.keys():
            all_bands_valid &= yy[band] != NODATA

        all_bands_valid = all_bands_valid.astype("uint8")

        yy["qa"] = 1 + all_bands_valid - is_ever_wet * (1 - all_bands_valid)

        if self.count_valid:
            yy["count_valid"] = valid.sum(axis=0, dtype="int16")

        return yy


register("fc-percentiles", StatsFCP)
