"""
Plugin of Module A0 in LandCover PipeLine
"""

from functools import partial
from typing import Tuple, Dict, Iterable, Optional

import numpy as np
import xarray as xr
import dask.array as da
from odc.algo import keep_good_only
from odc.algo._masking import (
    _fuse_mean_np,
    _xr_fuse,
    mask_cleanup,
    to_float,
    _nodata_fuser,
)
from odc.stats._algebra import expr_eval

from ._registry import StatsPluginInterface, register

NODATA = 255


class StatsVegCount(StatsPluginInterface):
    NAME = "ga_ls_lccs_fc_wo_a0"
    SHORT_NAME = NAME
    VERSION = "0.0.1"
    PRODUCT_FAMILY = "lccs"

    BAD_BITS_MASK = dict(cloud=(1 << 6), cloud_shadow=(1 << 5))

    def __init__(
        self,
        ue_threshold: Optional[int] = None,
        cloud_filters: Dict[str, Iterable[Tuple[str, int]]] = None,
        **kwargs,
    ):
        super().__init__(input_bands=["water", "pv", "bs", "npv", "ue"], **kwargs)

        self.ue_threshold = ue_threshold if ue_threshold is not None else 30
        self.cloud_filters = cloud_filters if cloud_filters is not None else {}

    @property
    def measurements(self) -> Tuple[str, ...]:
        _measurements = ["veg_frequency"]
        return _measurements

    def native_transform(self, xx):
        """
        Loads data in its native projection. It performs the following:

        1. Load all fc and WO bands
        3. Set all pixels that are not clear and dry to NODATA
        4. Calculate the clear wet pixels
        5. Drop the WOfS band
        """

        # clear and dry pixels not mask against bit 4: terrain high slope,
        # bit 3: terrain shadow, and
        # bit 2: low solar angle
        valid = (xx["water"] & ~((1 << 4) | (1 << 3) | (1 << 2))) == 0

        # clear and wet pixels not mask against bit 2: low solar angle
        wet = (xx["water"] & ~(1 << 2)) == 128

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

        # get valid wo pixels, both dry and wet
        data = expr_eval(
            "where(a|b, a, _nan)",
            ["a", "b"],
            wet.data,
            m_v_1=valid.data,
            name="get_valid_pixels",
            dtype="float32",
            **dict(_nan=np.nan),
        )

        # Pick out the fc pixels that have an unmixing error of less than the threshold
        valid &= xx["ue"] < self.ue_threshold
        xx = xx.drop_vars(["ue"])
        xx = keep_good_only(xx, valid, nodata=NODATA)
        xx = to_float(xx)

        xx["wet"] = xr.DataArray(data, dims=wet.dims, coords=wet.coords)

        return xx

    def fuser(self, xx):

        wet = xx["wet"]

        xx = _xr_fuse(xx.drop_vars(["wet"]), partial(_fuse_mean_np, nodata=np.nan), "")

        xx["wet"] = _nodata_fuser(wet, nodata=np.nan)

        return xx

    def reduce(self, xx: xr.Dataset) -> xr.Dataset:

        xx = xx.groupby("time.month").median(dim="spec", skipna=True)
        print(xx)

        # either pv or npv > bs: 1
        # otherwise 0
        data = expr_eval(
            "where((a>b)|(c>b), 1, 0)",
            ["a", "c", "b"],
            xx["pv"].data,
            xx["npv"].data,
            xx["bs"].data,
            name="get_veg",
            dtype="float32",
        )

        # mark nans
        data = expr_eval(
            "where(a!=a, _nan, b)",
            ["a", "b"],
            xx["pv"].data,
            data,
            name="get_veg",
            dtype="float32",
            **dict(_nan=np.nan),
        )

        # mark water freq >= 0.5 as 0
        data = expr_eval(
            "where(a>0, 0, b)",
            ["a", "b"],
            xx["wet"].data,
            data,
            name="get_veg",
            dtype="float32",
        )

        tmp = da.zeros(data.shape[1:], chunks=data.chunks[1:])
        max_count = da.zeros(data.shape[1:], chunks=data.chunks[1:])

        for t in data:
            # +1 if not nan
            tmp = expr_eval(
                "where(a!=a, b, a+b)",
                ["a", "b"],
                t,
                tmp,
                name="compute_consecutive_month",
                dtype="float32",
            )

            # save the max
            max_count = expr_eval(
                "where(a>b, a, b)",
                ["a", "b"],
                max_count,
                tmp,
                name="compute_consecutive_month",
                dtype="float32",
            )

            # reset if not veg
            tmp = expr_eval(
                "where((a<=0), 0, b)",
                ["a", "b"],
                t,
                tmp,
                name="compute_consecutive_month",
                dtype="float32",
            )

        data_vars = {
            "veg_frequency": xr.DataArray(
                max_count, dims=xx["wet"].dims[1:], attrs=xx.attrs
            )
        }
        coords = dict((dim, xx.coords[dim]) for dim in xx["wet"].dims[1:])
        return xr.Dataset(data_vars=data_vars, coords=coords, attrs=xx["wet"].attrs)


register("veg_count", StatsVegCount)
