"""
Tasseled cap index Percentiles
"""

from functools import partial
from typing import Sequence, Tuple, Iterable, Dict
import xarray as xr
import numpy as np
import logging
from odc.algo import keep_good_only
from odc.algo._percentile import xr_quantile_bands
from odc.algo._masking import (
    _xr_fuse,
    _fuse_mean_np,
    enum_to_bool,
    mask_cleanup,
)
from ._registry import StatsPluginInterface, register

_log = logging.getLogger(__name__)
NODATA = -9999  # output NODATA


class StatsTCWPC(StatsPluginInterface):
    NAME = "ga_tc_percentiles"
    SHORT_NAME = NAME
    VERSION = "1.0.1"
    PRODUCT_FAMILY = "tci"

    def __init__(
        self,
        coefficients: Dict[str, Dict[str, float]] = None,
        input_bands: Sequence[str] = None,
        output_bands: Sequence[str] = None,
        cloud_filters: Dict[str, Iterable[Tuple[str, int]]] = None,
        **kwargs,
    ):
        self.cloud_filters = cloud_filters
        if coefficients is None:
            self.coefficients = dict(
                [
                    (
                        "wet",
                        dict(
                            [
                                ("blue", 0.0315),
                                ("green", 0.2021),
                                ("red", 0.3102),
                                ("nir", 0.1594),
                                ("swir1", -0.6806),
                                ("swir2", -0.6109),
                            ]
                        ),
                    ),
                    (
                        "bright",
                        dict(
                            [
                                ("blue", 0.2043),
                                ("green", 0.4158),
                                ("red", 0.5524),
                                ("nir", 0.5741),
                                ("swir1", 0.3124),
                                ("swir2", 0.2303),
                            ]
                        ),
                    ),
                    (
                        "green",
                        dict(
                            [
                                ("blue", -0.1603),
                                ("green", -0.2819),
                                ("red", -0.4934),
                                ("nir", 0.7940),
                                ("swir1", -0.0002),
                                ("swir2", -0.1446),
                            ]
                        ),
                    ),
                ]
            )
        self.input_bands = (
            [
                "blue",
                "green",
                "red",
                "nir",
                "swir1",
                "swir2",
                "fmask",
                "nbart_contiguity",
            ]
            if input_bands is None
            else input_bands
        )

        super().__init__(input_bands=self.input_bands, **kwargs)
        self.output_bands = (
            ["wet", "bright", "green"] if output_bands is None else output_bands
        )

    @property
    def measurements(self) -> Tuple[str, ...]:
        _measurments = []
        for band in self.output_bands:
            _measurments += [f"{band}_pc_10", f"{band}_pc_50", f"{band}_pc_90"]
        return _measurments

    def native_transform(self, xx):
        """
        Loads data in its native projection.
        """
        mask = xx["fmask"]
        nodata = enum_to_bool(mask, ("nodata",))
        non_contiguent = xx["nbart_contiguity"] == 0
        bad = nodata | non_contiguent
        # Now exclude cloud and cloud shadow (including buffered pixels)
        if self.cloud_filters is not None:
            for cloud_class, c_filter in self.cloud_filters.items():
                cloud_mask = enum_to_bool(mask, (cloud_class,))
                cloud_mask_buffered = mask_cleanup(cloud_mask, mask_filters=c_filter)
                bad = cloud_mask_buffered | bad
        else:
            _log.info("There is no cloud/shadow buffering.")
            cloud_shadow_mask = enum_to_bool(mask, ("cloud", "shadow"))
            bad = bad | cloud_shadow_mask

        for band in xx.data_vars.keys():
            bad = bad | (xx[band] == -999)

        yy = xx.drop_vars(self.input_bands)
        for m in self.output_bands:
            yy[m] = sum(
                coeff * xx[band] for band, coeff in self.coefficients[m].items()
            ).astype(np.int16)
            yy[m].attrs = xx.blue.attrs
            yy[m].attrs["nodata"] = NODATA

        yy = keep_good_only(yy, ~bad, nodata=NODATA)
        return yy

    def fuser(self, xx):
        xx = _xr_fuse(xx, partial(_fuse_mean_np, nodata=NODATA), "")

        return xx

    def reduce(self, xx: xr.Dataset) -> xr.Dataset:
        yy = xr_quantile_bands(xx, [0.1, 0.5, 0.9], nodata=NODATA)
        return yy


register("tcw-percentiles", StatsTCWPC)
