"""
Plugin of TF urban model in LandCover PipeLine
"""

from typing import Tuple
import xarray as xr
import dask.array as da

from odc.stats._algebra import expr_eval
from ._registry import register
from .lc_ml_treelite import StatsMLTree, mask_and_predict

NODATA = 255


class StatsWoodyCover(StatsMLTree):
    NAME = "ga_ls_woody_cover"
    SHORT_NAME = NAME
    VERSION = "0.0.1"
    PRODUCT_FAMILY = "lccs"

    @property
    def measurements(self) -> Tuple[str, ...]:
        _measurements = ["woody_cover"]
        return _measurements

    def predict(self, input_array):
        wc = da.map_blocks(
            mask_and_predict,
            input_array,
            ptype="regression",
            nodata=NODATA,
            drop_axis=-1,
            dtype="uint8",
        )
        return wc

    def aggregate_results_from_group(self, predict_output):
        # >= 0.2 -> woody < 0.2 -> herbaceous
        # if there are >= 2 images
        # any valid -> valid
        # any herbaceous -> herbaceous
        woody_covers = predict_output
        m_size = len(woody_covers)
        if m_size > 1:
            woody_covers = da.stack(woody_covers)
        else:
            woody_covers = woody_covers[0]

        woody_covers = expr_eval(
            "where(a<=20, 1, a)",
            {"a": woody_covers},
            name="mark_herbaceous",
            dtype="float32",
        )

        woody_covers = expr_eval(
            "where((a>20)&(a<nodata), 0, a)",
            {"a": woody_covers},
            name="mark_woody",
            dtype="float32",
            **{"nodata": NODATA},
        )

        if m_size > 1:
            woody_covers = da.sum(woody_covers, axis=0)

        woody_covers = expr_eval(
            "where((a/nodata)>=_l, nodata, a%nodata)",
            {"a": woody_covers},
            name="summary_over_classes",
            dtype="uint8",
            **{
                "_l": m_size,
                "nodata": NODATA,
            },
        )

        woody_covers = expr_eval(
            "where((a>0)&(a<nodata), _nw, a)",
            {"a": woody_covers},
            name="output_classes_herbaceous",
            dtype="uint8",
            **{"nodata": NODATA, "_nw": self.output_classes["herbaceous"]},
        )

        woody_covers = expr_eval(
            "where(a<=0, _nw, a)",
            {"a": woody_covers},
            name="output_classes_woody",
            dtype="uint8",
            **{"_nw": self.output_classes["woody"]},
        )

        return woody_covers.rechunk(-1, -1)

    def reduce(self, xx: xr.Dataset) -> xr.Dataset:
        res = super().reduce(xx)

        for var in res.data_vars:
            attrs = res[var].attrs.copy()
            attrs["nodata"] = int(NODATA)
            res[var].attrs = attrs
            var_rename = {var: "woody_cover"}
        return res.rename(var_rename)


register("woody_cover", StatsWoodyCover)
