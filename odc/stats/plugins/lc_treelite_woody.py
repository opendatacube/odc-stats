"""
Plugin of RFregressor woody cover model in LandCover PipeLine
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
            dtype="float32",
            name="woody_cover_predict",
        )
        return wc

    def aggregate_results_from_group(self, predict_output):
        # >= 0.2 -> woody < 0.2 -> herbaceous
        # if there are >= 2 images
        # any valid -> valid
        # any herbaceous -> herbaceous
        m_size = len(predict_output)
        if m_size > 1:
            predict_output = da.stack(predict_output)
        else:
            predict_output = predict_output[0]

        predict_output = expr_eval(
            "where(a<=20, 1, a)",
            {"a": predict_output},
            name="mark_herbaceous",
            dtype="float32",
        )

        predict_output = expr_eval(
            "where((a>20)&(a<nodata), 0, a)",
            {"a": predict_output},
            name="mark_woody",
            dtype="float32",
            **{"nodata": NODATA},
        )

        if m_size > 1:
            predict_output = predict_output.sum(axis=0)

        predict_output = expr_eval(
            "where((a/nodata)>=_l, nodata, a%nodata)",
            {"a": predict_output},
            name="summary_over_classes",
            dtype="float32",
            **{
                "_l": m_size,
                "nodata": NODATA,
            },
        )

        predict_output = expr_eval(
            "where((a>0)&(a<nodata), _nw, a)",
            {"a": predict_output},
            name="output_classes_herbaceous",
            dtype="float32",
            **{"nodata": NODATA, "_nw": self.output_classes["herbaceous"]},
        )

        predict_output = expr_eval(
            "where(a<=0, _nw, a)",
            {"a": predict_output},
            name="output_classes_woody",
            dtype="uint8",
            **{"_nw": self.output_classes["woody"]},
        )

        return predict_output.rechunk(-1, -1)

    def reduce(self, xx: xr.Dataset) -> xr.Dataset:
        res = super().reduce(xx)

        for var in res.data_vars:
            attrs = res[var].attrs.copy()
            attrs["nodata"] = int(NODATA)
            res[var].attrs = attrs
            var_rename = {var: "woody_cover"}
        return res.rename(var_rename)


register("woody_cover", StatsWoodyCover)
