"""
Plugin of RFclassfication cultivated  model in LandCover PipeLine
"""

from typing import Tuple
import numpy as np
import xarray as xr
import dask.array as da
import numexpr as ne

from odc.stats._algebra import expr_eval
from ._registry import register
from .lc_ml_treelite import StatsMLTree, mask_and_predict

NODATA = 255

# selected features from sklearn model
# ['nbart_blue', 'nbart_green', 'nbart_swir_1', 'nbart_swir_2', 'sdev',
#        'edev', 'bcdev', 'MNDWI', 'BUI', 'BSI', 'TCW', 'NDMI', 'AWEI_sh',
#        'BAEI', 'NDSI'],


# pylint: disable=invalid-name
def feature_MNDWI(input_block, nbart_green, nbart_swir_1):
    return ne.evaluate(
        "(a-b)/(a+b)",
        local_dict={
            "a": input_block[..., nbart_green],
            "b": input_block[..., nbart_swir_1],
        },
    ).astype("float32")


def feature_BUI(input_block, nbart_swir_1, nbart_nir, nbart_red):
    return ne.evaluate(
        "((a-b)/(a+b))-((b-c)/(b+c))",
        local_dict={
            "a": input_block[..., nbart_swir_1],
            "b": input_block[..., nbart_nir],
            "c": input_block[..., nbart_red],
        },
    ).astype("float32")


def feature_BSI(input_block, nbart_swir_1, nbart_red, nbart_nir, nbart_blue):
    return ne.evaluate(
        "((a+b)-(c+d))/((a+b)+(c+d))",
        local_dict={
            "a": input_block[..., nbart_swir_1],
            "b": input_block[..., nbart_red],
            "c": input_block[..., nbart_nir],
            "d": input_block[..., nbart_blue],
        },
    ).astype("float32")


def feature_TCW(
    input_block,
    nbart_blue,
    nbart_green,
    nbart_red,
    nbart_nir,
    nbart_swir_1,
    nbart_swir_2,
):
    return ne.evaluate(
        "(0.0315*a+0.2021*b+0.3102*c+0.1594*d+-0.6806*e+-0.6109*f)",
        local_dict={
            "a": input_block[..., nbart_blue],
            "b": input_block[..., nbart_green],
            "c": input_block[..., nbart_red],
            "d": input_block[..., nbart_nir],
            "e": input_block[..., nbart_swir_1],
            "f": input_block[..., nbart_swir_2],
        },
    ).astype("float32")


def feature_NDMI(input_block, nbart_nir, nbart_swir_1):
    return ne.evaluate(
        "(a-b)/(a+b)",
        local_dict={
            "a": input_block[..., nbart_nir],
            "b": input_block[..., nbart_swir_1],
        },
    ).astype("float32")


def feature_AWEI_sh(
    input_block, nbart_blue, nbart_green, nbart_nir, nbart_swir_1, nbart_swir_2
):
    return ne.evaluate(
        "(a+2.5*b-1.5*(c+d)-0.25*e)",
        local_dict={
            "a": input_block[..., nbart_blue],
            "b": input_block[..., nbart_green],
            "c": input_block[..., nbart_nir],
            "d": input_block[..., nbart_swir_1],
            "e": input_block[..., nbart_swir_2],
        },
    ).astype("float32")


def feature_BAEI(input_block, nbart_red, nbart_green, nbart_swir_1):
    return ne.evaluate(
        "(a+0.3)/(b+c)",
        local_dict={
            "a": input_block[..., nbart_red],
            "b": input_block[..., nbart_green],
            "c": input_block[..., nbart_swir_1],
        },
    ).astype("float32")


def feature_NDSI(input_block, nbart_green, nbart_swir_1):
    return ne.evaluate(
        "(a-b)/(a+b)",
        local_dict={
            "a": input_block[..., nbart_green],
            "b": input_block[..., nbart_swir_1],
        },
    ).astype("float32")


def generate_features(input_block, bands_indices):
    feature_input_indices = [
        [bands_indices[k] for k in ["nbart_green", "nbart_swir_1"]],
        [bands_indices[k] for k in ["nbart_swir_1", "nbart_nir", "nbart_red"]],
        [
            bands_indices[k]
            for k in ["nbart_swir_1", "nbart_red", "nbart_nir", "nbart_blue"]
        ],
        [
            bands_indices[k]
            for k in [
                "nbart_blue",
                "nbart_green",
                "nbart_red",
                "nbart_nir",
                "nbart_swir_1",
                "nbart_swir_2",
            ]
        ],
        [bands_indices[k] for k in ["nbart_nir", "nbart_swir_1"]],
        [
            bands_indices[k]
            for k in [
                "nbart_blue",
                "nbart_green",
                "nbart_nir",
                "nbart_swir_1",
                "nbart_swir_2",
            ]
        ],
        [bands_indices[k] for k in ["nbart_red", "nbart_green", "nbart_swir_1"]],
        [bands_indices[k] for k in ["nbart_green", "nbart_swir_1"]],
    ]

    # normalized against 6 bands
    norm = np.linalg.norm(
        input_block[..., : bands_indices["nbart_swir_2"] + 1], axis=-1, keepdims=True
    )
    output_block = input_block[..., : bands_indices["nbart_swir_2"] + 1] / np.maximum(
        norm, 1e-8
    )  # Avoid division by zero, fine if it's nan

    # reassemble the array
    output_block = np.concatenate(
        [output_block, input_block[..., bands_indices["sdev"]][..., np.newaxis]],
        axis=-1,
    ).astype("float32")
    # scale edev \in [0, 1]
    edev = input_block[..., bands_indices["edev"]] / 1e4
    output_block = np.concatenate(
        [output_block, edev[..., np.newaxis]], axis=-1
    ).astype("float32")
    output_block = np.concatenate(
        [output_block, input_block[..., bands_indices["bcdev"]][..., np.newaxis]],
        axis=-1,
    ).astype("float32")

    for f, p in zip(
        [
            feature_MNDWI,
            feature_BUI,
            feature_BSI,
            feature_TCW,
            feature_NDMI,
            feature_AWEI_sh,
            feature_BAEI,
            feature_NDSI,
        ],
        feature_input_indices,
    ):
        ib = f(output_block[..., : bands_indices["nbart_swir_2"] + 1], *p)
        output_block = np.concatenate([output_block, ib[..., np.newaxis]], axis=-1)

    selected_indices = np.r_[
        [
            bands_indices[k]
            for k in [
                "nbart_blue",
                "nbart_green",
                "nbart_swir_1",
                "nbart_swir_2",
                "sdev",
                "edev",
                "bcdev",
            ]
        ],
        (bands_indices["bcdev"] + 1) : output_block.shape[-1],
    ]
    output_block = np.concatenate(
        [
            output_block[..., selected_indices],
            input_block[..., (bands_indices["bcdev"] + 1) :],
        ],
        axis=-1,
    )
    return output_block


class StatsCultivatedClass(StatsMLTree):
    NAME = "ga_ls_cultivated"
    SHORT_NAME = NAME
    VERSION = "0.0.1"
    PRODUCT_FAMILY = "lccs"

    @property
    def measurements(self) -> Tuple[str, ...]:
        _measurements = ["cultivated_class"]
        return _measurements

    def predict(self, input_array):
        bands_indices = dict(zip(self.input_bands, np.arange(len(self.input_bands))))
        input_features = da.map_blocks(
            generate_features,
            input_array,
            bands_indices,
            chunks=(
                input_array.chunks[0],
                input_array.chunks[1],
                15 + len(bands_indices) - bands_indices["bcdev"] - 1,
            ),
            dtype="float32",
            name="generate_features",
        )
        cc = da.map_blocks(
            mask_and_predict,
            input_features,
            ptype="categorical",
            nodata=NODATA,
            drop_axis=-1,
            dtype="float32",
            name="cultivated_predict",
        )

        return cc

    def aggregate_results_from_group(self, predict_output):
        # if there are >= 2 images
        # any is cultivated -> final class is cultivated
        # any is valid -> final class is valid
        # for each pixel
        m_size = len(predict_output)
        if m_size > 1:
            predict_output = da.stack(predict_output)
        else:
            predict_output = predict_output[0]

        predict_output = expr_eval(
            "where(a<nodata, 1-a, a)",
            {"a": predict_output},
            name="invert_output",
            dtype="float32",
            **{"nodata": NODATA},
        )

        if m_size > 1:
            predict_output = predict_output.sum(axis=0)

        predict_output = expr_eval(
            "where((a/nodata)>=_l, nodata, a%nodata)",
            {"a": predict_output},
            name="mark_nodata",
            dtype="float32",
            **{"_l": m_size, "nodata": NODATA},
        )

        predict_output = expr_eval(
            "where((a>0)&(a<nodata), _u, a)",
            {"a": predict_output},
            name="output_classes_cultivated",
            dtype="float32",
            **{"_u": self.output_classes["cultivated"], "nodata": NODATA},
        )

        predict_output = expr_eval(
            "where(a<=0, _nu, a)",
            {"a": predict_output},
            name="output_classes_natural",
            dtype="uint8",
            **{"_nu": self.output_classes["natural"]},
        )

        return predict_output.rechunk(-1, -1)

    def reduce(self, xx: xr.Dataset) -> xr.Dataset:
        res = super().reduce(xx)

        for var in res.data_vars:
            attrs = res[var].attrs.copy()
            attrs["nodata"] = int(NODATA)
            res[var].attrs = attrs
            var_rename = {var: "cultivated_class"}
        return res.rename(var_rename)


register("cultivated_class", StatsCultivatedClass)
