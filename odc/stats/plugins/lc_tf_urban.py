"""
Plugin of TF urban model in LandCover PipeLine
"""

from typing import Tuple, Dict, Sequence

import os
import numpy as np
import numexpr as ne
import xarray as xr
import dask.array as da
from dask.distributed import get_worker

from datacube.model import Dataset
from datacube.utils.geometry import GeoBox
from odc.algo._memsink import yxbt_sink
from odc.algo.io import load_with_native_transform

from odc.stats._algebra import expr_eval
from ._registry import StatsPluginInterface, register
from ._worker import TensorFlowLiteModelPlugin

NODATA = 255


# the padding and prediction function
def pad_and_predict(block, crop_size=(256, 256), nodata=NODATA):
    worker = get_worker()
    plugin_instance = worker.plugin_instance
    interpreter = plugin_instance.get_interpreter()

    if block.shape[0] < crop_size[0] or block.shape[1] < crop_size[1]:
        pad_width = [
            (0, max(0, crop_size[0] - block.shape[0])),
            (0, max(0, crop_size[1] - block.shape[1])),
        ] + [(0, 0)] * (block.ndim - 2)
        input_block = np.pad(block, pad_width, mode="edge")
    else:
        input_block = np.copy(block)

    # mark small holes as 0
    input_block = ne.evaluate(
        "where(a==a, a, 0)", local_dict={"a": input_block}
    ).astype("float32")
    # normalize pixels
    norm = np.linalg.norm(input_block, axis=-1, keepdims=True)
    input_block = input_block / np.maximum(norm, 1e-8)  # Avoid division by zero

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(
        input_details[0]["index"], input_block[np.newaxis, ...].astype(np.float32)
    )
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]["index"])

    prediction = np.copy(output_data[0])
    prediction = np.squeeze(prediction, axis=-1)[0 : block.shape[0], 0 : block.shape[1]]
    prediction = ne.evaluate("where(a>0.5, 1, 0)", local_dict={"a": prediction})

    # mark small holes as nodata
    prediction = ne.evaluate(
        "where(a==a, b, nodata)",
        local_dict={"a": block[:, :, 0], "b": prediction, "nodata": nodata},
    ).astype("uint8")
    return prediction


class StatsUrbanClass(StatsPluginInterface):
    NAME = "ga_ls_urban"
    SHORT_NAME = NAME
    VERSION = "0.0.1"
    PRODUCT_FAMILY = "lccs"

    def __init__(
        self,
        output_classes: Dict,
        model_path: str,
        crop_size=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{self.model_path} not found")
        self.dask_worker_plugin = TensorFlowLiteModelPlugin(model_path)
        self.output_classes = output_classes
        if crop_size is None:
            self.crop_size = (256, 256)
        else:
            self.crop_size = crop_size

    @property
    def measurements(self) -> Tuple[str, ...]:
        _measurements = ["urban_classes"]
        return _measurements

    def input_data(
        self, datasets: Sequence[Dataset], geobox: GeoBox, **kwargs
    ) -> xr.Dataset:
        # load data in the same time and location but different sensors
        data_vars = {}

        for ds in datasets:
            xx = load_with_native_transform(
                [ds],
                bands=self.input_bands,
                geobox=geobox,
                native_transform=self.native_transform,
                basis=self.basis,
                groupby=None,
                fuser=None,
                resampling=self.resampling,
                chunks=self.chunks,
                optional_bands=self.optional_bands,
                **kwargs,
            )
            input_array = yxbt_sink(
                xx,
                (self.crop_size[0], self.crop_size[0], -1, -1),
                name=ds.type.name + "_yxbt",
            ).squeeze("spec", drop=True)
            data_vars[ds.type.name] = input_array

        coords = dict((dim, input_array.coords[dim]) for dim in input_array.dims)
        return xr.Dataset(data_vars=data_vars, coords=coords)

    def urban_class(self, input_array):
        urban_mask = da.map_blocks(
            pad_and_predict,
            input_array,
            drop_axis=-1,
            dtype="uint8",
            crop_size=self.crop_size,
        )
        return urban_mask

    def impute_missing_values_from_group(self, xx):
        # Impute the missing values for each image
        images = []
        for img in xx.data_vars:
            image = xx[img].data
            nodata = xx[img].attrs.get("nodata", -999)
            for other_image in xx.data_vars:
                if other_image != img:
                    image = expr_eval(
                        "where((a<=nodata), b, a)",
                        {"a": image, "b": xx[other_image].data},
                        name="impute_missing_values",
                        **{
                            "nodata": nodata,
                        },
                    )

            # convert data type and mark nodata as nan
            image = expr_eval(
                "where((a<=nodata), _nan, a)",
                {
                    "a": image,
                },
                name="convert_dtype",
                dtype="float32",
                **{"nodata": nodata, "_nan": np.nan},
            )
            images += [image]
        return images

    def aggregate_results_from_group(self, urban_masks):
        # if there are >= 2 images
        # any is urban -> final class is urban
        # any is valid -> final class is valid
        # for each pixel
        m_size = len(urban_masks)
        if m_size > 1:
            urban_masks = da.stack(urban_masks).sum(axis=0)
        else:
            urban_masks = urban_masks[0]

        urban_masks = expr_eval(
            "where((a/nodata)>=_l, nodata, a%nodata)",
            {"a": urban_masks},
            name="mark_nodata",
            dtype="float32",
            **{"_l": m_size, "nodata": NODATA},
        )

        urban_masks = expr_eval(
            "where((a>0)&(a<nodata), _u, a)",
            {"a": urban_masks},
            name="output_classes_artificial",
            dtype="float32",
            **{
                "_u": self.output_classes["artificial"],
                "nodata": NODATA,
            },
        )

        urban_masks = expr_eval(
            "where(a<=0, _nu, a)",
            {"a": urban_masks},
            name="output_classes_natrual",
            dtype="uint8",
            **{
                "_nu": self.output_classes["natural"],
            },
        )

        return urban_masks.rechunk(-1, -1)

    def reduce(self, xx: xr.Dataset) -> xr.Dataset:
        urban_masks = []
        images = self.impute_missing_values_from_group(xx)
        for image in images:
            urban_masks += [self.urban_class(image)]

        um = self.aggregate_results_from_group(urban_masks)

        attrs = xx.attrs.copy()
        attrs["nodata"] = int(NODATA)
        dims = list(xx.dims.keys())[:2]
        data_vars = {"urban_classes": xr.DataArray(um, dims=dims, attrs=attrs)}
        coords = {dim: xx.coords[dim] for dim in dims}
        return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


register("urban_class", StatsUrbanClass)
