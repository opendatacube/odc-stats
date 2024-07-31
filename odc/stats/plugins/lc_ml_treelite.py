"""
Plugin of TF urban model in LandCover PipeLine
"""

from abc import abstractmethod
from typing import Dict, Sequence, Optional

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
from ._registry import StatsPluginInterface
from ._worker import TreeliteModelPlugin
import tl2cgen


def mask_and_predict(
    block, block_info=None, ptype="categorical", nodata=np.nan, output_dtype="float32"
):
    worker = get_worker()
    plugin_instance = worker.plugin_instance
    predictor = plugin_instance.get_predictor()

    block_flat = block.reshape(-1, block.shape[-1])
    # mask nodata and non-veg
    mask_flat = ne.evaluate(
        "where((a==a)&(b>0), 1, 0)",
        local_dict={"a": block_flat[:, 0], "b": block_flat[:, -1]},
    ).astype("bool")
    block_masked = block_flat[mask_flat, :-1]

    prediction = np.full(
        (block.shape[0] * block.shape[1], 1), nodata, dtype=output_dtype
    )
    if block_masked.shape[0] > 0:
        dmat = tl2cgen.DMatrix(block_masked)
        output_data = predictor.predict(dmat).squeeze(axis=1)
        if ptype == "categorical":
            prediction[mask_flat] = output_data.argmax(axis=-1)
        else:
            prediction[mask_flat] = output_data
    return prediction.reshape(*block.shape[:-1])


class StatsMLTree(StatsPluginInterface):
    NAME = "ga_ls_ml_tree"
    SHORT_NAME = NAME
    VERSION = "0.0.1"
    PRODUCT_FAMILY = "lccs"

    def __init__(
        self,
        output_classes: Dict,
        model_path: str,
        mask_bands: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{self.model_path} not found")
        self.dask_worker_plugin = TreeliteModelPlugin(model_path)
        self.output_classes = output_classes
        self.mask_bands = mask_bands

    def input_data(
        self, datasets: Sequence[Dataset], geobox: GeoBox, **kwargs
    ) -> xr.Dataset:
        # load data in the same time and location but different sensors
        data_vars = {}

        for ds in datasets:
            if "gm" in ds.type.name:
                input_bands = self.input_bands[:-1]
            else:
                input_bands = self.input_bands[-1:]

            xx = load_with_native_transform(
                [ds],
                bands=input_bands,
                geobox=geobox,
                native_transform=self.native_transform,
                basis=self.basis,
                groupby=None,
                fuser=None,
                resampling=self.resampling,
                chunks={"y": -1, "x": -1},
                optional_bands=self.optional_bands,
                **kwargs,
            )
            if "gm" in ds.type.name:
                input_array = yxbt_sink(
                    xx,
                    (self.chunks["x"], self.chunks["y"], -1, -1),
                    name=ds.type.name + "_yxbt",
                ).squeeze("spec", drop=True)
                data_vars[ds.type.name] = input_array
            else:
                for var in xx.data_vars:
                    data_vars[var] = xx[var].squeeze(dim="spec")

        coords = dict((dim, input_array.coords[dim]) for dim in input_array.dims)
        return xr.Dataset(data_vars=data_vars, coords=coords)

    def preprocess_predict_input(self, xx: xr.Dataset):
        images = []
        for var in xx.data_vars:
            image = xx[var].data
            if var not in self.mask_bands:
                nodata = xx[var].attrs.get("nodata", -999)
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
            else:
                veg_mask = expr_eval(
                    "where(a==_v, 1, 0)",
                    {"a": image},
                    name="make_mask",
                    dtype="float32",
                    **{"_v": int(self.mask_bands[var])},
                )

        images = [
            da.concatenate([image, veg_mask[..., np.newaxis]], axis=-1)
            for image in images
        ]
        return images

    @abstractmethod
    def predict(self, input_array):
        pass

    @abstractmethod
    def aggregate_results_from_group(self, predict_output):
        pass

    def reduce(self, xx: xr.Dataset) -> xr.Dataset:
        print(f"input dataset {xx}")
        images = self.preprocess_predict_input(xx)
        print(f"after preprocess  {images}")
        res = []

        for image in images:
            res += [self.predict(image)]

        res = self.aggregate_results_from_group(res)
        attrs = xx.attrs.copy()
        dims = list(xx.dims.keys())[:2]
        data_vars = {"predict_output": xr.DataArray(res, dims=dims, attrs=attrs)}
        coords = {dim: xx.coords[dim] for dim in dims}
        return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
