"""
Plugin of Module A3 in LandCover PipeLine
"""

from typing import Optional, Dict, List

import xarray as xr
import s3fs
import os
import pandas as pd
import dask.array as da
import logging

from ._registry import StatsPluginInterface, register
from ._utils import rasterize_vector_mask, generate_numexpr_expressions
from odc.stats._algebra import expr_eval
from osgeo import gdal

NODATA = 255
_log = logging.getLogger(__name__)


class StatsLccsLevel4(StatsPluginInterface):
    NAME = "ga_ls_lccs_Level34"
    SHORT_NAME = NAME
    VERSION = "0.0.1"
    PRODUCT_FAMILY = "lccs"

    def __init__(
        self,
        class_def_path: str = None,
        class_condition: Dict[str, List] = None,
        urban_mask: str = None,
        filter_expression: str = None,
        mask_threshold: Optional[float] = None,
        data_var_condition: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if class_def_path is None:
            raise ValueError("Missing level34 class definition csv")

        if class_def_path.startswith("s3://"):
            if not s3fs.S3FileSystem(anon=True).exists(class_def_path):
                raise FileNotFoundError(f"{class_def_path} not found")
        elif not os.path.exists(class_def_path):
            raise FileNotFoundError(f"{class_def_path} not found")

        if class_condition is None:
            raise ValueError("Missing input to generate classification conditions")

        if urban_mask is None:
            raise ValueError("Missing urban mask shapefile")

        file_metadata = gdal.VSIStatL(urban_mask)
        if file_metadata is None:
            raise FileNotFoundError(f"{urban_mask} not found")

        if filter_expression is None:
            raise ValueError("Missing urban mask filter")

        self.class_def = pd.read_csv(class_def_path)
        self.class_condition = class_condition
        cols = set()
        for k, v in self.class_condition.items():
            cols |= {k} | set(v)

        self.class_def = self.class_def[list(cols)].astype(str).fillna("nan")

        self.urban_mask = urban_mask
        self.filter_expression = filter_expression
        self.mask_threshold = mask_threshold
        self.data_var_condition = (
            {} if data_var_condition is None else data_var_condition
        )

    def fuser(self, xx):
        return xx

    def classification(self, xx, class_def, con_cols, class_col):
        expressions = generate_numexpr_expressions(
            class_def[con_cols + [class_col]], class_col, "res"
        )
        local_dict = {
            key: xx[self.data_var_condition.get(key, key)].data for key in con_cols
        }
        res = da.full(xx.level_3_4.shape, 0, dtype="uint8")

        for expression in expressions:
            _log.debug(expression)
            local_dict.update({"res": res})
            res = expr_eval(
                expression,
                local_dict,
                name="apply_rule",
                dtype="uint8",
            )

        # This seems redundant while res can be init to NODATA above,
        # but it's a point to sanity check no class is missed
        res = expr_eval(
            "where((a!=a)|(a>=_n), _n, b)",
            {"a": xx.level_3_4.data, "b": res},
            name="mark_nodata",
            dtype="uint8",
            **{"_n": NODATA},
        )

        return res

    def reduce(self, xx: xr.Dataset) -> xr.Dataset:
        class_col = "level3"
        level3 = self.classification(
            xx, self.class_def, self.class_condition[class_col], class_col
        )

        # apply urban mask
        # 215 -> 216 if urban_mask == 0
        urban_mask = rasterize_vector_mask(
            self.urban_mask,
            xx.geobox.transform,
            xx.artificial_surface.shape,
            filter_expression=self.filter_expression,
            threshold=self.mask_threshold,
        )

        level3 = expr_eval(
            "where((a==215)&(b<1), 216, a)",
            {"a": level3, "b": urban_mask},
            name="mask_non_urban",
            dtype="uint8",
        )

        # append level3 to the input dataset so it can be used
        # to classify level4
        attrs = xx.attrs.copy()
        attrs["nodata"] = NODATA
        dims = xx.level_3_4.dims[1:]
        coords = dict((dim, xx.coords[dim]) for dim in dims)
        xx["level3"] = xr.DataArray(
            level3.squeeze(), dims=dims, attrs=attrs, coords=coords
        )

        class_col = "level4"
        level4 = self.classification(
            xx, self.class_def, self.class_condition[class_col], class_col
        )

        data_vars = {
            k: xr.DataArray(v, dims=dims, attrs=attrs)
            for k, v in zip(self.measurements, [level3.squeeze(), level4.squeeze()])
        }

        leve34 = xr.Dataset(data_vars=data_vars, coords=coords, attrs=xx.attrs)
        return leve34


register("lccs_level34", StatsLccsLevel4)
