"""
 Unit tests for LandCover Natural Aquatic Vegetation classes
"""

import numpy as np
import xarray as xr
import dask.array as da

from odc.stats.plugins.lc_level34 import StatsLccsLevel4
from odc.stats.plugins.l34_utils import (
    l4_cultivated,
    lc_level3,
    l4_veg_cover,
    l4_natural_veg,
    l4_natural_aquatic,
    l4_surface,
    l4_bare_gradation,
    lc_lifeform,
    lc_water_seasonality,
)

import pandas as pd

NODATA = 255


def image_groups(l34, urban, woody, bs_pc_50, pv_pc_50, cultivated, water_frequency):

    tuples = [
        (np.datetime64("2000-01-01T00"), np.datetime64("2000-01-01")),
    ]
    index = pd.MultiIndex.from_tuples(tuples, names=["time", "solar_day"])
    coords = {
        "x": np.linspace(10, 20, l34.shape[2]),
        "y": np.linspace(0, 5, l34.shape[1]),
    }

    data_vars = {
        "level_3_4": xr.DataArray(
            da.from_array(l34, chunks=(1, -1, -1)),
            dims=("spec", "y", "x"),
            attrs={"nodata": 255},
        ),
        "urban_classes": xr.DataArray(
            da.from_array(urban, chunks=(1, -1, -1)),
            dims=("spec", "y", "x"),
            attrs={"nodata": 255},
        ),
        "cultivated": xr.DataArray(
            da.from_array(cultivated, chunks=(1, -1, -1)),
            dims=("spec", "y", "x"),
            attrs={"nodata": 255},
        ),
        "woody": xr.DataArray(
            da.from_array(woody, chunks=(1, -1, -1)),
            dims=("spec", "y", "x"),
            attrs={"nodata": 255},
        ),
        "pv_pc_50": xr.DataArray(
            da.from_array(pv_pc_50, chunks=(1, -1, -1)),
            dims=("spec", "y", "x"),
            attrs={"nodata": 255},
        ),
        "bs_pc_50": xr.DataArray(
            da.from_array(bs_pc_50, chunks=(1, -1, -1)),
            dims=("spec", "y", "x"),
            attrs={"nodata": 255},
        ),
        "water_frequency": xr.DataArray(
            da.from_array(water_frequency, chunks=(1, -1, -1)),
            dims=("spec", "y", "x"),
            attrs={"nodata": 255},
        ),
    }
    xx = xr.Dataset(data_vars=data_vars, coords=coords)
    xx = xx.assign_coords(xr.Coordinates.from_pandas_multiindex(index, "spec"))
    return xx


def test_ns():
    expected_l4_srf_classes = [
        [95, 97, 93],
        [97, 96, 96],
        [95, 95, 95],
        [94, 95, 96],
    ]

    l34 = np.array(
        [
            [
                [210, 210, 210],
                [210, 210, 210],
                [210, 210, 210],
                [210, 210, 210],
            ]
        ],
        dtype="uint8",
    )

    urban = np.array(
        [
            [
                [216, 216, 215],
                [216, 216, 216],
                [216, 216, 216],
                [216, 216, 216],
            ]
        ],
        dtype="uint8",
    )

    woody = np.array(
        [
            [
                [113, 113, 113],
                [113, 113, 255],
                [114, 114, 114],
                [114, 114, 255],
            ]
        ],
        dtype="uint8",
    )

    pv_pc_50 = np.array(
        [
            [
                [1, 64, 65],
                [66, 40, 41],
                [3, 16, 15],
                [4, 1, 42],
            ]
        ],
        dtype="uint8",
    )

    bs_pc_50 = np.array(
        [
            [
                [1, 64, NODATA],
                [66, 40, 41],
                [3, 16, 15],
                [NODATA, 1, 42],
            ]
        ],
        dtype="uint8",
    )
    # 112 --> natural veg
    cultivated = np.array(
        [
            [
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
            ]
        ],
        dtype="uint8",
    )

    water_frequency = np.array(
        [
            [
                [1, 3, 2],
                [4, 5, 6],
                [9, 2, 11],
                [10, 11, 12],
            ]
        ],
        dtype="uint8",
    )

    xx = image_groups(
        l34, urban, woody, bs_pc_50, pv_pc_50, cultivated, water_frequency
    )

    stats_l4 = StatsLccsLevel4()
    level3 = lc_level3.lc_level3(xx)
    lifeform = lc_lifeform.lifeform(xx)
    veg_cover = l4_veg_cover.canopyco_veg_con(xx, stats_l4.veg_threshold)

    # Apply cultivated to match the code in Level4 processing
    l4_ctv = l4_cultivated.lc_l4_cultivated(xx.level_3_4, level3, lifeform, veg_cover)
    l4_ctv_ntv = l4_natural_veg.lc_l4_natural_veg(l4_ctv, level3, lifeform, veg_cover)

    water_seasonality = lc_water_seasonality.water_seasonality(
        xx, stats_l4.water_seasonality_threshold
    )
    l4_ctv_ntv_nav = l4_natural_aquatic.natural_auquatic_veg(
        l4_ctv_ntv, veg_cover, water_seasonality
    )

    # Bare gradation
    bare_gradation = l4_bare_gradation.bare_gradation(
        xx, stats_l4.bare_threshold, veg_cover
    )

    l4_ctv_ntv_nav_surface = l4_surface.lc_l4_surface(
        l4_ctv_ntv_nav, level3, bare_gradation
    )

    assert (l4_ctv_ntv_nav_surface.compute() == expected_l4_srf_classes).all()
