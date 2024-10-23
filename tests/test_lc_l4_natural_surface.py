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
)

import pytest
import pandas as pd

NODATA = 255
FC_NODATA = -999
WAT_FREQ_NODATA = -999


def image_groups(l34, urban, woody, bs_pc_50, pv_pc_50, cultivated, water_frequency):

    tuples = [
        (np.datetime64("2000-01-01T00"), np.datetime64("2000-01-01")),
    ]
    index = pd.MultiIndex.from_tuples(tuples, names=["time", "solar_day"])
    coords = {
        "x": np.linspace(10, 20, l34.shape[2]),
        "y": np.linspace(0, 5, l34.shape[1]),
        "spec": index,
    }

    data_vars = {
        "classes_l3_l4": xr.DataArray(
            l34, dims=("spec", "y", "x"), attrs={"nodata": 255}
        ),
        "urban_classes": xr.DataArray(
            urban, dims=("spec", "y", "x"), attrs={"nodata": 255}
        ),
        "cultivated_class": xr.DataArray(
            cultivated, dims=("spec", "y", "x"), attrs={"nodata": 255}
        ),
        "woody_cover": xr.DataArray(
            woody, dims=("spec", "y", "x"), attrs={"nodata": 255}
        ),
        "pv_pc_50": xr.DataArray(
            pv_pc_50, dims=("spec", "y", "x"), attrs={"nodata": 255}
        ),
        "bs_pc_50": xr.DataArray(
            bs_pc_50, dims=("spec", "y", "x"), attrs={"nodata": 255}
        ),
        "water_frequency": xr.DataArray(
            water_frequency, dims=("spec", "y", "x"), attrs={"nodata": 255}
        ),
    }
    xx = xr.Dataset(data_vars=data_vars, coords=coords)
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
        dtype="int",
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
        dtype="int",
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
        dtype="int",
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
        dtype="int",
    )

    bs_pc_50 = np.array(
        [
            [
                [1, 64, FC_NODATA],
                [66, 40, 41],
                [3, 16, 15],
                [FC_NODATA, 1, 42],
            ]
        ],
        dtype="int",
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
        dtype="int",
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
        dtype="int",
    )

    xx = image_groups(
        l34, urban, woody, bs_pc_50, pv_pc_50, cultivated, water_frequency
    )

    stats_l4 = StatsLccsLevel4()
    intertidal_mask, level3 = lc_level3.lc_level3(xx, NODATA)
    lifeform = stats_l4.define_life_form(xx)
    veg_cover = l4_veg_cover.canopyco_veg_con(
        xx, stats_l4.veg_threshold, NODATA, FC_NODATA
    )
    veg_cover = stats_l4.apply_mapping(veg_cover, stats_l4.veg_mapping)

    # Apply cultivated to match the code in Level4 processing
    l4_ctv = l4_cultivated.lc_l4_cultivated(
        xx.classes_l3_l4, level3, lifeform, veg_cover
    )
    l4_ctv_ntv = l4_natural_veg.lc_l4_natural_veg(l4_ctv, level3, lifeform, veg_cover)

    water_seasonality = stats_l4.define_water_seasonality(xx)
    l4_ctv_ntv_nav = l4_natural_aquatic.natural_auquatic_veg(
        l4_ctv_ntv, lifeform, veg_cover, water_seasonality
    )

    # Bare gradation
    bare_gradation = l4_bare_gradation.bare_gradation(
        xx, stats_l4.bare_threshold, veg_cover, NODATA
    )
    # Apply bare gradation expected output classes
    bare_gradation = stats_l4.apply_mapping(bare_gradation, stats_l4.bs_mapping)

    l4_ctv_ntv_nav_surface = l4_surface.lc_l4_surface(
        l4_ctv_ntv_nav, level3, bare_gradation
    )

    assert (l4_ctv_ntv_nav_surface.compute() == expected_l4_srf_classes).all()
