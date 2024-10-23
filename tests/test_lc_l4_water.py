"""
 Unit tests for LandCover water classes
"""

import numpy as np
import xarray as xr
import dask.array as da
from odc.stats.plugins.lc_level34 import StatsLccsLevel4
from odc.stats.plugins.lc_level3 import StatsLccsLevel3
from odc.stats.plugins.l34_utils import (
    l4_cultivated,
    lc_level3,
    l4_veg_cover,
    l4_natural_veg,
    l4_natural_aquatic,
    l4_bare_gradation,
    l4_water_persistence,
    l4_surface,
    l4_water,
)

import pytest
import pandas as pd

NODATA = 255
FC_NODATA = -999


# @pytest.fixture(scope="module")
def image_groups(l34, urban, cultivated, woody, bs_pc_50, pv_pc_50, water_frequency):

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


def test_water_classes():

    expected_water_classes = [
        [104, 104, 104],
        [103, 103, 103],
        [102, 102, 101],
        [101, 101, 101],
    ]

    l34 = np.array(
        [
            [
                [221, 221, 221],
                [221, 221, 221],
                [221, 221, 221],
                [221, 221, 221],
            ]
        ],
        dtype="int",
    )

    urban = np.array(
        [
            [
                [216, 216, 216],
                [216, 216, 216],
                [216, 216, 216],
                [216, 216, 216],
            ]
        ],
        dtype="int",
    )
    # 112 --> natural veg
    cultivated = np.array(
        [
            [
                [112, 112, 112],
                [255, 112, 112],
                [112, 112, 112],
                [112, 112, 112],
            ]
        ],
        dtype="int",
    )

    woody = np.array(
        [
            [
                [114, 114, 114],
                [114, 114, 114],
                [114, 114, 114],
                [114, 114, 114],
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
    water_frequency = np.array(
        [
            [
                [1, 3, 2],
                [4, 5, 6],
                [9, 7, 11],
                [10, 11, 12],
            ]
        ],
        dtype="int",
    )
    xx = image_groups(
        l34, urban, cultivated, woody, bs_pc_50, pv_pc_50, water_frequency
    )

    stats_l4 = StatsLccsLevel4()
    stats_l3 = StatsLccsLevel3()
    intertidal_mask, level3 = stats_l3.reduce(xx)

    lifeform = stats_l4.define_life_form(xx)
    veg_cover = l4_veg_cover.canopyco_veg_con(
        xx, stats_l4.veg_threshold, NODATA, FC_NODATA
    )
    veg_cover = stats_l4.apply_mapping(veg_cover, stats_l4.veg_mapping)

    # Water persistence
    water_persistence = l4_water_persistence.water_persistence(
        xx, stats_l4.watper_threshold, NODATA
    )
    # Apply water persistence expcted classes
    water_persistence = stats_l4.apply_mapping(
        water_persistence, stats_l4.waterper_wat_mapping
    )
    l4_water_classes = l4_water.water_classification(
        xx.classes_l3_l4, level3, intertidal_mask, water_persistence, NODATA
    )

    assert (l4_water_classes.compute() == expected_water_classes).all()


def test_water_intertidal():

    expected_water_classes = [
        [104, 104, 104],
        [103, 103, 103],
        [102, 102, 101],
        [101, 99, 99],
    ]

    l34 = np.array(
        [
            [
                [221, 221, 221],
                [221, 221, 221],
                [221, 221, 221],
                [221, 221, 221],
            ]
        ],
        dtype="int",
    )

    urban = np.array(
        [
            [
                [216, 216, 216],
                [216, 216, 216],
                [216, 216, 216],
                [216, 216, 216],
            ]
        ],
        dtype="int",
    )
    # 112 --> natural veg
    cultivated = np.array(
        [
            [
                [112, 112, 112],
                [255, 112, 112],
                [112, 112, 112],
                [112, 112, 112],
            ]
        ],
        dtype="int",
    )

    woody = np.array(
        [
            [
                [114, 114, 114],
                [114, 114, 114],
                [114, 114, 114],
                [114, 114, 114],
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
    water_frequency = np.array(
        [
            [
                [1, 3, 2],
                [4, 5, 6],
                [9, 7, 11],
                [10, 255, 255],
            ]
        ],
        dtype="int",
    )
    xx = image_groups(
        l34, urban, cultivated, woody, bs_pc_50, pv_pc_50, water_frequency
    )

    stats_l4 = StatsLccsLevel4()
    stats_l3 = StatsLccsLevel3()
    intertidal_mask, level3 = stats_l3.reduce(xx)

    lifeform = stats_l4.define_life_form(xx)
    veg_cover = l4_veg_cover.canopyco_veg_con(
        xx, stats_l4.veg_threshold, NODATA, FC_NODATA
    )
    veg_cover = stats_l4.apply_mapping(veg_cover, stats_l4.veg_mapping)

    # Water persistence
    water_persistence = l4_water_persistence.water_persistence(
        xx, stats_l4.watper_threshold, NODATA
    )
    # Apply water persistence expcted classes
    water_persistence = stats_l4.apply_mapping(
        water_persistence, stats_l4.waterper_wat_mapping
    )
    l4_water_classes = l4_water.water_classification(
        xx.classes_l3_l4, level3, intertidal_mask, water_persistence, NODATA
    )

    assert (l4_water_classes.compute() == expected_water_classes).all()
