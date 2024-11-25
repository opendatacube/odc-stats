"""
 Unit tests for LandCover water classes
"""

import numpy as np
import xarray as xr
import dask.array as da

from odc.stats.plugins.l34_utils import (
    l4_water_persistence,
    l4_water,
)

import pandas as pd

NODATA = 255


# @pytest.fixture(scope="module")
def image_groups(l34, urban, cultivated, woody, bs_pc_50, pv_pc_50, water_frequency):

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
        "artificial_surface": xr.DataArray(
            da.from_array(urban, chunks=(1, -1, -1)),
            dims=("spec", "y", "x"),
            attrs={"nodata": 255},
        ),
        "cultivated_class": xr.DataArray(
            da.from_array(cultivated, chunks=(1, -1, -1)),
            dims=("spec", "y", "x"),
            attrs={"nodata": 255},
        ),
        "woody_cover": xr.DataArray(
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


def test_water_classes(watper_threshold):
    expected_water_classes = [
        [[104, 104, 104], [103, 103, 103], [102, 102, 101], [99, 101, 101]],
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
        dtype="uint8",
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
        dtype="uint8",
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
        dtype="uint8",
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
    water_frequency = np.array(
        [
            [
                [1, 3, 2],
                [4, 5, 6],
                [9, 7, 11],
                [NODATA, 11, 12],
            ]
        ],
        dtype="float",
    )
    xx = image_groups(
        l34, urban, cultivated, woody, bs_pc_50, pv_pc_50, water_frequency
    )

    # Water persistence
    water_persistence = l4_water_persistence.water_persistence(xx, watper_threshold)

    l4_water_classes = l4_water.water_classification(xx, water_persistence)

    assert (l4_water_classes.compute() == expected_water_classes).all()


def test_water_intertidal(watper_threshold):

    expected_water_classes = [
        [100, 100, 100],
        [100, 100, 100],
        [102, 102, 101],
        [101, 99, 100],
    ]

    l34 = np.array(
        [
            [
                [223, 223, 223],
                [223, 223, 223],
                [221, 221, 221],
                [221, 221, 223],
            ]
        ],
        dtype="uint8",
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
        dtype="uint8",
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
        dtype="uint8",
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
    water_frequency = np.array(
        [
            [
                [1, 3, 2],
                [4, 5, 6],
                [9, 7, 11],
                [10, 255, 255],
            ]
        ],
        dtype="uint8",
    )
    xx = image_groups(
        l34, urban, cultivated, woody, bs_pc_50, pv_pc_50, water_frequency
    )

    # Water persistence
    water_persistence = l4_water_persistence.water_persistence(xx, watper_threshold)

    l4_water_classes = l4_water.water_classification(xx, water_persistence)

    assert (l4_water_classes.compute() == expected_water_classes).all()
