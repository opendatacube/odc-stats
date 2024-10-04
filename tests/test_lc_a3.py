import numpy as np
import pandas as pd
import xarray as xr

from odc.stats.plugins.lc_veg_bare_water_a3 import StatsVegDryClassA3
import pytest


expected_canopyco_veg_con = np.array(
    [
        [0, 16, 16],
        [15, 255, 13],
        [12, 12, 12],
        [10, 10, 10],
    ]
)

expected_baregrad_phy_con = [
    [10, 10, 12],
    [12, 15, 15],
    [12, 12, 15],
    [255, 10, 15],
]

expected_waterper_wat_cin = [
    [0, 9, 9],
    [9, 8, 8],
    [7, 7, 7],
    [1, 1, 1],
]

expected_level3 = [
    [1, 93, 94],
    [55, 94, 1],
    [19, 19, 255],
    [98, 55, 98],
]


@pytest.fixture(scope="module")
def dataset():

    pv_pc_50 = np.array(
        [
            [
                [0, 1, 2],
                [4, -999, 15],
                [40, 41, 62],
                [65, 90, 100],
            ]
        ],
        dtype="int",
    )

    bs_pc_50 = np.array(
        [
            [
                [2, 1, 20],
                [25, 60, 65],
                [30, 46, 70],
                [-999, 1, 100],
            ]
        ],
        dtype="int",
    )

    water_frequency = np.array(
        [
            [
                [0, 1, 2],
                [3, 4, 6],
                [7, 8, 9],
                [10, 11, 12],
            ]
        ],
        dtype="float",
    )

    l3 = np.array(
        [
            [
                [111, 215, 216],
                [124, 216, 111],
                [112, 112, 255],
                [220, 124, 220],
            ]
        ],
        dtype="int",
    )

    tuples = [
        (np.datetime64("2000-01-01T00"), np.datetime64("2000-01-01")),
    ]
    index = pd.MultiIndex.from_tuples(tuples, names=["time", "solar_day"])
    coords = {
        "x": np.linspace(10, 20, l3.shape[2]),
        "y": np.linspace(0, 5, l3.shape[1]),
        "spec": index,
    }

    data_vars = {
        "pv_pc_50": xr.DataArray(
            pv_pc_50, dims=("spec", "y", "x"), attrs={"nodata": -999}
        ),
        "bs_pc_50": xr.DataArray(
            bs_pc_50, dims=("spec", "y", "x"), attrs={"nodata": -999}
        ),
        "water_frequency": xr.DataArray(
            water_frequency, dims=("spec", "y", "x"), attrs={"nodata": np.nan}
        ),
        "level3_class": xr.DataArray(
            l3, dims=("spec", "y", "x"), attrs={"nodata": 255}
        ),
    }

    xx = xr.Dataset(data_vars=data_vars, coords=coords)
    return xx


def test_veg_cover_class(dataset):

    lc_a3 = StatsVegDryClassA3()
    veg_cover_classes = lc_a3.a3_veg_cover(dataset)
    assert (veg_cover_classes.compute() == expected_canopyco_veg_con).all()


def test_bare_gradation_class(dataset):

    lc_a3 = StatsVegDryClassA3()
    bg_classes = lc_a3.bare_gradation(dataset)
    assert (bg_classes.compute() == expected_baregrad_phy_con).all()


def test_water_persistence(dataset):
    lc_a3 = StatsVegDryClassA3()
    wp_classes = lc_a3.water_persistence(dataset)
    assert (wp_classes.compute() == expected_waterper_wat_cin).all()


def test_reduce(dataset):
    lc_a3 = StatsVegDryClassA3()
    a3_ds = lc_a3.reduce(dataset)
    assert (a3_ds.canopyco_veg_con.values == expected_canopyco_veg_con).all()
    assert (a3_ds.baregrad_phy_con.values == expected_baregrad_phy_con).all()
    assert (a3_ds.waterper_wat_cin.values == expected_waterper_wat_cin).all()
    assert (a3_ds.level3.values == expected_level3).all()
