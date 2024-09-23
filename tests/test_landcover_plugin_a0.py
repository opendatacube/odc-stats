from functools import partial
import numpy as np
import xarray as xr
import dask.array as da
from odc.stats.plugins.lc_fc_wo_a0 import StatsVegCount
from odc.stats._algebra import median_ds
import pytest
import pandas as pd


@pytest.fixture
def dataset_md():
    band_1 = np.array(
        [
            [
                [77.0, np.nan, 59.0, np.nan],
                [8.0, 80.0, 70.0, 97.0],
                [48.0, 75.0, 80.0, 53.0],
                [np.nan, 23.0, 70.0, 49.0],
            ],
            [
                [90.0, 90.0, 41.0, np.nan],
                [np.nan, 51.0, 60.0, np.nan],
                [42.0, 76.0, 80.0, 86.0],
                [34.0, np.nan, 52.0, 46.0],
            ],
            [
                [51.0, np.nan, 55.0, np.nan],
                [31.0, 67.0, 15.0, 69.0],
                [45.0, 86.0, 29.0, np.nan],
                [87.0, 83.0, np.nan, 2.0],
            ],
        ],
        dtype="float32",
    )
    band_1 = da.from_array(band_1, chunks=(3, -1, -1))
    tuples = [
        (np.datetime64("2000-01-01T00"), np.datetime64("2000-01-01")),
        (np.datetime64("2000-01-02T08"), np.datetime64("2000-01-02")),
        (np.datetime64("2000-01-03T12"), np.datetime64("2000-01-03")),
    ]
    index = pd.MultiIndex.from_tuples(tuples, names=["time", "solar_day"])
    coords = {
        "x": np.linspace(10, 20, band_1.shape[2]),
        "y": np.linspace(0, 5, band_1.shape[1]),
        "spec": index,
    }
    data_vars = {
        "band_1": xr.DataArray(
            band_1, dims=("spec", "y", "x"), attrs={"nodata": np.nan}
        ),
    }
    xx = xr.Dataset(data_vars=data_vars, coords=coords)
    return xx


@pytest.fixture
def consecutive_count():
    test_array = np.array(
        [
            [[1, 0, 1, 1], [1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1]],
            [[1, 255, 0, 1], [0, 0, 1, 1], [1, 255, 1, 1], [0, 0, 0, 0]],
            [[0, 0, 0, 1], [1, 1, 1, 1], [0, 0, 0, 1], [255, 0, 1, 0]],
            [[0, 0, 1, 1], [1, 255, 1, 0], [0, 0, 0, 1], [0, 1, 1, 0]],
            [[1, 0, 255, 1], [1, 0, 1, 1], [1, 0, 0, 255], [255, 0, 0, 1]],
            [[0, 1, 1, 0], [1, 0, 1, 0], [1, 255, 0, 1], [1, 1, 0, 0]],
            [[1, 1, 255, 1], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]],
            [[0, 1, 1, 255], [0, 0, 0, 0], [255, 0, 1, 1], [1, 1, 1, 0]],
            [[1, 0, 1, 0], [0, 1, 1, 1], [255, 0, 1, 0], [1, 1, 1, 1]],
            [[0, 1, 1, 1], [0, 1, 1, 255], [1, 1, 0, 255], [1, 1, 1, 1]],
            [[1, 0, 1, 1], [0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 255]],
            [[1, 1, 255, 0], [1, 255, 0, 1], [1, 1, 1, 0], [255, 0, 255, 1]],
        ]
    ).astype("uint8")

    return da.from_array(test_array, chunks=(12, -1, -1))


@pytest.fixture
def fc_wo_dataset():
    water = np.array(
        [
            [
                [3, 60, 69, 9, 73, 22, 98],
                [12, 68, 67, 24, 34, 37, 123],
                [125, 91, 116, 109, 114, 104, 19],
                [38, 66, 0, 73, 124, 125, 98],
                [115, 112, 108, 48, 122, 23, 52],
                [53, 24, 44, 80, 72, 62, 9],
                [122, 50, 25, 0, 109, 40, 93],
            ],
            [
                [17, 29, 88, 57, 103, 86, 49],
                [23, 56, 61, 27, 25, 59, 18],
                [29, 55, 83, 59, 33, 0, 6],
                [19, 74, 63, 108, 30, 13, 75],
                [106, 66, 75, 52, 71, 105, 113],
                [111, 38, 109, 3, 90, 17, 13],
                [44, 2, 34, 62, 59, 113, 128],
            ],
            [
                [36, 85, 10, 82, 75, 33, 90],
                [19, 97, 62, 15, 71, 80, 62],
                [26, 128, 87, 104, 16, 79, 41],
                [65, 77, 55, 60, 60, 72, 101],
                [2, 21, 107, 111, 83, 91, 58],
                [2, 56, 75, 104, 38, 8, 119],
                [58, 3, 69, 76, 38, 15, 16],
            ],
            [
                [24, 23, 128, 71, 19, 49, 25],
                [42, 54, 111, 1, 42, 68, 109],
                [47, 108, 85, 111, 0, 18, 80],
                [121, 98, 8, 38, 90, 31, 70],
                [114, 23, 88, 81, 41, 25, 76],
                [106, 120, 3, 70, 74, 5, 101],
                [89, 78, 72, 60, 103, 91, 87],
            ],
        ]
    ).astype(np.uint8)
    ue = np.array(
        [
            [
                [30, 30, 31, 30, 30, 30, 30],
                [29, 31, 31, 31, 29, 31, 30],
                [29, 30, 31, 31, 31, 30, 29],
                [30, 29, 31, 31, 29, 30, 29],
                [31, 31, 31, 29, 30, 31, 29],
                [31, 29, 30, 31, 29, 30, 29],
                [30, 29, 30, 30, 30, 30, 31],
            ],
            [
                [30, 31, 31, 31, 31, 31, 31],
                [31, 30, 31, 31, 30, 29, 29],
                [30, 31, 31, 30, 31, 30, 30],
                [31, 30, 30, 31, 29, 31, 30],
                [29, 29, 29, 31, 29, 29, 30],
                [29, 31, 29, 31, 31, 31, 29],
                [31, 31, 31, 29, 31, 31, 29],
            ],
            [
                [29, 31, 31, 29, 29, 29, 29],
                [29, 29, 31, 29, 29, 29, 29],
                [31, 31, 29, 29, 29, 29, 29],
                [31, 29, 29, 29, 29, 29, 29],
                [29, 29, 31, 29, 29, 31, 29],
                [31, 29, 29, 31, 29, 31, 31],
                [29, 29, 29, 29, 29, 29, 31],
            ],
            [
                [29, 31, 29, 31, 31, 29, 31],
                [31, 29, 31, 29, 29, 29, 29],
                [29, 31, 29, 29, 29, 31, 31],
                [31, 29, 29, 31, 31, 29, 29],
                [29, 29, 31, 29, 31, 29, 31],
                [29, 29, 31, 29, 29, 31, 31],
                [29, 29, 29, 29, 31, 29, 31],
            ],
        ]
    ).astype(np.uint8)
    pv = np.array(
        [
            [
                [26, 92, 77, 7, 80, 56, 69],
                [20, 46, 24, 67, 0, 42, 51],
                [39, 84, 95, 57, 81, 14, 1],
                [83, 1, 30, 63, 45, 60, 85],
                [11, 0, 14, 59, 80, 23, 12],
                [66, 77, 41, 87, 38, 10, 84],
                [96, 51, 72, 4, 84, 32, 3],
            ],
            [
                [15, 74, 72, 33, 74, 88, 17],
                [64, 17, 76, 94, 15, 57, 68],
                [50, 1, 44, 57, 78, 85, 2],
                [74, 1, 20, 14, 70, 75, 96],
                [77, 30, 11, 67, 50, 45, 80],
                [99, 92, 19, 25, 9, 73, 98],
                [68, 92, 64, 9, 80, 86, 54],
            ],
            [
                [25, 24, 72, 27, 7, 17, 38],
                [50, 11, 59, 51, 30, 66, 25],
                [43, 82, 89, 1, 23, 76, 81],
                [8, 63, 16, 41, 95, 28, 37],
                [48, 26, 20, 90, 90, 44, 99],
                [69, 91, 25, 91, 73, 37, 98],
                [56, 39, 18, 84, 61, 64, 26],
            ],
            [
                [17, 92, 68, 27, 10, 89, 21],
                [64, 0, 77, 18, 75, 50, 65],
                [10, 30, 30, 88, 53, 92, 23],
                [42, 59, 89, 73, 20, 15, 83],
                [87, 60, 77, 95, 72, 41, 33],
                [50, 32, 82, 52, 17, 87, 25],
                [76, 96, 0, 13, 46, 55, 71],
            ],
        ]
    ).astype(np.uint8)

    npv = np.array(
        [
            [
                [17, 69, 79, 0, 13, 25, 23],
                [85, 64, 55, 4, 17, 42, 48],
                [80, 5, 90, 66, 47, 30, 85],
                [80, 80, 1, 62, 93, 58, 39],
                [99, 75, 19, 50, 66, 44, 47],
                [71, 47, 45, 70, 13, 60, 89],
                [10, 46, 41, 98, 25, 6, 46],
            ],
            [
                [57, 49, 36, 8, 90, 46, 26],
                [9, 14, 56, 80, 85, 8, 97],
                [33, 86, 6, 97, 11, 70, 42],
                [55, 64, 83, 18, 87, 85, 86],
                [0, 78, 86, 73, 73, 12, 23],
                [99, 36, 86, 78, 65, 47, 54],
                [56, 27, 1, 68, 30, 17, 16],
            ],
            [
                [83, 22, 80, 8, 84, 26, 23],
                [45, 72, 48, 35, 75, 68, 53],
                [75, 19, 93, 80, 32, 5, 54],
                [6, 92, 0, 31, 93, 34, 86],
                [38, 57, 28, 52, 16, 99, 87],
                [85, 2, 28, 91, 95, 4, 65],
                [99, 63, 18, 57, 39, 57, 11],
            ],
            [
                [63, 3, 66, 57, 20, 51, 74],
                [23, 67, 37, 92, 87, 28, 94],
                [51, 60, 25, 23, 97, 97, 0],
                [41, 62, 19, 26, 88, 66, 31],
                [62, 86, 82, 71, 57, 67, 27],
                [9, 6, 36, 67, 46, 12, 19],
                [53, 1, 30, 49, 9, 75, 59],
            ],
        ]
    ).astype(np.uint8)

    bs = np.array(
        [
            [
                [77, 0, 4, 11, 74, 8, 7],
                [10, 61, 62, 42, 57, 46, 57],
                [17, 89, 32, 60, 51, 64, 65],
                [35, 40, 28, 57, 28, 2, 60],
                [69, 38, 62, 48, 45, 78, 44],
                [80, 46, 59, 38, 64, 41, 75],
                [59, 45, 31, 36, 13, 14, 6],
            ],
            [
                [63, 0, 93, 46, 49, 1, 50],
                [99, 72, 76, 62, 7, 62, 17],
                [47, 68, 58, 35, 9, 0, 52],
                [80, 23, 6, 0, 22, 0, 29],
                [33, 41, 20, 45, 42, 9, 61],
                [57, 57, 90, 3, 38, 32, 39],
                [44, 63, 22, 93, 72, 59, 40],
            ],
            [
                [60, 49, 44, 49, 27, 39, 86],
                [23, 8, 47, 76, 59, 88, 55],
                [20, 25, 83, 24, 50, 48, 79],
                [54, 30, 69, 12, 46, 45, 51],
                [52, 55, 51, 75, 36, 28, 54],
                [51, 0, 28, 70, 12, 55, 50],
                [53, 48, 78, 0, 51, 2, 2],
            ],
            [
                [37, 99, 54, 8, 75, 8, 63],
                [61, 42, 87, 68, 30, 35, 55],
                [81, 68, 25, 31, 0, 17, 19],
                [46, 16, 43, 83, 54, 0, 21],
                [41, 71, 36, 46, 32, 9, 12],
                [33, 36, 19, 62, 87, 51, 23],
                [28, 67, 26, 20, 38, 0, 52],
            ],
        ]
    ).astype(np.uint8)

    water = da.from_array(water, chunks=(4, -1, -1))
    ue = da.from_array(ue, chunks=(4, -1, -1))
    pv = da.from_array(pv, chunks=(4, -1, -1))
    npv = da.from_array(npv, chunks=(4, -1, -1))
    bs = da.from_array(bs, chunks=(4, -1, -1))

    tuples = [
        (np.datetime64("2000-01-01T00"), np.datetime64("2000-01-01")),
        (np.datetime64("2000-02-01T01"), np.datetime64("2000-02-01")),
        (np.datetime64("2000-03-02T08"), np.datetime64("2000-03-02")),
        (np.datetime64("2000-03-02T12"), np.datetime64("2000-03-02")),
    ]
    index = pd.MultiIndex.from_tuples(tuples, names=["time", "solar_day"])
    coords = {
        "x": np.linspace(10, 20, water.shape[2]),
        "y": np.linspace(0, 5, water.shape[1]),
        "spec": index,
    }
    data_vars = {
        "water": xr.DataArray(water, dims=("spec", "y", "x"), attrs={"nodata": 1}),
        "ue": xr.DataArray(ue, dims=("spec", "y", "x"), attrs={"nodata": 255}),
        "pv": xr.DataArray(pv, dims=("spec", "y", "x"), attrs={"nodata": 255}),
        "npv": xr.DataArray(npv, dims=("spec", "y", "x"), attrs={"nodata": 255}),
        "bs": xr.DataArray(bs, dims=("spec", "y", "x"), attrs={"nodata": 255}),
    }
    xx = xr.Dataset(data_vars=data_vars, coords=coords)

    return xx


@pytest.mark.parametrize("bits", [0b0000_0000, 0b0000_0100])
def test_native_transform(fc_wo_dataset, bits):
    xx = fc_wo_dataset.copy()
    xx["water"] = da.bitwise_or(xx["water"], bits)
    stats_veg = StatsVegCount()
    out_xx = stats_veg.native_transform(xx).compute()

    expected_valid = (
        np.array([0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]),
        np.array([1, 1, 3, 5, 6, 2, 6, 2, 2, 5, 6, 0, 0, 2, 3]),
        np.array([0, 3, 2, 1, 3, 5, 6, 1, 4, 5, 6, 0, 2, 4, 2]),
    )
    result = np.where(out_xx["wet"].data == out_xx["wet"].data)
    for a, b in zip(expected_valid, result):
        assert (a == b).all()

    expected_valid = (np.array([1, 2, 3]), np.array([6, 2, 0]), np.array([6, 1, 2]))
    result = np.where(out_xx["wet"].data == 1)

    for a, b in zip(expected_valid, result):
        assert (a == b).all()

    result = np.where(out_xx["pv"].data == out_xx["pv"].data)
    expected_valid = (
        np.array([0, 0, 2, 3, 3, 3]),
        np.array([1, 5, 2, 0, 2, 3]),
        np.array([0, 1, 4, 0, 4, 2]),
    )

    for a, b in zip(expected_valid, result):
        assert (a == b).all()


def test_fusing(fc_wo_dataset):
    stats_veg = StatsVegCount()
    xx = stats_veg.native_transform(fc_wo_dataset)
    xx = xx.groupby("solar_day").map(partial(StatsVegCount.fuser, None)).compute()
    valid_index = (
        np.array([0, 0, 2, 2, 2]),
        np.array([1, 5, 0, 2, 3]),
        np.array([0, 1, 0, 4, 2]),
    )
    pv_valid = np.array([20, 77, 17, 38, 89])
    npv_valid = np.array([85, 47, 63, 64, 19])
    bs_valid = np.array([10, 46, 37, 25, 43])
    i = 0
    for idx in zip(*valid_index):
        assert xx.pv.data[idx] == pv_valid[i]
        assert xx.npv.data[idx] == npv_valid[i]
        assert xx.bs.data[idx] == bs_valid[i]
        i += 1


def test_veg_or_not(fc_wo_dataset):
    stats_veg = StatsVegCount()
    xx = stats_veg.native_transform(fc_wo_dataset)
    xx = xx.groupby("solar_day").map(partial(StatsVegCount.fuser, None))
    yy = stats_veg._veg_or_not(xx).compute()
    valid_index = (
        np.array([0, 0, 1, 2, 2, 2, 2, 2]),
        np.array([1, 5, 6, 0, 0, 2, 2, 3]),
        np.array([0, 1, 6, 0, 2, 1, 4, 2]),
    )
    expected_value = np.array([1, 1, 0, 1, 0, 0, 1, 1])
    i = 0
    for idx in zip(*valid_index):
        assert yy[idx] == expected_value[i]
        i += 1


def test_water_or_not(fc_wo_dataset):
    stats_veg = StatsVegCount()
    xx = stats_veg.native_transform(fc_wo_dataset)
    xx = xx.groupby("solar_day").map(partial(StatsVegCount.fuser, None))
    yy = stats_veg._water_or_not(xx).compute()
    valid_index = (
        np.array([0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2]),
        np.array([1, 1, 3, 5, 6, 2, 6, 0, 0, 2, 2, 3, 5, 6]),
        np.array([0, 3, 2, 1, 3, 5, 6, 0, 2, 1, 4, 2, 5, 6]),
    )
    expected_value = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0])
    i = 0
    for idx in zip(*valid_index):
        assert yy[idx] == expected_value[i]
        i += 1


def test_reduce(fc_wo_dataset):
    stats_veg = StatsVegCount()
    xx = stats_veg.native_transform(fc_wo_dataset)
    xx = xx.groupby("solar_day").map(partial(StatsVegCount.fuser, None))
    xx = stats_veg.reduce(xx).compute()
    expected_value = np.array(
        [
            [1, 255, 0, 255, 255, 255, 255],
            [1, 255, 255, 255, 255, 255, 255],
            [255, 0, 255, 255, 1, 255, 255],
            [255, 255, 1, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255],
            [255, 1, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 0],
        ]
    )

    assert (xx.veg_frequency.data == expected_value).all()

    expected_value = np.array(
        [
            [0, 255, 1, 255, 255, 255, 255],
            [0, 255, 255, 0, 255, 255, 255],
            [255, 1, 255, 255, 0, 0, 255],
            [255, 255, 0, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255],
            [255, 0, 255, 255, 255, 0, 255],
            [255, 255, 255, 0, 255, 255, 1],
        ]
    )

    assert (xx.water_frequency.data == expected_value).all()


def test_consecutive_month(consecutive_count):
    stats_veg = StatsVegCount()
    xx = stats_veg._max_consecutive_months(consecutive_count, 255).compute()
    expected_value = np.array(
        [
            [2, 3, 6, 5],
            [4, 2, 6, 2],
            [3, 3, 2, 4],
            [4, 4, 3, 3],
        ]
    )
    assert (xx == expected_value).all()


def test_median_ds(dataset_md):
    xx = dataset_md.groupby("time.month").map(median_ds, dim="spec").compute()
    yy = dataset_md.groupby("time.month").median(dim="spec", skipna=True).compute()

    assert (
        (xx.band_1.data == xx.band_1.data) == (yy.band_1.data == yy.band_1.data)
    ).all()
    assert np.where(yy.band_1.data != xx.band_1.data) == (
        np.array([0]),
        np.array([0]),
        np.array([3]),
    )
