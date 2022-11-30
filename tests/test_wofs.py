from functools import partial
import numpy as np
import xarray as xr
import dask.array as da
from odc.stats.plugins.wofs import StatsWofs
import pytest
import pandas as pd


@pytest.fixture
def dataset():
    band_1 = np.array(
        [
            [[0, (1 << 7)], [(1 << 7), 1], [(1 << 7), 1]],
            [[0, (1 << 7) | (1 << 6) | (1 << 5)], [0, (1 << 1)], [(1 << 7), 1]],
            [
                [(1 << 7), (1 << 7) | (1 << 6) | (1 << 5)],
                [(1 << 2) | (1 << 1), 0],
                [(1 << 1), 1],
            ],
        ]
    ).astype(np.uint8)

    band_1 = da.from_array(band_1, chunks=(3, -1, -1))

    tuples = [
        (np.datetime64("2000-01-01T00"), np.datetime64("2000-01-01")),
        (np.datetime64("2000-01-01T01"), np.datetime64("2000-01-01")),
        (np.datetime64("2000-01-02T12"), np.datetime64("2000-01-02")),
    ]
    index = pd.MultiIndex.from_tuples(tuples, names=["time", "solar_day"])
    coords = {
        "x": np.linspace(10, 20, band_1.shape[2]),
        "y": np.linspace(0, 5, band_1.shape[1]),
        "spec": index,
    }

    data_vars = {
        "water": xr.DataArray(band_1, dims=("spec", "y", "x"), attrs={"nodata": 0}),
    }
    xx = xr.Dataset(data_vars=data_vars, coords=coords)

    return xx


def test_native_transform(dataset):
    stats_wofs = StatsWofs()
    out_xx = stats_wofs.native_transform(dataset)
    out_xx.load()
    expected = np.array(
        [
            [[True, False], [False, False], [False, False]],
            [[True, False], [True, False], [False, False]],
            [[False, False], [False, True], [False, False]],
        ]
    )
    assert (out_xx["dry"].data == expected).all()

    expected = np.array(
        [
            [[False, True], [True, False], [True, False]],
            [[False, False], [False, False], [True, False]],
            [[True, False], [False, False], [False, False]],
        ]
    )
    assert (out_xx["wet"].data == expected).all()

    expected = np.array(
        [
            [[False, False], [False, False], [False, False]],
            [[False, True], [False, True], [False, False]],
            [[False, True], [True, False], [True, False]],
        ]
    )
    assert (out_xx["bad"].data == expected).all()


def test_fusing(dataset):
    stats_wofs = StatsWofs()
    out_xx = stats_wofs.native_transform(dataset)
    out_xx = out_xx.groupby("solar_day").map(partial(stats_wofs.fuser))
    out_xx.load()
    expected = np.array(
        [
            [[True, False], [True, False], [False, False]],
            [[False, False], [False, True], [False, False]],
        ]
    )
    assert (out_xx["dry"].data == expected).all()

    expected = np.array(
        [
            [[False, True], [True, False], [True, False]],
            [[True, False], [False, False], [False, False]],
        ]
    )
    assert (out_xx["wet"].data == expected).all()

    expected = np.array(
        [
            [[False, True], [False, True], [False, False]],
            [[False, True], [True, False], [True, False]],
        ]
    )
    assert (out_xx["bad"].data == expected).all()


def test_reduce(dataset):
    stats_wofs = StatsWofs()
    out_xx = stats_wofs.native_transform(dataset)
    out_xx = out_xx.groupby("solar_day").map(partial(stats_wofs.fuser))
    out_xx = stats_wofs.reduce(out_xx)
    assert out_xx.count_wet.attrs.get("nodata", 0) == -999
    assert out_xx.count_clear.attrs.get("nodata", 0) == -999
    out_xx.load()
    expected = np.array([[1, 0], [0, 0], [1, -999]])
    assert (out_xx.count_wet.data == expected).all()
    expected = np.array([[2, 0], [0, 1], [1, -999]])
    assert (out_xx.count_clear.data == expected).all()
    expected = np.array([[0.5, np.nan], [np.nan, 0.0], [1.0, np.nan]])
    assert (
        out_xx.frequency.where(~np.isnan(out_xx.frequency.data), -1)
        == np.where(~np.isnan(expected), expected, -1)
    ).all()
