from functools import partial
import numpy as np
import xarray as xr
import dask.array as da
from odc.stats.plugins.mangroves import Mangroves
import pytest
import pandas as pd


@pytest.fixture
def dataset():
    band_1 = np.array(
        [
            [[255, 57], [20, 50], [10, 15],
             [30, 40], [65, 80], [20, 39],
             [90, 52], [73, 98], [30, 40]],
        ]
    ).astype(np.uint8)

    band_2 = np.array(
        [
            [[0, 1], [2, 2], [1, 2],
             [2, 2], [2, 2], [2, 2],
             [2, 2], [2, 2], [1, 1]],
        ]
    ).astype(np.uint8)

    band_3 = np.array(
        [
            [[-1849, 0], [-1851, 0], [0, 0],
             [0, 5], [0, 0], [0, 0],
             [0, 0], [0, 45], [0, 0]],
        ]
    ).astype(np.int16)

    band_1 = da.from_array(band_1, chunks=(1, -1, -1))
    band_2 = da.from_array(band_2, chunks=(1, -1, -1))
    band_3 = da.from_array(band_3, chunks=(1, -1, -1))

    index = [np.datetime64("2000-01-01T00")]
    coords = {
        "x": np.linspace(10, 20, band_1.shape[2]),
        "y": np.linspace(0, 5, band_1.shape[1]),
        "time": index,
    }

    data_vars = {
        "pv_pc_10": xr.DataArray(
            band_1, dims=("time", "y", "x"), attrs={"nodata": 255}
        ),
        "qa": xr.DataArray(band_2, dims=("time", "y", "x")),
        "wet_pc_10": xr.DataArray(band_3, dims=("time", "y", "x"),
                                  attrs={"nodata": -9999}),
    }
    xx = xr.Dataset(data_vars=data_vars, coords=coords)

    return xx


def test_native_transform(dataset):
    mangroves = Mangroves()
    out_xx = mangroves.native_transform(dataset)
    assert (out_xx==dataset).all()

def test_reduce(dataset):
    mangroves = Mangroves()
    yy = mangroves.reduce(dataset)
    expected_results =  dataset.pv_pc_10.copy(True)
    expected_results.data = np.array([[[255,   0],
                                       [255,   2],
                                       [  0,   1],
                                       [  1,   2],
                                       [  3,   3],
                                       [  1,   2],
                                       [  3,   2],
                                       [  3,   3],
                                       [  0,   0]]], dtype=np.uint8)
    expected_results.attrs['nodata'] = 255
    expected_results = expected_results.to_dataset(name="canopy_cover_class")
    assert (yy.canopy_cover_class.dtype == np.uint8)
    assert (yy.canopy_cover_class.attrs == expected_results.canopy_cover_class.attrs)
    assert (yy == expected_results).all()