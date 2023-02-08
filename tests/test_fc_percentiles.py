from functools import partial
import numpy as np
import xarray as xr
import dask.array as da
from odc.stats.plugins.fc_percentiles import StatsFCP
import pytest
import pandas as pd


@pytest.fixture
def dataset():
    band_1 = np.array(
        [
            [[255, 57], [20, 50], [10, 20]],
            [[30, 40], [70, 80], [20, 30]],
            [[127, 52], [73, 98], [30, 40]],
        ]
    ).astype(np.uint8)

    band_2 = np.array(
        [
            [[0, 128], [128, 0], [0, 0]],
            [[0, 0], [128, 0], [0, 0]],
            [[0, 0], [0b0110_1110, 0], [0, 0]],
        ]
    ).astype(np.uint8)

    band_3 = np.array(
        [
            [[0, 0], [0, 0], [0, 0]],
            [[0, 5], [0, 0], [0, 0]],
            [[0, 0], [0, 45], [0, 0]],
        ]
    ).astype(np.uint8)

    band_1 = da.from_array(band_1, chunks=(3, -1, -1))
    band_2 = da.from_array(band_2, chunks=(3, -1, -1))
    band_3 = da.from_array(band_3, chunks=(3, -1, -1))

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
        "band_1": xr.DataArray(
            band_1, dims=("spec", "y", "x"), attrs={"test_attr": 57}
        ),
        "water": (("spec", "y", "x"), band_2),
        "ue": xr.DataArray(band_3, dims=("spec", "y", "x")),
    }
    xx = xr.Dataset(data_vars=data_vars, coords=coords)

    return xx


@pytest.mark.parametrize("bits", [0b0000_0000, 0b0001_0000])
def test_native_transform(dataset, bits):
    xx = dataset.copy()
    xx["water"] = da.bitwise_or(xx["water"], bits)
    stats_fcp = StatsFCP()
    out_xx = stats_fcp.native_transform(xx)
    assert out_xx["band_1"].attrs["test_attr"] == 57

    expected_result = np.array(
        [
            [[255, 255], [255, 50], [10, 20]],
            [[30, 40], [255, 80], [20, 30]],
            [[127, 52], [255, 98], [30, 40]],
        ]
    )
    result = out_xx.compute()["band_1"].data
    print(result)
    assert (result == expected_result).all()

    expected_result = np.array(
        [
            [[False, True], [True, False], [False, False]],
            [[False, False], [True, False], [False, False]],
            [[False, False], [False, False], [False, False]],
        ]
    )
    result = out_xx.compute()["wet"].data
    print(result)
    assert (result == expected_result).all()

    # Test native transform with UE filter and sum 120 limit
    stats_fcp_ue30_sum120 = StatsFCP(ue_threshold=30, max_sum_limit=120)
    out_xx_ue30_sum120 = stats_fcp_ue30_sum120.native_transform(xx)
    expected_result = np.array(
        [
            [[255, 255], [255, 50], [10, 20]],
            [[30, 40], [255, 80], [20, 30]],
            [[255, 52], [255, 255], [30, 40]],
        ]
    )

    result = out_xx_ue30_sum120.compute()["band_1"].data
    print(result)
    assert (result == expected_result).all()


def test_fusing(dataset):
    stats_fcp = StatsFCP()
    xx = stats_fcp.native_transform(dataset)
    for dim in xx.dims:
        if isinstance(xx.get_index(dim), pd.MultiIndex):
            xx = xx.reset_index(dim)
    xx = xx.set_xindex("solar_day")
    xx = xx.groupby("solar_day").map(partial(StatsFCP.fuser, None))
    assert xx["band_1"].attrs["test_attr"] == 57

    expected_result = np.array(
        [
            [[30, 40], [255, 65], [15, 25]],
            [[127, 52], [255, 98], [30, 40]],
        ]
    )
    result = xx.compute()["band_1"].data

    print(result)

    assert (result == expected_result).all()

    expected_result = np.array(
        [
            [[False, True], [True, False], [False, False]],
            [[False, False], [False, False], [False, False]],
        ]
    )
    result = xx.compute()["wet"].data
    print(result)
    assert (result == expected_result).all()

    # Test fusing with UE filter and sum 120 limit
    stats_fcp_ue30_sum120 = StatsFCP(ue_threshold=30, max_sum_limit=120)
    xx_ue30_sum120 = stats_fcp_ue30_sum120.native_transform(dataset)
    for dim in xx_ue30_sum120.dims:
        if isinstance(xx_ue30_sum120.get_index(dim), pd.MultiIndex):
            xx_ue30_sum120 = xx_ue30_sum120.reset_index(dim)
    xx_ue30_sum120 = xx_ue30_sum120.set_xindex("solar_day")

    xx_ue30_sum120 = xx_ue30_sum120.groupby("solar_day").map(
        partial(StatsFCP.fuser, None)
    )
    expected_result = np.array(
        [
            [[30, 40], [255, 65], [15, 25]],
            [[255, 52], [255, 255], [30, 40]],
        ]
    )
    result = xx_ue30_sum120.compute()["band_1"].data
    print(result)
    assert (result == expected_result).all()


def test_reduce(dataset):
    stats_fcp = StatsFCP(count_valid=True)
    xx = stats_fcp.native_transform(dataset)
    for dim in xx.dims:
        if isinstance(xx.get_index(dim), pd.MultiIndex):
            xx = xx.reset_index(dim)
    xx = xx.set_xindex("solar_day")
    xx = xx.groupby("solar_day").map(partial(StatsFCP.fuser, None))
    xx = xx.compute()
    xx = stats_fcp.reduce(xx)

    result = xx.compute()["band_1_pc_10"].data
    assert (result[0, :] == 255).all()
    assert (result[1, 0] == 255).all()
    assert (result[1, 1] == 255).all()

    expected_result = np.array(
        [[1, 1], [0, 1], [1, 1]],
    )
    result = xx.compute()["qa"].data
    print(result)
    assert (result == expected_result).all()

    # Check count
    # 2 valid (1 value > 120), 2 valid (1 wet), 0 valid (3 wet), 2 valid (1 UE > 30)
    expected_result = np.array([[[2, 2], [0, 2], [2, 2]]])

    result = xx.compute()["count_valid"].data
    print(result)
    assert (result == expected_result).all()

    assert set(xx.data_vars.keys()) == set(
        ["band_1_pc_10", "band_1_pc_50", "band_1_pc_90", "qa", "count_valid"]
    )

    for band_name in xx.data_vars.keys():
        if band_name not in ["count_valid"]:
            assert xx.data_vars[band_name].dtype == np.uint8
        else:
            assert xx.data_vars[band_name].dtype == np.int16

        if band_name not in ["qa", "count_valid"]:
            assert xx[band_name].attrs["test_attr"] == 57
