import numpy as np
import xarray as xr
import dask.array as da
from odc.stats.plugins.gm_ls_bitmask import StatsGMLSBitmask
import pytest
import pandas as pd


@pytest.fixture
def dataset(usgs_ls8_sr_definition):
    band_red = np.array(
        [
            [[255, 57], [20, 50]],
            [[30, 0], [70, 80]],
            [[25, 52], [0, 0]],
        ]
    )
    cloud_mask = 0b0000_0000_0000_1100
    no_data = 0b0000_0000_0000_0001
    band_pq = np.array(
        [
            [[0, 0], [0, no_data]],
            [[1, 0], [0, 0]],
            [[0, cloud_mask], [0, 0]],
        ]
    )

    band_red = da.from_array(band_red, chunks=(3, -1, -1))
    band_pq = da.from_array(band_pq, chunks=(3, -1, -1))

    tuples = [
        (np.datetime64(f"2000-01-01T0{i}"), np.datetime64("2000-01-01"))
        for i in range(3)
    ]
    index = pd.MultiIndex.from_tuples(tuples, names=["time", "solar_day"])
    coords = {
        "x": np.linspace(10, 20, band_red.shape[2]),
        "y": np.linspace(0, 5, band_pq.shape[1]),
    }
    pq_flags_definition = {}
    for measurement in usgs_ls8_sr_definition["measurements"]:
        if measurement["name"] == "QA_PIXEL":
            pq_flags_definition = measurement["flags_definition"]
    attrs = dict(
        units="bit_index",
        nodata="1",
        crs="epsg:32633",
        grid_mapping="spatial_ref",
        flags_definition=pq_flags_definition,
    )

    data_vars = {
        "band_red": (("spec", "y", "x"), band_red),
        "QA_PIXEL": (("spec", "y", "x"), band_pq, attrs),
    }
    xx = xr.Dataset(data_vars=data_vars, coords=coords)
    xx = xx.assign_coords(xr.Coordinates.from_pandas_multiindex(index, "spec"))
    xx["band_red"].attrs["nodata"] = 0
    return xx


@pytest.fixture
def large_dataset(usgs_ls8_sr_definition):
    band_red = np.array(
        [
            [
                [209, 157, 90, 80, 50],
                [120, 150, 99, 80, 50],
                [122, 136, 60, 70, 50],
                [145, 120, 85, 60, 50],
                [110, 110, 82, 60, 50],
            ],
            [
                [115, 157, 90, 80, 60],
                [120, 150, 99, 80, 58],
                [122, 136, 60, 70, 59],
                [145, 120, 85, 60, 31],
                [110, 110, 82, 60, 83],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        ]
    )
    C = 0b0000_0000_0001_1100  # CLOUD_MASK
    N = 0b0000_0000_0000_0001  # NO_DATA
    band_pq = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [C, C, C, 0, 0],
                [C, C, 0, 0, 0],
                [C, C, C, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, C, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, N],
                [0, 0, 0, N, N],
                [0, 0, N, N, N],
                [0, N, N, N, N],
                [N, N, N, N, N],
            ],
        ]
    )

    band_red = da.from_array(band_red, chunks=(1, -1, -1))
    band_pq = da.from_array(band_pq, chunks=(1, -1, -1))

    tuples = [
        (np.datetime64(f"2000-01-01T0{i}"), np.datetime64("2000-01-01"))
        for i in range(3)
    ]
    index = pd.MultiIndex.from_tuples(tuples, names=["time", "solar_day"])
    coords = {
        "x": np.linspace(10, 20, band_red.shape[2]),
        "y": np.linspace(0, 5, band_pq.shape[1]),
    }
    pq_flags_definition = {}
    for measurement in usgs_ls8_sr_definition["measurements"]:
        if measurement["name"] == "QA_PIXEL":
            pq_flags_definition = measurement["flags_definition"]
    attrs = dict(
        units="bit_index",
        nodata="1",
        crs="epsg:32633",
        grid_mapping="spatial_ref",
        flags_definition=pq_flags_definition,
    )

    data_vars = {
        "band_red": (("spec", "y", "x"), band_red),
        "QA_PIXEL": (("spec", "y", "x"), band_pq, attrs),
    }
    xx = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    xx = xx.assign_coords(xr.Coordinates.from_pandas_multiindex(index, "spec"))
    xx["band_red"].attrs["nodata"] = 0
    return xx


def test_native_transform(dataset):
    gm = StatsGMLSBitmask(bands=["band_red"], offset=-0.2, scale=0.00975)

    xx = gm.native_transform(dataset)
    expected_result = np.array(
        [
            [[255, 57], [0, 0]],
            [[0, 0], [70, 80]],
            [[25, 52], [0, 0]],
        ]
    )
    result = xx.compute()["band_red"].data
    assert (result == expected_result).all()

    expected_result = np.array(
        [
            [[False, False], [False, False]],
            [[False, False], [False, False]],
            [[False, True], [False, False]],
        ]
    )
    result = xx.compute()["cloud_mask"].data
    assert (result == expected_result).all()


def test_fuser(dataset):
    gm = StatsGMLSBitmask(bands=["band_red"], offset=-0.2, scale=0.00975)

    xx = gm.native_transform(dataset)
    for dim in xx.dims:
        if isinstance(xx.get_index(dim), pd.MultiIndex):
            xx = xx.reset_index(dim)
    xx = xx.set_xindex("solar_day")

    xx = xx.groupby("solar_day").map(gm.fuser)

    expected_result = np.array(
        [[255, 57], [70, 80]],
    )
    result = xx.compute()["band_red"].data
    assert (result == expected_result).all()

    expected_result = np.array(
        [[False, True], [False, False]],
    )
    result = xx.compute()["cloud_mask"].data
    assert (result == expected_result).all()


def test_reduce(dataset):
    _ = pytest.importorskip("geomad")
    gm = StatsGMLSBitmask(
        bands=["band_red"], offset=-0.2, scale=0.00975, output_scale=100
    )

    xx = gm.native_transform(dataset)
    xx = gm.reduce(xx)

    result = xx.compute()

    assert set(xx.data_vars.keys()) == {"band_red", "smad", "emad", "bcmad", "count"}

    assert result["band_red"].dtype == np.uint16
    assert result["emad"].dtype == np.uint16

    expected_result = np.array([[116, 36], [48, 58]])
    band_red = result["band_red"].data
    assert (band_red == expected_result).all()

    expected_result = np.array([[112, 0], [0, 0]])
    band_emad = result["emad"].data
    assert (band_emad == expected_result).all()

    expected_result = np.array(
        [[2, 1], [1, 1]],
    )
    count = result["count"].data
    assert (count == expected_result).all()


def test_reduce_with_filters(large_dataset):
    _ = pytest.importorskip("geomad")
    mask_filters = [("closing", 2), ("dilation", 1)]
    gm = StatsGMLSBitmask(
        bands=["band_red"],
        filters=mask_filters,
        offset=-0.2,
        scale=0.00975,
        output_scale=100,
    )

    xx = gm.native_transform(large_dataset)
    xx = gm.reduce(xx)

    result = xx.compute()

    assert set(xx.data_vars.keys()) == {"band_red", "smad", "emad", "bcmad", "count"}

    expected_result = np.array([
        [ 92, 133,  68,  58,  33],
        [ 97, 125,  77,   0,  33],
        [ 99, 113,   0,  48,  29],
        [121,  97,  63,   0,  19],
        [ 87,  87,  60,  38,  38]
    ])
    band_red = result["band_red"].data
    assert (band_red == expected_result).all()

    expected_result = np.array([
        [ 0,  0,  0,  0,  5],
        [ 0,  0,  0,  0,  4],
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  9],
        [ 0,  0,  0,  0, 16]
    ])
    band_emad = result["emad"].data
    assert (band_emad == expected_result).all()

    expected_result = np.array(
        [
            [1, 1, 1, 2, 2],
            [1, 1, 1, 0, 2],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 0, 2],
            [1, 1, 1, 2, 2]
        ],
    )
    count = result["count"].data
    assert (count == expected_result).all()


def test_aux_result_bands_to_match_inputs(dataset):
    _ = pytest.importorskip("geomad")
    aux_names = dict(smad="SMAD", emad="EMAD", bcmad="BCMAD", count="COUNT")
    gm = StatsGMLSBitmask(
        bands=["band_red"],
        aux_names=aux_names,
        offset=-0.2,
        scale=0.00975,
        output_scale=100,
    )

    xx = gm.native_transform(dataset)
    xx = gm.reduce(xx)

    assert set(xx.data_vars.keys()) == {"band_red", "SMAD", "EMAD", "BCMAD", "COUNT"}
