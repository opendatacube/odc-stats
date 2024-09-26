import numpy as np
import xarray as xr
import dask.array as da
from odc.stats.plugins.lc_veg_class_a1 import StatsVegClassL1
import pytest
import pandas as pd


@pytest.fixture
def dataset():
    wo_fq = np.array(
        [
            [
                [0.62672422, 0.43978265, 0.15441408, 0.04682869],
                [0.96487812, 0.27011441, 0.53843789, np.nan],
                [0.30329266, 0.31192191, 0.09097385, 0.57931479],
                [0.47824468, np.nan, 0.98643992, 0.72656195],
            ]
        ],
        dtype="float32",
    )
    wo_fq = da.from_array(wo_fq, chunks=(1, -1, -1))

    veg_fq = np.array(
        [[[0, 3, 1, 2], [0, 7, 5, 0], [0, 2, 11, 3], [11, 5, 8, 4]]], dtype="uint8"
    )
    veg_fq = da.from_array(veg_fq, chunks=(1, -1, -1))

    dem_h = np.array(
        [
            [
                [6.8908989, 2.11757315, 7.28265996, 6.10788634],
                [1.15206482, 8.04202054, 8.32279935, 1.14564906],
                [6.1375122, 6.33845174, 4.75274509, 7.67689331],
                [3.73826997, 3.73637066, 6.50784659, 7.80991549],
            ]
        ],
        dtype="float32",
    )
    dem_h = da.from_array(dem_h, chunks=(1, -1, -1))

    nidem = np.array(
        [
            [
                [0.08363985, np.nan, np.nan, 0.62890192],
                [0.86666632, 0.73258238, 0.01919135, np.nan],
                [0.67498768, np.nan, 0.27675497, 0.4076583],
                [np.nan, 0.06840416, 0.9580603, 0.10029552],
            ]
        ],
        dtype="float32",
    )
    nidem = da.from_array(nidem, chunks=(1, -1, -1))

    nbart_blue = np.array(
        [
            [
                [5529, 833, 580, 1144],
                [1172, 4680, 4999, 1746],
                [2702, 5572, 3048, 1382],
                [3080, 3149, 4080, 2463],
            ]
        ],
        dtype="int16",
    )
    nbart_blue = da.from_array(nbart_blue, chunks=(1, -1, -1))

    nbart_red = np.array(
        [
            [
                [5159, 801, 4187, 1861],
                [1123, 5827, 5080, 3464],
                [1209, 1744, 4020, 413],
                [4375, 4321, 4531, 4030],
            ]
        ],
        dtype="int16",
    )
    nbart_red = da.from_array(nbart_red, chunks=(1, -1, -1))

    nbart_green = np.array(
        [
            [
                [2798, 5539, 4431, 5996],
                [705, 2869, 4741, 4349],
                [1716, 4392, 5325, 878],
                [4174, 3233, 3368, 1118],
            ]
        ],
        dtype="int16",
    )
    nbart_green = da.from_array(nbart_green, chunks=(1, -1, -1))

    tuples = [
        (np.datetime64("2000-01-01T00"), np.datetime64("2000-01-01")),
    ]
    index = pd.MultiIndex.from_tuples(tuples, names=["time", "solar_day"])
    coords = {
        "x": np.linspace(10, 20, wo_fq.shape[2]),
        "y": np.linspace(0, 5, wo_fq.shape[1]),
        "spec": index,
    }
    data_vars = {
        "frequency": xr.DataArray(
            wo_fq, dims=("spec", "y", "x"), attrs={"nodata": np.nan}
        ),
        "veg_frequency": xr.DataArray(
            veg_fq, dims=("spec", "y", "x"), attrs={"nodata": 255}
        ),
        "dem_h": xr.DataArray(dem_h, dims=("spec", "y", "x"), attrs={"nodata": np.nan}),
        "elevation": xr.DataArray(
            nidem, dims=("spec", "y", "x"), attrs={"nodata": np.nan}
        ),
        "nbart_blue": xr.DataArray(
            nbart_blue, dims=("spec", "y", "x"), attrs={"nodata": -999}
        ),
        "nbart_red": xr.DataArray(
            nbart_red, dims=("spec", "y", "x"), attrs={"nodata": -999}
        ),
        "nbart_green": xr.DataArray(
            nbart_green, dims=("spec", "y", "x"), attrs={"nodata": -999}
        ),
    }
    xx = xr.Dataset(data_vars=data_vars, coords=coords)
    return xx


def test_l3_classes(dataset):
    stats_l3 = StatsVegClassL1(
        output_classes={
            "aquatic_veg": 124,
            "terrestrial_veg": 110,
            "water": 221,
            "intertidal": 223,
            "surface": 210,
        },
        optional_bands=["canopy_cover_class", "elevation"],
    )

    expected_res = np.array(
        [
            [
                [223, 221, 210, 124],
                [223, 223, 223, 210],
                [223, 221, 223, 223],
                [221, 223, 223, 223],
            ]
        ],
        dtype="uint8",
    )

    res, water_seasonality = stats_l3.l3_class(dataset)
    assert (res == expected_res).all()


def test_l4_water_seasonality(dataset):
    stats_l3 = StatsVegClassL1(
        output_classes={
            "aquatic_veg": 124,
            "terrestrial_veg": 110,
            "water": 221,
            "intertidal": 223,
            "surface": 210,
        },
        optional_bands=["canopy_cover_class", "elevation"],
    )

    wo_fq = np.array(
        [
            [
                [0.0, 0.021, 0.152, np.nan],
                [0.249, 0.273, 0.252, 0.0375],
                [0.302, 0.311, 0.789, 0.078],
                [0.021, 0.243, np.nan, 0.255],
            ]
        ],
        dtype="float32",
    )
    wo_fq = da.from_array(wo_fq, chunks=(1, -1, -1))

    dataset["frequency"] = xr.DataArray(
        wo_fq, dims=("spec", "y", "x"), attrs={"nodata": np.nan}
    )

    expected_water_seasonality = np.array(
        [
            [
                [0.0, 0.25, 0.25, np.nan],
                [0.25, 1.0, 1, 0.25],
                [1.0, 1.0, 1.0, 0.25],
                [0.25, 0.25, np.nan, 1.0],
            ]
        ],
        dtype="float32",
    )

    res, water_seasonality = stats_l3.l3_class(dataset)
    assert np.allclose(water_seasonality, expected_water_seasonality, equal_nan=True)


def test_reduce(dataset):
    stats_l3 = StatsVegClassL1(
        output_classes={
            "aquatic_veg": 124,
            "terrestrial_veg": 110,
            "water": 221,
            "intertidal": 223,
            "surface": 210,
        },
        optional_bands=["canopy_cover_class", "elevation"],
    )
    res = stats_l3.reduce(dataset)

    for var in res:
        assert res[var].attrs.get("nodata") is not None
        if res[var].dtype == "uint8":
            assert res[var].attrs.get("nodata") == 255
