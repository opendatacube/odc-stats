import numpy as np
import xarray as xr
import dask.array as da
from odc.stats.plugins.lc_veg_class_a1 import StatsVegClassL1
from odc.stats.plugins._utils import replace_nodata_with_mode
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
        [[[0, 3, 1, 2], [0, 7, 5, 0], [0, 2, 11, 3], [11, 255, 8, 4]]], dtype="uint8"
    )
    veg_fq = da.from_array(veg_fq, chunks=(1, -1, -1))

    wet_percentage = np.array(
        [[[0, 10, 20, 30], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 80, 40]]], dtype="uint8"
    )

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
                [3080, -999, 4080, 2463],
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
                [4375, -999, 4531, 4030],
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
                [4174, -999, 3368, 1118],
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
    }
    data_vars = {
        "frequency": xr.DataArray(
            wo_fq, dims=("spec", "y", "x"), attrs={"nodata": np.nan}
        ),
        "veg_frequency": xr.DataArray(
            veg_fq, dims=("spec", "y", "x"), attrs={"nodata": 255}
        ),
        "wet_percentage": xr.DataArray(
            wet_percentage, dims=("spec", "y", "x"), attrs={"nodata": 255}
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
    xx = xx.assign_coords(xr.Coordinates.from_pandas_multiindex(index, "spec"))
    return xx


@pytest.fixture
def setup_data():
    target_value = 0
    # Case 1: Replace within smallest neighborhood (3x3)
    input_1 = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 2, 0, 1],
            [1, 3, 4, 3, 1],
            [1, 0, 2, 0, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    expected_1 = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 2, 1, 1],
            [1, 3, 4, 3, 1],
            [1, 1, 2, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )

    # Case 2: Replace after expanding to maximum neighborhood (5x5)
    input_2 = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ]
    )
    expected_2 = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
        ]
    )  # Correct propagation of '1' to the valid 5x5 neighborhood.

    # Case 3: No valid replacement (everything excluded)
    input_3 = np.array(
        [
            [5, 5, 5, 5, 5],
            [5, 0, 5, 0, 5],
            [5, 5, 5, 5, 5],
            [5, 0, 5, 0, 5],
            [5, 5, 5, 5, 5],
        ]
    )
    exclude_values_3 = [5]
    expected_3 = np.array(
        [
            [5, 5, 5, 5, 5],
            [5, 0, 5, 0, 5],
            [5, 5, 5, 5, 5],
            [5, 0, 5, 0, 5],
            [5, 5, 5, 5, 5],
        ]
    )  # Zeros remain unchanged because '5' is excluded.

    input_1 = da.from_array(input_1, chunks=(5, 5))

    input_2 = da.from_array(input_2, chunks=(5, 5))

    input_3 = da.from_array(input_3, chunks=(5, 5))

    return [
        (input_1, target_value, expected_1, None),
        (input_2, target_value, expected_2, None),
        (input_3, target_value, expected_3, exclude_values_3),
    ]


def test_replace_nodata_with_mode(setup_data):
    for input_dask_array, target_value, expected, exclude_values in setup_data:
        result = replace_nodata_with_mode(
            input_dask_array,
            target_value,
            exclude_values=exclude_values,
            neighbourhood_size=5,
        )

        assert (result.compute() == expected).all()


def test_l3_classes(dataset):
    stats_l3 = StatsVegClassL1(
        output_classes={
            "aquatic_veg_wood": 124,
            "aquatic_veg_herb": 125,
            "terrestrial_veg": 110,
            "water": 221,
            "intertidal": 223,
            "surface": 210,
        },
        optional_bands=["canopy_cover_class", "elevation"],
        measurements=["level_3_4"],
    )

    expected_res = np.array(
        [
            [
                [223, 221, 210, 125],
                [223, 223, 223, 210],
                [223, 221, 223, 223],
                [221, 223, 223, 223],
            ]
        ],
        dtype="uint8",
    )

    res = stats_l3.l3_class(dataset).compute()
    assert (res == expected_res).all()

    res = stats_l3.reduce(dataset)
    for var in res:
        assert res[var].attrs.get("nodata") is not None
        if res[var].dtype == "uint8":
            assert res[var].attrs.get("nodata") == 255
