import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da

from odc.stats.plugins.lc_level3 import StatsLccsLevel3
import pytest

expected_l3_classes = [
    [111, 112, 215],
    [124, 112, 215],
    [221, 215, 216],
    [223, 255, 223],
]


@pytest.fixture(scope="module")
def image_groups():
    l34 = np.array(
        [
            [
                [110, 110, 210],
                [124, 110, 210],
                [221, 210, 210],
                [223, 255, 223],
            ]
        ],
        dtype="uint8",
    )

    urban = np.array(
        [
            [
                [215, 215, 215],
                [216, 216, 215],
                [116, 215, 216],
                [216, 216, 216],
            ]
        ],
        dtype="uint8",
    )

    cultivated = np.array(
        [
            [
                [111, 112, 255],
                [255, 112, 255],
                [255, 255, 255],
                [255, 255, 255],
            ]
        ],
        dtype="uint8",
    )

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
            da.from_array(l34, chunks=(1, -1, -1)),
            dims=("spec", "y", "x"),
            attrs={"nodata": 255},
        ),
        "urban_classes": xr.DataArray(
            da.from_array(urban, chunks=(1, -1, -1)),
            dims=("spec", "y", "x"),
            attrs={"nodata": 255},
        ),
        "cultivated_class": xr.DataArray(
            da.from_array(cultivated, chunks=(1, -1, -1)),
            dims=("spec", "y", "x"),
            attrs={"nodata": 255},
        ),
    }
    xx = xr.Dataset(data_vars=data_vars, coords=coords)
    return xx


def test_urban_class(image_groups):

    lc_level3 = StatsLccsLevel3()
    level3_classes = lc_level3.reduce(image_groups)
    assert (level3_classes.level3_class.values == expected_l3_classes).all()
