from odc.stats.plugins.lc_level34 import StatsLccsLevel4
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da

import pytest


NODATA = 255

@pytest.fixture(scope="module")
def image_groups():
    l34 = np.array(
        [
            [
                [210, 210, 210],
                [210, 210, 210],
                [210, 210, 210],
                [210, 210, 210],
            ]
        ],
        dtype="uint8",
    )

    urban = np.array(
        [
            [
                [216, 216, 215],
                [216, 216, 216],
                [215, 215, 215],
                [215, 215, 215],
            ]
        ],
        dtype="uint8",
    )

    woody = np.array(
        [
            [
                [113, 113, 113],
                [113, 113, 255],
                [114, 114, 114],
                [114, 114, 255],
            ]
        ],
        dtype="uint8",
    )

    pv_pc_50 = np.array(
        [
            [
                [1, 64, 65],
                [66, 40, 41],
                [3, 61, 78],
                [4, 23, 42],
            ]
        ],
        dtype="uint8",
    )

    bs_pc_50 = np.array(
        [
            [
                [1, 64, NODATA],
                [66, 40, 41],
                [1, 40, 66],
                [NODATA, 1, 42],
            ]
        ],
        dtype="uint8",
    )

    cultivated = np.array(
        [
            [
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
            ]
        ],
        dtype="uint8",
    )

    water_frequency = np.array(
        [
            [
                [1, 3, 2],
                [4, 5, 6],
                [9, 2, 11],
                [10, 11, 12],
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
    return xx


def test_l4_classes(image_groups):
    expected_l3 = [
       [216, 216, 215],
       [216, 216, 216],
       [215, 215, 215],
       [215, 215, 215]]

    expected_l4 = [
       [95, 97, 93],
       [97, 96, 96],
       [93, 93, 93],
       [93, 93, 93]]
    stats_l4 = StatsLccsLevel4()
    ds = stats_l4.reduce(image_groups)
    
    assert (ds.level3.compute() == expected_l3).all()
    assert (ds.level4.compute() == expected_l4).all()

