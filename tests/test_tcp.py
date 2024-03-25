import numpy as np
import xarray as xr
import dask.array as da
from odc.stats.plugins.tcw_percentiles import StatsTCWPC
from odc.stats.model import product_for_plugin
from odc.stats.tasks import TaskReader

import pytest
from pathlib import Path


@pytest.fixture
def dataset():
    no_data = 0
    cloud = 2
    shadow = 3

    contiguity = np.array(
        [
            [[1, 1], [1, 1]],
            [[1, 1], [1, 1]],
            [[1, 1], [1, 0]],
        ]
    )
    band = np.array(
        [
            [[255, 57], [20, 0]],
            [[30, 10], [70, 80]],
            [[25, 1], [120, 121]],
        ]
    )
    band_fmask = np.array(
        [
            [[1, 1], [1, no_data]],
            [[cloud, no_data], [cloud, shadow]],
            [[1, 5], [no_data, 1]],
        ]
    )

    band = da.from_array(band, chunks=(3, -1, -1))
    times = [np.datetime64(f"2000-01-01T0{i}") for i in range(3)]

    coords = {
        "x": np.linspace(10, 20, band.shape[2]),
        "y": np.linspace(0, 5, band.shape[1]),
        "time": times,
    }

    data_vars = {
        "red": (("time", "y", "x"), band),
        "green": (("time", "y", "x"), band),
        "blue": (("time", "y", "x"), band),
        "nir": (("time", "y", "x"), band),
        "swir1": (("time", "y", "x"), band),
        "swir2": (("time", "y", "x"), band),
        "nbart_contiguity": (("time", "y", "x"), contiguity),
        "fmask": (("time", "y", "x"), band_fmask),
    }

    xx = xr.Dataset(data_vars=data_vars, coords=coords)
    xx["fmask"] = xx.fmask.assign_attrs(
        units=1,
        nodata=0,
        flags_definition={
            "fmask": {
                "bits": [0, 1, 2, 3, 4, 5, 6, 7],
                "values": {
                    "0": "nodata",
                    "1": "valid",
                    "2": "cloud",
                    "3": "shadow",
                    "4": "snow",
                    "5": "water",
                },
                "description": "Fmask",
            }
        },
        crs="EPSG:3577",
        grid_mapping="spatial_ref",
    )
    return xx


def test_band_names(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-southeast-2")
    # Our test data is in dea-public-data, which for now is free to read anonymously
    monkeypatch.setenv("AWS_NO_SIGN_REQUEST", "YES")

    project_root = Path(__file__).parents[1]
    data_dir = f"{project_root}/tests/data//ga_ls8c_ard_3_2015-01--P3M.db"

    stats_tcp = StatsTCWPC()
    product = product_for_plugin(stats_tcp, location="/tmp/")

    rdr = TaskReader(data_dir, product=product)
    tidx = ("2015--P1Y", 40, 8)
    task = rdr.load_task(tidx)

    # This test only requires a single dataset, which will make it run much faster
    task.datasets = task.datasets[2:3]

    xx_0_0 = stats_tcp.input_data(task.datasets, task.geobox)
    xx_0_0 = xx_0_0.sel(
        indexers={"x": slice(None, None, 100), "y": slice(None, None, 100)}
    )
    tcp = stats_tcp.reduce(xx_0_0)
    result = tcp.compute()

    expected_band_names = set(
        [
            "wet_pc_10",
            "wet_pc_50",
            "wet_pc_90",
            "bright_pc_10",
            "bright_pc_50",
            "bright_pc_90",
            "green_pc_10",
            "green_pc_50",
            "green_pc_90",
        ]
    )

    assert set(result.data_vars.keys()) == expected_band_names


def test_no_data(dataset):
    dataset = dataset.copy()
    stats_tcp = StatsTCWPC()
    xx = stats_tcp.native_transform(dataset)
    result = xx.compute()

    for band in result.data_vars.keys():
        assert result[band].attrs["nodata"] == -9999


def test_fusing(dataset):
    dataset = dataset.copy()
    stats_tcp = StatsTCWPC()
    xx = stats_tcp.native_transform(dataset)
    result = xx.compute()
    nodata = -9999
    expected_results = np.array(
        [
            [[-150, -33], [-11, nodata]],
            [[nodata, nodata], [nodata, nodata]],
            [[-14, 0], [nodata, nodata]],
        ]
    )
    assert (expected_results == result.wet.data).all()
