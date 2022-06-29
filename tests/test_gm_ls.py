import numpy as np
import xarray as xr
import dask.array as da
from pathlib import Path
import pytest

from odc.stats.model import product_for_plugin
from odc.stats.plugins.gm import StatsGMLS
from odc.stats.tasks import TaskReader


@pytest.fixture
def dataset():
    no_data = 0
    contiguity = np.array(
        [
            [[1, 1], [1, 1]],
            [[1, 1], [1, 1]],
            [[1, 1], [1, 1]],
        ]
    )
    band = np.array(
        [
            [[255, 57], [20, 0]],
            [[30, 10], [70, 80]],
            [[25, 1], [120, 0]],
        ]
    )
    band_fmask = np.array(
        [
            [[0, 0], [0, no_data]],
            [[3, no_data], [3, 3]],
            [[0, 0], [no_data, 0]],
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
        "nbart_red": (("time", "y", "x"), band),
        "nbart_green": (("time", "y", "x"), band),
        "nbart_blue": (("time", "y", "x"), band),
        "nbart_nir": (("time", "y", "x"), band),
        "nbart_swir_1": (("time", "y", "x"), band),
        "nbart_swir_2": (("time", "y", "x"), band),
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


def test_native_transform(dataset):
    _ = pytest.importorskip("hdstats")
    mask_filters = {
        "cloud": [("closing", 0), ("dilation", 0)],
        "shadow": [("closing", 0), ("dilation", 0)],
    }

    dataset = dataset.copy()
    stats_gmls = StatsGMLS(cloud_filters=mask_filters, nodata_classes=(-999,))
    xx = stats_gmls.native_transform(dataset)
    result = xx.compute()

    expected_result = np.array(
        [[[255, 57], [20, -999]], [[30, -999], [70, 80]], [[25, 1], [-999, 0]]]
    )
    assert (result == expected_result).all()


def test_result_bands_to_match_inputs(dataset):
    _ = pytest.importorskip("hdstats")
    mask_filters = {
        "cloud": [("closing", 2), ("dilation", 1)],
        "shadow": [("closing", 0), ("dilation", 5)],
    }

    dataset = dataset.copy()
    stats_gmls = StatsGMLS(cloud_filters=mask_filters, nodata_classes=(-999,))
    xx = stats_gmls.native_transform(dataset)
    result = stats_gmls.reduce(xx)

    assert set(result.data_vars.keys()) == set(
        [
            "nbart_red",
            "nbart_green",
            "nbart_blue",
            "nbart_nir",
            "nbart_swir_1",
            "nbart_swir_2",
            "sdev",
            "edev",
            "bcdev",
            "count",
        ]
    )


def test_result_aux_bands_to_match_inputs(dataset):
    _ = pytest.importorskip("hdstats")
    _ = pytest.importorskip("hdstats")
    mask_filters = {
        "cloud": [("closing", 2), ("dilation", 1)],
        "shadow": [("closing", 0), ("dilation", 5)],
    }

    dataset = dataset.copy()
    aux_names = dict(
        smad="SDEV",
        emad="EDEV",
        bcmad="BCDEV",
        count="COUNT",
    )
    stats_gmls = StatsGMLS(
        cloud_filters=mask_filters, nodata_classes=(-999,), aux_names=aux_names
    )

    xx = stats_gmls.native_transform(dataset)
    result = stats_gmls.reduce(xx)

    assert set(result.data_vars.keys()) == set(
        [
            "nbart_red",
            "nbart_green",
            "nbart_blue",
            "nbart_nir",
            "nbart_swir_1",
            "nbart_swir_2",
            "SDEV",
            "EDEV",
            "BCDEV",
            "COUNT",
        ]
    )


def test_no_cloud_buffering(monkeypatch):
    gm_ls_0_0 = StatsGMLS()
    assert gm_ls_0_0.cloud_filters is None

    mask_filters_0_0 = {
        "cloud": [("closing", 0), ("dilation", 0)],
        "shadow": [("closing", 0), ("dilation", 0)],
    }

    gm_ls_0_0 = StatsGMLS(cloud_filters=mask_filters_0_0)
    assert gm_ls_0_0.cloud_filters == mask_filters_0_0


def test_no_buffering_vs_masking(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-southeast-2")
    # Our test data is in dea-public-data, which for now is free to read anonymously
    monkeypatch.setenv("AWS_NO_SIGN_REQUEST", "YES")

    project_root = Path(__file__).parents[1]
    data_dir = f"{project_root}/tests/data//ga_ls8c_ard_3_2015-01--P3M.db"

    mask_filters_0_0 = {
        "cloud": [("closing", 0), ("dilation", 0)],
        "shadow": [("closing", 0), ("dilation", 0)],
    }

    gm_ls_0_0 = StatsGMLS(cloud_filters=mask_filters_0_0)
    product = product_for_plugin(gm_ls_0_0, location="/tmp/")

    rdr = TaskReader(data_dir, product=product)
    tidx = ("2015--P1Y", 40, 8)
    task = rdr.load_task(tidx)

    # This test only requires a single dataset, which will make it run much faster
    task.datasets = task.datasets[2:3]

    xx_0_0 = gm_ls_0_0.input_data(task.datasets, task.geobox)
    xx_0_0 = xx_0_0.sel(
        indexers={"x": slice(None, None, 100), "y": slice(None, None, 100)}
    )
    gm_0_0 = gm_ls_0_0.reduce(xx_0_0)
    result_0_0 = gm_0_0.compute()

    gm_ls_0_1 = StatsGMLS()
    xx_0_1 = gm_ls_0_1.input_data(task.datasets, task.geobox)
    xx_0_1 = xx_0_1.sel(
        indexers={"x": slice(None, None, 100), "y": slice(None, None, 100)}
    )

    gm_0_1 = gm_ls_0_1.reduce(xx_0_1)
    result_0_1 = gm_0_1.compute()

    zero_buffering_count = np.array(result_0_0.nbart_red.data == -999).flatten().sum()

    no_bufferig_count = np.array(result_0_1.nbart_red.data == -999).flatten().sum()
    print(
        "No Buffering: ",
        no_bufferig_count,
        "  Zero Buffering:  ",
        zero_buffering_count,
    )
    assert no_bufferig_count > 0
    assert no_bufferig_count == zero_buffering_count
