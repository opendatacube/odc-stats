import numpy as np
import xarray as xr
import dask.array as da
from odc.stats.plugins.lc_tf_urban import StatsUrbanClass
from pathlib import Path
import pytest
import boto3
from datacube.utils.dask import start_local_dask

client = start_local_dask(n_workers=1, threads_per_worker=2)

project_root = Path(__file__).parents[1]
data_dir = f"{project_root}/tests/data/"


@pytest.fixture(scope="module")
def tflite_model_path():
    s3_bucket = "dea-public-data-dev"
    s3_key = "lccs_models/urban_models/tflite/urban_model_tf_2_16_2.tflite"
    local_path = "/tmp/model.tflite"

    # Download the model from S3
    s3 = boto3.client("s3")
    s3.download_file(s3_bucket, s3_key, local_path)

    yield local_path


@pytest.fixture(scope="module")
def output_classes():
    return {"artificial": 215, "natural": 216}


@pytest.fixture(scope="module")
def image_groups():

    img_1 = np.load(f"{data_dir}/img_1.npy")
    img_1 = da.from_array(img_1, chunks=(-1, -1, -1))
    img_2 = np.load(f"{data_dir}/img_2.npy")
    img_2 = da.from_array(img_2, chunks=(-1, -1, -1))

    coords = {
        "x": np.linspace(10, 20, img_1.shape[1]),
        "y": np.linspace(0, 5, img_1.shape[0]),
        "bands": [
            "nbart_blue",
            "nbart_red",
            "nbart_green",
            "nbart_nir",
            "nbart_swir_1",
            "nbart_swir_2",
        ],
    }
    data_vars = {
        "ga_ls7": xr.DataArray(img_1, dims=("y", "x", "bands"), attrs={"nodata": -999}),
        "ga_ls8": xr.DataArray(img_2, dims=("y", "x", "bands"), attrs={"nodata": -999}),
    }
    xx = xr.Dataset(data_vars=data_vars, coords=coords)
    return xx


def test_impute_missing_values(tflite_model_path, image_groups):
    stats_urban = StatsUrbanClass(output_classes, tflite_model_path)
    res = stats_urban.impute_missing_values_from_group(image_groups)
    expect_res = np.load(f"{data_dir}/expected_img.npy")
    assert res[0].dtype == "float32"
    assert res[1].dtype == "float32"
    assert (res[0][5:, 5:, :] == expect_res[0, 5:, 5:, :]).all()
    assert (res[0][:5, :5, :] == expect_res[1, :5, :5, :]).all()
    assert (res[1][10:15, 10:15, :] == expect_res[0, 10:15, 10:15, :]).all()
    assert (res[1][:10, :10, :] == expect_res[1, :10, :10, :]).all()
    assert (res[1][15:, 15:, :] == expect_res[1, 15:, 15:, :]).all()


def test_urban_class(tflite_model_path, image_groups):
    # test better than random for a prediction
    # check correctness in integration test
    stats_urban = StatsUrbanClass(output_classes, tflite_model_path)
    client.register_plugin(stats_urban.dask_worker_plugin)
    input_img = np.load(f"{data_dir}/expected_img.npy")
    urban_mask = []
    for img in input_img:
        img = da.from_array(img, chunks=(-1, -1, -1))
        urban_mask += [stats_urban.urban_class(img)]
    assert (np.array(urban_mask) == 0).all()
