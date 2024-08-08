import numpy as np
import xarray as xr
import dask.array as da
from odc.stats.plugins.lc_tf_urban import StatsUrbanClass
from pathlib import Path
import pytest
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from datacube.utils.dask import start_local_dask

client = start_local_dask(n_workers=1, threads_per_worker=2)

project_root = Path(__file__).parents[1]
data_dir = f"{project_root}/tests/data/"


@pytest.fixture(scope="module")
def dask_client():
    client = start_local_dask(n_workers=1, threads_per_worker=2)
    yield client
    client.close()


@pytest.fixture(scope="module")
def tflite_model_path():
    s3_bucket = "dea-public-data-dev"
    s3_key = "lccs_models/urban_models/tflite/urban_model_tf_2_16_2.tflite"
    local_path = "/tmp/model.tflite"

    # Download the model from S3
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    s3.download_file(s3_bucket, s3_key, local_path)

    yield local_path


@pytest.fixture(scope="module")
def output_classes():
    return {"artificial": 215, "natural": 216}


@pytest.fixture(scope="module")
def urban_masks():
    return [
        da.array([[0, 0, 0], [1, 255, 0], [0, 0, 1]], dtype="uint8"),
        da.array([[255, 0, 0], [0, 255, 1], [1, 0, 0]], dtype="uint8"),
    ]


@pytest.fixture(scope="module")
def image_groups():

    img_1 = np.array(
        [
            [
                [-999, -999, -999, -999, -999, -999],
                [491, 878, 315, 324, 820, 610],
                [134, 178, 458, 55, 832, 684],
                [896, 345, 392, 755, 742, 752],
            ],
            [
                [707, 980, 767, 665, 101, 229],
                [352, 410, 176, 400, 72, 722],
                [0, 858, 629, 121, 662, 477],
                [891, 934, 766, 929, 626, 561],
            ],
            [
                [19, 586, 496, 964, 869, 389],
                [447, 325, 609, 366, 490, 457],
                [706, 156, 950, 171, 848, 994],
                [474, 100, 985, 277, 579, 289],
            ],
            [
                [186, 365, 275, 109, 800, 927],
                [365, 509, 872, 288, 390, 262],
                [200, 503, 323, 566, 861, 659],
                [796, 117, 4, 814, 631, 789],
            ],
        ],
        dtype="int16",
    )

    img_1 = da.from_array(img_1, chunks=(-1, -1, -1))
    img_2 = np.array(
        [
            [
                [772, 115, 814, 44, 951, 824],
                [8, 602, 170, 331, 117, 483],
                [121, 112, 124, 172, 704, 388],
                [741, 588, 289, 665, 320, 303],
            ],
            [
                [846, 126, 357, 805, 192, 380],
                [875, 880, 446, 458, 116, 828],
                [672, 290, 795, 727, 746, 967],
                [170, 813, 471, 885, 919, 944],
            ],
            [
                [63, 101, 718, 772, 313, 637],
                [618, 576, 254, 541, 138, 13],
                [403, 248, 891, 169, 164, 132],
                [830, 66, 87, 129, 703, 476],
            ],
            [
                [532, 620, 928, 617, 630, 666],
                [275, 253, 586, 604, 662, 948],
                [532, 807, 59, 505, 210, 149],
                [-999, -999, -999, -999, -999, -999],
            ],
        ],
        dtype="int16",
    )
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


def test_impute_missing_values(output_classes, tflite_model_path, image_groups):
    stats_urban = StatsUrbanClass(output_classes, tflite_model_path)
    res = stats_urban.impute_missing_values_from_group(image_groups)
    assert res[0].dtype == "float32"
    assert res[1].dtype == "float32"
    assert (res[0][0, 0, :] == res[1][0, 0, :]).all()
    assert (res[0][3, 3, :] == res[1][3, 3, :]).all()
    assert (res[0][1:, 1:, :] == image_groups["ga_ls7"][1:, 1:, :]).all()
    assert (res[1][:3, :3, :] == image_groups["ga_ls8"][:3, :3, :]).all()


def test_urban_class(output_classes, tflite_model_path, image_groups, dask_client):
    # test better than random for a prediction
    # check correctness in integration test
    stats_urban = StatsUrbanClass(output_classes, tflite_model_path)
    dask_client.register_plugin(stats_urban.dask_worker_plugin)
    input_img = stats_urban.impute_missing_values_from_group(image_groups)
    input_img[0][1, 1, :] = np.nan
    input_img[1][1, 1, :] = np.nan
    for img in input_img:
        urban_mask = stats_urban.urban_class(img)
        urban_mask = urban_mask.compute()
        assert (urban_mask[1, 1] == 255).all()
        assert (
            urban_mask[np.where(urban_mask < 255)[0], np.where(urban_mask < 255)[1]]
            == 0
        ).all()


def test_aggregate_results_from_group(output_classes, tflite_model_path, urban_masks):
    stats_urban = StatsUrbanClass(output_classes, tflite_model_path)
    res = stats_urban.aggregate_results_from_group([urban_masks[0]])
    expected_res = np.array(
        [[216, 216, 216], [215, 255, 216], [216, 216, 215]], dtype="uint8"
    )
    assert (res == expected_res).all()
    res = stats_urban.aggregate_results_from_group(urban_masks)
    expected_res = np.array(
        [[216, 216, 216], [215, 255, 215], [215, 216, 215]], dtype="uint8"
    )
    assert (res == expected_res).all()
