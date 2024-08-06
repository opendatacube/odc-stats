import numpy as np
import xarray as xr
import dask.array as da
from odc.stats.plugins.lc_treelite_cultivated import (
    StatsCultivatedClass,
    generate_features,
)
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
def cultivated_model_path():
    s3_bucket = "dea-public-data-dev"
    s3_key = "lccs_models/cultivated/treelite/cultivated_treelite.so"
    local_path = "/tmp/model.so"

    # Download the model from S3
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    s3.download_file(s3_bucket, s3_key, local_path)

    yield local_path


@pytest.fixture(scope="module")
def woody_model_path():
    s3_bucket = "dea-public-data-dev"
    s3_key = "lccs_models/wcf/treelite/woody_treelite.so"
    local_path = "/tmp/model.so"

    # Download the model from S3
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    s3.download_file(s3_bucket, s3_key, local_path)

    yield local_path


@pytest.fixture(scope="module")
def cultivated_classes():
    return {"cultivated": 111, "natural": 112}


@pytest.fixture(scope="module")
def mask_bands():
    return {"classes_l3_l4": 110}


@pytest.fixture(scope="module")
def input_arrays():
    data = np.array(
        [
            [
                [
                    3.32e02,
                    1.05e02,
                    1.92e02,
                    8.7e01,
                    7.4e01,
                    6.4e01,
                    1.4010848e-02,
                    1.7472041e02,
                    2.2962935e-01,
                ],
                [
                    3.37e02,
                    1.11e02,
                    1.96e02,
                    9.7e01,
                    8.2e01,
                    7.1e01,
                    1.9971754e-02,
                    1.9717915e02,
                    2.7176377e-01,
                ],
            ],
            [
                [
                    3.33e02,
                    1.06e02,
                    1.93e02,
                    8.8e01,
                    7.6e01,
                    6.5e01,
                    1.2302631e-02,
                    1.8164543e02,
                    2.3910587e-01,
                ],
                [
                    3.4e02,
                    1.1e02,
                    1.99e02,
                    8.9e01,
                    7.7e01,
                    6.6e01,
                    1.3445648e-02,
                    1.6739888e02,
                    2.6565611e-01,
                ],
            ],
        ],
        dtype="float32",
    )

    coords = {
        "x": np.linspace(10, 20, data.shape[1]),
        "y": np.linspace(0, 5, data.shape[0]),
        "bands": [
            "nbart_blue",
            "nbart_red",
            "nbart_green",
            "nbart_nir",
            "nbart_swir_1",
            "nbart_swir_2",
            "sdev",
            "edev",
            "bcdev",
        ],
    }

    return xr.DataArray(
        data, dims=("y", "x", "bands"), coords=coords, attrs={"nodata": -999}
    )


@pytest.fixture(scope="module")
def input_datasets():
    img_1 = np.array(
        [
            [
                [
                    3.32e02,
                    1.05e02,
                    1.92e02,
                    8.7e01,
                    7.4e01,
                    6.4e01,
                    1.4010848e-02,
                    1.7472041e02,
                    2.2962935e-01,
                ],
                [
                    3.37e02,
                    1.11e02,
                    1.96e02,
                    9.7e01,
                    8.2e01,
                    7.1e01,
                    1.9971754e-02,
                    1.9717915e02,
                    2.7176377e-01,
                ],
            ],
            [
                [
                    3.33e02,
                    1.06e02,
                    1.93e02,
                    8.8e01,
                    7.6e01,
                    6.5e01,
                    1.2302631e-02,
                    1.8164543e02,
                    2.3910587e-011,
                ],
                [-999, -999, -999, -999, -999, -999, np.nan, np.nan, np.nan],
            ],
        ],
        dtype="float32",
    )

    img_1 = da.from_array(img_1, chunks=(-1, -1, -1))

    img_2 = np.array(
        [
            [
                [
                    2.74e02,
                    1.72e02,
                    1.07e02,
                    1.08e02,
                    7.0e01,
                    7.0e01,
                    1.9283541e-02,
                    1.4928413e02,
                    1.9332452e-01,
                ],
                [
                    2.74e02,
                    1.67e02,
                    9.6e01,
                    1.02e02,
                    6.6e01,
                    6.8e01,
                    1.2028454e-02,
                    1.3096216e02,
                    1.9045363e-01,
                ],
            ],
            [
                [
                    2.92e02,
                    1.84e02,
                    1.17e02,
                    1.11e02,
                    8.6e01,
                    7.0e01,
                    1.5552766e-02,
                    1.4088402e02,
                    1.9069320e-01,
                ],
                [
                    2.91e02,
                    1.8e02,
                    1.14e02,
                    1.1e02,
                    7.2e01,
                    5.7e01,
                    1.1063328e-02,
                    1.4271451e02,
                    1.7535155e-01,
                ],
            ],
        ],
        dtype="float32",
    )

    img_2 = da.from_array(img_2, chunks=(-1, -1, -1))

    img_3 = np.array([[110, 213], [110, 110]], dtype="float32")

    img_3 = da.from_array(img_3, chunks=(-1, -1))

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
            "sdev",
            "edev",
            "bcdev",
        ],
    }

    data_vars = {
        "ga_ls7": xr.DataArray(img_1, dims=("y", "x", "bands"), attrs={"nodata": -999}),
        "ga_ls8": xr.DataArray(img_2, dims=("y", "x", "bands"), attrs={"nodata": -999}),
        "classes_l3_l4": xr.DataArray(img_3, dims=("y", "x")),
    }
    xx = xr.Dataset(data_vars=data_vars, coords=coords)
    return xx


@pytest.fixture(scope="module")
def cultivated_results():
    res = [
        da.array([[1.0, 255.0], [0.0, 1.0]], dtype="uint8"),
        da.array([[0.0, 1], [0.0, 1.0]], dtype="uint8"),
    ]
    return res


@pytest.fixture(scope="module")
def cultivated_input_bands():
    return [
        "nbart_blue",
        "nbart_green",
        "nbart_red",
        "nbart_nir",
        "nbart_swir_1",
        "nbart_swir_2",
        "sdev",
        "edev",
        "bcdev",
        "classes_l3_l4",
    ]


def test_genrate_features(cultivated_input_bands, input_arrays):
    bands_indices = dict(
        zip(cultivated_input_bands, np.arange(len(cultivated_input_bands)))
    )
    res = generate_features(input_arrays.data, bands_indices)
    expected_res = np.array(
        [
            [
                [
                    0.7930565,
                    0.25081605,
                    0.1767656,
                    0.15287836,
                    0.01401085,
                    0.01747204,
                    0.22962935,
                    0.17318432,
                    0.29559875,
                    -0.22335766,
                    0.03736609,
                    0.08074532,
                    0.8050001,
                    1.774246,
                    0.17318432,
                ],
                [
                    0.7824946,
                    0.2577356,
                    0.19039929,
                    0.16485791,
                    0.01997175,
                    0.01971791,
                    0.27176377,
                    0.15025905,
                    0.2540851,
                    -0.21910109,
                    0.02351315,
                    0.08379885,
                    0.7621776,
                    1.6849853,
                    0.15025905,
                ],
            ],
            [
                [
                    0.79124624,
                    0.2518682,
                    0.18058473,
                    0.15444747,
                    0.01230263,
                    0.01816454,
                    0.23910587,
                    0.16483518,
                    0.3004947,
                    -0.22028984,
                    0.03415381,
                    0.07317075,
                    0.7977806,
                    1.7541567,
                    0.16483518,
                ],
                [
                    0.789403,
                    0.25539508,
                    0.17877656,
                    0.15323706,
                    0.01344565,
                    0.01673989,
                    0.2656561,
                    0.17647058,
                    0.3096553,
                    -0.21702127,
                    0.03745439,
                    0.07228915,
                    0.81145984,
                    1.7551421,
                    0.17647058,
                ],
            ],
        ],
        dtype="float32",
    )

    assert np.allclose(res, expected_res, rtol=1e-6, atol=1e-8)


def test_preprocess_predict_intput(
    cultivated_input_bands,
    cultivated_model_path,
    mask_bands,
    cultivated_classes,
    input_datasets,
):
    cultivated = StatsCultivatedClass(
        cultivated_classes,
        cultivated_model_path,
        mask_bands,
        input_bands=cultivated_input_bands,
    )
    res = cultivated.preprocess_predict_input(input_datasets)
    for r in res:
        assert (r[..., -1] == np.array([[1, 0], [1, 1]])).all()


def test_cultivated_predict(
    cultivated_input_bands,
    cultivated_model_path,
    mask_bands,
    cultivated_classes,
    input_datasets,
):
    cultivated = StatsCultivatedClass(
        cultivated_classes,
        cultivated_model_path,
        mask_bands,
        input_bands=cultivated_input_bands,
    )
    client.register_plugin(cultivated.dask_worker_plugin)
    imgs = cultivated.preprocess_predict_input(input_datasets)
    res = [cultivated.predict(img).compute() for img in imgs]
    assert (
        np.array(res)
        == np.array([[[1.0, 255.0], [1.0, 255.0]], [[1.0, 255.0], [1.0, 1.0]]])
    ).all()


def test_cultivated_aggregate_results(
    cultivated_input_bands,
    cultivated_model_path,
    mask_bands,
    cultivated_classes,
    cultivated_results,
):
    cultivated = StatsCultivatedClass(
        cultivated_classes,
        cultivated_model_path,
        mask_bands,
        input_bands=cultivated_input_bands,
    )
    res = cultivated.aggregate_results_from_group([cultivated_results[0]])
    assert (res.compute() == np.array([[112, 255], [111, 112]], dtype="uint8")).all()
    res = cultivated.aggregate_results_from_group(cultivated_results)
    assert (res.compute() == np.array([[111, 112], [111, 112]], dtype="uint8")).all()


def test_cultivated_reduce(
    cultivated_input_bands,
    cultivated_model_path,
    mask_bands,
    cultivated_classes,
    input_datasets,
):
    cultivated = StatsCultivatedClass(
        cultivated_classes,
        cultivated_model_path,
        mask_bands,
        input_bands=cultivated_input_bands,
    )
    client.register_plugin(cultivated.dask_worker_plugin)
    res = cultivated.reduce(input_datasets)
    assert res["cultivated_class"].attrs["nodata"] == 255
    assert (
        res["cultivated_class"].data.compute()
        == np.array([[112, 255], [112, 112]], dtype="uint8")
    ).all()
