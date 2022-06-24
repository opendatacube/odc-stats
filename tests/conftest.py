import pathlib
import pytest
from mock import MagicMock
import boto3
from moto import mock_sqs
from odc.stats.plugins import register
from . import DummyPlugin

TEST_DIR = pathlib.Path(__file__).parent.absolute()
TEST_DATA_FOLDER = TEST_DIR / "data"

# pylint: disable=redefined-outer-name


@pytest.fixture
def aws_env(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")


@pytest.fixture
def test_db_path():
    return str(TEST_DATA_FOLDER / "test_tiles.db")


@pytest.fixture
def test_geom_path():
    return str(TEST_DATA_FOLDER / "testing_extent.geojson")


@pytest.fixture
def test_db_filter_path():
    return str(TEST_DATA_FOLDER / "test_tiles_filter.db")


@pytest.fixture
def dummy_plugin_name():
    name = "dummy-plugin"
    register(name, DummyPlugin)
    return name


@pytest.fixture
def sqs_message():
    response = {
        "ResponseMetadata": {
            "RequestId": "45ff2253-2bfe-5395-9f14-7af67a6b8f27",
            "HTTPStatusCode": 200,
            "HTTPHeaders": {
                "x-amzn-requestid": "45ff2253-2bfe-5395-9f14-7af67a6b8f27",
                "date": "Tue, 16 Feb 2021 04:51:33 GMT",
                "content-type": "text/xml",
                "content-length": "215",
            },
            "RetryAttempts": 0,
        }
    }

    msg = MagicMock()
    msg.delete = lambda: response
    msg.change_visibility = lambda VisibilityTimeout=0: response
    msg.body = ""
    return msg


@pytest.fixture
def sqs_queue_by_name(aws_env):
    qname = "test-sqs"
    with mock_sqs():
        sqs = boto3.resource("sqs")
        sqs.create_queue(QueueName=qname)

        yield qname


@pytest.fixture
def usgs_ls8_sr_definition():
    definition = {
        "name": "ls8_sr",
        "description": "USGS Landsat 8 Collection 2 Level-2 \
                        Surface Reflectance",
        "metadata_type": "eo3",
        "measurements": [
            {
                "name": "QA_PIXEL",
                "dtype": "uint16",
                "units": "bit_index",
                "nodata": "1",
                "flags_definition": {
                    "snow": {
                        "bits": 5,
                        "values": {"0": "not_high_confidence", "1": "high_confidence"},
                    },
                    "clear": {"bits": 6, "values": {"0": False, "1": True}},
                    "cloud": {
                        "bits": 3,
                        "values": {"0": "not_high_confidence", "1": "high_confidence"},
                    },
                    "water": {
                        "bits": 7,
                        "values": {"0": "land_or_cloud", "1": "water"},
                    },
                    "cirrus": {
                        "bits": 2,
                        "values": {"0": "not_high_confidence", "1": "high_confidence"},
                    },
                    "nodata": {"bits": 0, "values": {"0": False, "1": True}},
                    "cloud_shadow": {
                        "bits": 4,
                        "values": {"0": "not_high_confidence", "1": "high_confidence"},
                    },
                    "dilated_cloud": {
                        "bits": 1,
                        "values": {"0": "not_dilated", "1": "dilated"},
                    },
                    "cloud_confidence": {
                        "bits": [8, 9],
                        "values": {"0": "none", "1": "low", "2": "medium", "3": "high"},
                    },
                    "cirrus_confidence": {
                        "bits": [14, 15],
                        "values": {
                            "0": "none",
                            "1": "low",
                            "2": "reserved",
                            "3": "high",
                        },
                    },
                    "snow_ice_confidence": {
                        "bits": [12, 13],
                        "values": {
                            "0": "none",
                            "1": "low",
                            "2": "reserved",
                            "3": "high",
                        },
                    },
                    "cloud_shadow_confidence": {
                        "bits": [10, 11],
                        "values": {
                            "0": "none",
                            "1": "low",
                            "2": "reserved",
                            "3": "high",
                        },
                    },
                },
            }
        ],
    }
    return definition
