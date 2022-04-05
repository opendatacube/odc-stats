import pathlib
import pytest
from mock import MagicMock
import boto3
from moto import mock_sqs
from . import DummyPlugin

TEST_DIR = pathlib.Path(__file__).parent.absolute()
TEST_DATA_FOLDER = TEST_DIR / "data"


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
    from odc.stats.plugins import register

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
