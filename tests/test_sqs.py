import json
from datetime import datetime, timedelta

import boto3
import moto
from datacube.utils.geometry import Geometry
from odc.aws.queue import get_queue, publish_message
from odc.stats._cli_publish_tasks import filter_tasks, get_geometry, publish_tasks
from odc.stats._sqs import SQSWorkToken
from odc.stats.model import OutputProduct
from odc.stats.tasks import TaskReader, render_sqs


def test_geojson(test_geom_path):
    geometry = get_geometry(test_geom_path)
    assert isinstance(geometry, Geometry)


def test_partition_area(test_db_filter_path, test_geom_path):
    TEST_GRID_NAME = "africa_30"
    geometry = get_geometry(test_geom_path)
    tasks = TaskReader(test_db_filter_path).all_tiles

    assert len(tasks) == 111

    filtered = list(filter_tasks(tasks, geometry, TEST_GRID_NAME))
    assert len(filtered) == 21


@moto.mock_aws
def test_publish_sqs(test_db_filter_path, test_geom_path):
    TEST_QUEUE_NAME = "test-queue"
    # Create an SQS queue
    sqs = boto3.resource("sqs")
    _ = sqs.create_queue(QueueName=TEST_QUEUE_NAME)

    # Test the publishing to that queue
    publish_tasks(test_db_filter_path, "", test_geom_path, False, "test-queue")


def test_sqs_work_token(sqs_message):
    tk = SQSWorkToken(sqs_message, 60)

    assert tk.active_seconds < 2
    assert tk.start_time < datetime.utcnow()
    assert tk.deadline > datetime.utcnow()

    deadline0 = tk.deadline
    assert tk.extend_if_needed(1000, 1)
    assert tk.deadline == deadline0
    assert tk.extend_if_needed(100, 60)
    assert tk.deadline > deadline0

    deadline0 = tk.deadline
    assert tk.extend(200)
    assert tk.deadline > deadline0

    tk.done()
    assert tk._msg is None
    # should be no-op
    tk.done()
    tk.cancel()
    assert tk.extend(100) is False

    tk = SQSWorkToken(sqs_message, 60)

    assert tk.active_seconds < 2
    assert tk.deadline > datetime.utcnow()
    tk.cancel()
    assert tk._msg is None
    # should be no-op
    tk.done()
    tk.cancel()
    assert tk.extend(100) is False


def test_rdr_sqs(sqs_queue_by_name, test_db_path):
    q = get_queue(sqs_queue_by_name)
    product = OutputProduct.dummy()
    rdr = TaskReader(test_db_path, product)

    for tidx in rdr.all_tiles:
        publish_message(q, json.dumps(render_sqs(tidx, test_db_path)))

    for task in rdr.stream_from_sqs(
        sqs_queue_by_name, visibility_timeout=120, max_wait=0
    ):
        _now = datetime.utcnow()
        assert task.source is not None
        assert task.source.active_seconds < 2
        assert task.source.deadline > _now
        assert task.source.deadline < _now + timedelta(seconds=120 + 10)

        task.source.extend(3600)
        assert task.source.deadline > _now
        assert task.source.deadline < _now + timedelta(seconds=3600 + 10)
        task.source.done()
