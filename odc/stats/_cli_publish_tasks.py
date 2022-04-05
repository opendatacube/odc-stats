import json
import sys
from typing import List, Optional

import click
import fsspec
import toolz
from datacube.utils.geometry import Geometry
from odc.aws.queue import get_queue, publish_messages
from odc.dscache.tools.tiling import GRIDS
from odc.stats.model import TileIdx_txy
from odc.stats.tasks import TaskReader, render_sqs

from ._cli_common import main, parse_all_tasks


def do_dry_run(tasks):
    for period, ix, iy in tasks:
        print(f"{period}/{ix:+04d}/{iy:+04d}")


def get_geometry(geojson_file: str) -> Geometry:
    with fsspec.open(geojson_file) as f:
        data = json.load(f)

    return Geometry(
        data["features"][0]["geometry"], crs=data["crs"]["properties"]["name"]
    )


def filter_tasks(tasks: List[TileIdx_txy], geometry: Geometry, grid_name: str):
    for task in tasks:
        task_geometry = GRIDS[grid_name].tile_geobox((task[1], task[2])).extent
        if task_geometry.intersects(geometry):
            yield task


def publish_tasks(
    db: str, task_filter: str, geojson_filter: Optional[str], dryrun: bool, queue: str
):
    reader = TaskReader(db)
    if len(task_filter) == 0:
        tasks = reader.all_tiles
        print(f"Found {len(tasks):,d} tasks in the file")
    else:
        try:
            tasks = parse_all_tasks(task_filter, reader.all_tiles)
            print(
                f"Found {len(tasks):,d} tasks in the file after filtering with {task_filter}"
            )
        except ValueError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)

    if geojson_filter is not None:
        geometry = get_geometry(geojson_filter)
        tasks = list(filter_tasks(tasks, geometry, reader.grid))
        print(
            f"Found {len(tasks):,d} tasks in the file after filtering with GeoJSON feature(s)"
        )

    if dryrun:
        do_dry_run(tasks)
        sys.exit(0)

    queue = get_queue(queue)

    # We assume the db files are always be the S3 uri. If they are not, there is no need to use SQS queue to process.
    messages = (
        dict(Id=str(idx), MessageBody=json.dumps(render_sqs(tidx, db)))
        for idx, tidx in enumerate(tasks)
    )

    for bunch in toolz.partition_all(10, messages):
        publish_messages(queue, bunch)


@main.command("publish-tasks")
@click.argument("db", type=str)
@click.argument("queue", type=str)
@click.option(
    "--dryrun", is_flag=True, help="Do not publish just print what would be submitted"
)
@click.option("--geojson-filter", help="GeoJSON file to use as a geometry filter")
@click.argument("task_filter", type=str, nargs=-1)
def publish_to_queue(db, queue, dryrun, geojson_filter, task_filter):
    """
    Publish tasks to SQS.

    Task filter can be one of the 3 things

    \b
    1. Comma-separated triplet: period,x,y or 'x[+-]<int>/y[+-]<int>/period
       2019--P1Y,+003,-004
       2019--P1Y/3/-4          `/` is also accepted
       x+003/y-004/2019--P1Y   is accepted as well
    2. A zero based index
    3. A slice following python convention <start>:<stop>[:<step]
        ::10 -- every tenth task: 0,10,20,..
       1::10 -- every tenth but skip first one 1, 11, 21 ..
        :100 -- first 100 tasks

    If no tasks are supplied all tasks will be published the queue.
    """

    publish_tasks(db, task_filter, geojson_filter, dryrun, queue)
