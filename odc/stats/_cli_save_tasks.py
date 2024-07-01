import json

import click
import sys
import logging
from datacube import Datacube
from .tasks import SaveTasks
from .model import DateTimeRange

from ._cli_common import main, click_range2d, click_yaml_cfg, setup_logging

CONFIG_ITEMS = [
    "grid",
    "frequency",
    "complevel",
    "overwrite",
    "gqa",
    "input_products",
    "dataset_filter",
    "ignore_time",
    "optional_products",
]


@main.command("save-tasks")
@click.option(
    "--grid",
    type=str,
    help=(
        "Grid name or spec: au-{10|20|30|60},africa-{10|20|30|60},"
        "albers-au-25 (legacy one) 'crs;pixel_resolution;shape_in_pixels'"
    ),
    default=None,
)
@click.option(
    "--year",
    type=int,
    help=(
        "Only extract datasets for a given year."
        "This is a shortcut for --temporal-range=<int>--P1Y"
    ),
)
@click.option(
    "--temporal-range",
    type=str,
    help=(
        "Only extract datasets for a given time range,"
        "Example '2020-05--P1M' month of May 2020"
    ),
)
@click.option(
    "--frequency",
    type=str,
    help=(
        "Specify temporal binning: "
        "annual|annual-fy|semiannual|seasonal|quartely|3month-seasons|rolling-3months|nov-mar|apr-oct|all"
    ),
)
@click.option("--env", "-E", type=str, help="Datacube environment name")
@click.option(
    "-z",
    "complevel",
    type=int,
    default=None,
    help="Compression setting for zstandard 1-fast, 9+ good but slow",
)
@click.option(
    "--overwrite", is_flag=True, default=None, help="Overwrite output if it exists"
)
@click.option(
    "--tiles", help='Limit query to tiles example: "0:3,2:4"', callback=click_range2d
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    hidden=True,
    help="Dump debug data to pickle",
)
@click.option(
    "--gqa",
    type=float,
    help="Only save datasets that pass `gqa_iterative_mean_xy <= gqa` test",
)
@click.option(
    "--usgs-collection-category",
    type=str,
    help=(
        "Only save datasets that pass "
        "`collection_category == usgs_collection_category` test"
    ),
)
@click.option(
    "--dataset-filter",
    type=str,
    default=None,
    help='Filter to apply on datasets - {"collection_category": "T1"}',
)
@click.option(
    "--ignore-time",
    multiple=True,
    default=None,
    help="Ignore time for particular products in input, e.g., --ignore-time ga_srtm_dem1sv1_0",
)
@click.option(
    "--optional-products",
    multiple=True,
    default=None,
    help="Allow the products to be optional and not present for every tile, "
    "e.g., --optional-products ga_ls_mangrove_cover_cyear_3",
)
@click_yaml_cfg("--config", help="Save tasks Config")
@click.option("--input-products", type=str, default="")
@click.argument("output", type=str, nargs=1, default="")
# pylint: disable=too-many-arguments, too-many-locals
# pylint: disable=too-many-branches, too-many-statements
def save_tasks(
    config,
    grid,
    year,
    temporal_range,
    frequency,
    output,
    input_products,
    dataset_filter,
    env,
    complevel,
    overwrite,
    tiles=None,
    debug=False,
    gqa=None,
    usgs_collection_category=None,
    ignore_time=None,
    optional_products=None,
):
    """
    Prepare tasks for processing (query db).

    <todo more help goes here>

    \b
    Not yet implemented features:
      - output

    """
    setup_logging()

    _log = logging.getLogger(__name__)
    if temporal_range is not None and year is not None:
        print("Can only supply one of --year or --temporal_range", file=sys.stderr)
        sys.exit(1)

    if config is None:
        config = {}

    _cfg = {k: config.get(k) for k in CONFIG_ITEMS if config.get(k) is not None}

    print(f"config from yaml {_cfg} {complevel}")

    cfg_from_cli = {
        k: v
        for k, v in {
            "grid": grid,
            "frequency": frequency,
            "gqa": gqa,
            "input_products": input_products,
            "complevel": complevel,
            "dataset_filter": dataset_filter,
            "overwrite": overwrite,
            "ignore_time": ignore_time,
            "optional_products": optional_products,
        }.items()
        if v
    }

    _log.info("Config overrides: %s", cfg_from_cli)
    _cfg.update(cfg_from_cli)
    _cfg.setdefault("complevel", 6)
    _log.info("Using config: %s", _cfg)

    gqa = _cfg.pop("gqa", None)
    input_products = _cfg.pop("input_products", None)
    dataset_filter = _cfg.pop("dataset_filter", None)
    ignore_time = _cfg.pop("ignore_time", None)
    optional_products = _cfg.pop("optional_products", None)

    if input_products is None:
        print("Input products has to be specified", file=sys.stderr)
        sys.exit(1)

    if _cfg.get("grid") is None:
        print(
            "grid must  be  one of au-{10|20|30|60}, africa-{10|20|30|60}, \
             albers_au_25 (legacy one) or custom like 'epsg:3857;30;5000' \
             (30m pixels 5,000 per side in epsg:3857) ",
            file=sys.stderr,
        )
        sys.exit(1)

    if _cfg.get("frequency") is not None:
        if _cfg.get("frequency") not in (
            "annual",
            "annual-fy",
            "semiannual",
            "seasonal",
            "quartely",
            "3month-seasons",
            "rolling-3months",
            "nov-mar",
            "apr-oct",
            "all",
        ):
            print(
                f"""Frequency must be one of annual|annual-fy|semiannual|seasonal|
                quartely|3month-seasons|rolling-3months|nov-mar|apr-oct|all
                and not '{frequency}'""",
                file=sys.stderr,
            )
            sys.exit(1)

    if temporal_range is not None:
        try:
            temporal_range = DateTimeRange(temporal_range)
        except ValueError:
            print(
                f"Failed to parse supplied temporal_range: '{temporal_range}'",
                file=sys.stderr,
            )
            sys.exit(1)

    if year is not None:
        temporal_range = DateTimeRange.year(year)

    if output == "":
        if temporal_range is not None:
            output = f"{input_products}_{temporal_range.short}.db"
        else:
            output = f"{input_products}_all.db"

    try:
        tasks = SaveTasks(output, **_cfg)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    def on_message(msg):
        print(msg)

    def gqa_predicate(ds):
        return ds.metadata.gqa_iterative_mean_xy <= gqa

    def collection_category_predicate(ds):
        if ds.type.name in ["ls5_sr", "ls7_sr", "ls8_sr", "ls9_sr"]:
            return ds.metadata.collection_category == usgs_collection_category
        else:
            return True

    predicate = None
    # These two are exclusive. GQA is from DEA, whereas collection_category is from USGS
    if gqa is not None:
        predicate = gqa_predicate
    if usgs_collection_category is not None:
        predicate = collection_category_predicate

    ds_filter = {}
    if dataset_filter:
        ds_filter = json.loads(dataset_filter)

    dc = Datacube(env=env)
    try:
        ok = tasks.save(
            dc,
            input_products,
            dataset_filter=ds_filter,
            temporal_range=temporal_range,
            tiles=tiles,
            predicate=predicate,
            debug=debug,
            ignore_time=ignore_time,
            optional_products=optional_products,
            msg=on_message,
        )
    except ValueError as e:
        print(str(e))
        sys.exit(2)

    if not ok:
        # exit with error code, failure message was already printed
        sys.exit(3)
