import random
from typing import Optional, Tuple, Union, Callable, Any, Dict, List, Iterable, Iterator
from types import SimpleNamespace
from collections import namedtuple
from datetime import datetime
from itertools import islice, chain, groupby
import pickle
import json
import os
from urllib.parse import urlparse
import logging
import ciso8601
import re

from odc.dscache import DatasetCache
from datacube import Datacube
from datacube.model import Dataset, GridSpec, DatasetType
from datacube.utils.geometry import Geometry
from datacube.utils.documents import transform_object_tree
from datacube.utils.dates import normalise_dt

from odc.dscache.tools import bin_dataset_stream, ordered_dss
from odc.dscache.tools.tiling import parse_gridspec_with_name
from odc.dscache.tools.profiling import ds_stream_test_func
from ._text import split_and_check

from odc.aws import s3_download, s3_url_parse

from .model import DateTimeRange, Task, OutputProduct, TileIdx, TileIdx_txy, TileIdx_xy
from ._gjson import gs_bounds, compute_grid_info, gjson_from_tasks
from .utils import (
    bin_annual,
    bin_full_history,
    bin_generic,
    bin_seasonal,
    fuse_ds,
    fuse_products,
)

TilesRange2d = Tuple[Tuple[int, int], Tuple[int, int]]
CompressedDataset = namedtuple("CompressedDataset", ["id", "time"])


def _xy(tidx: TileIdx) -> TileIdx_xy:
    return tidx[-2:]


def compress_ds(ds: Dataset) -> CompressedDataset:
    dt = normalise_dt(ds.center_time)
    return CompressedDataset(ds.id, dt)


def is_tile_in(tidx: Tuple[int, int], tiles: TilesRange2d) -> bool:
    (x0, x1), (y0, y1) = tiles
    x, y = tidx
    return (x0 <= x < x1) and (y0 <= y < y1)


def out_path(suffix: str, base: str) -> str:
    if base.endswith(".db"):
        base = base[:-3]
    return base + suffix


def sanitize_query(query):
    def sanitize(v):
        if isinstance(v, Geometry):
            return v.json
        if isinstance(v, datetime):
            return v.isoformat()
        return v

    return transform_object_tree(sanitize, query)


def render_task(tidx: TileIdx_txy) -> str:
    period, xi, yi = tidx
    return f"{period}/{xi:02d}/{yi:02d}"


def parse_task(s: str) -> TileIdx_txy:
    sep = "/" if "/" in s else ","
    t, x, y = split_and_check(s, sep, 3)
    if t.startswith("x"):
        t, x, y = y, t, x
    return (t, int(x.lstrip("x")), int(y.lstrip("y")))


def render_sqs(tidx: TileIdx_txy, filedb: str) -> Dict[str, str]:
    """
    Add extra layer to render task. Convert it to JSON for SQS message body.
    """
    period, xi, yi = tidx
    return {"filedb": filedb, "tile_idx": f"{period}/{xi:02d}/{yi:02d}"}


def parse_sqs(s: str) -> Tuple[TileIdx_txy, str]:
    """
    Add extra layer to parse task. Convert it from JSON for SQS message body.
    """

    message_body = json.loads(s)
    filedb = message_body.get("filedb", None)

    tile_info = message_body.get("tile_idx", None)

    sep = "/" if "/" in s else ","
    t, x, y = split_and_check(tile_info, sep, 3)
    if t.startswith("x"):
        t, x, y = y, t, x
    return ((t, int(x.lstrip("x")), int(y.lstrip("y"))), filedb)


class SaveTasks:
    def __init__(
        self,
        output: str,
        grid: str,
        frequency: str = "annual",
        overwrite: bool = False,
        complevel: int = 6,
    ):

        if DatasetCache.exists(output) and overwrite is False:
            raise ValueError(f"File database already exists: {output}")

        grid, gridspec = parse_gridspec_with_name(grid)

        self._output = output
        self._overwrite = overwrite
        self._complevel = complevel
        self._grid = grid
        self._gridspec = gridspec
        self._frequency = frequency

    def out_path(self, suffix: str) -> str:
        return out_path(suffix, self._output)

    def ds_align(self, dss: Iterable, product: DatasetType, group_size: int):
        def match_dss(groups, group_size):
            for _, ds_group in groups:
                ds_group = tuple(ds_group)
                if len(ds_group) == group_size:
                    yield ds_group

        grouped_dss = groupby(
            dss,
            key=lambda ds: (ds.center_time, ds.metadata.region_code)
            if hasattr(ds.metadata, "region_code")
            else (ds.center_time,),
        )
        grouped_dss = match_dss(grouped_dss, group_size)
        map_fuse_func = lambda x: fuse_ds(*x, product=product)
        dss = map(map_fuse_func, grouped_dss)
        return dss

    def _get_dss(
        self,
        dc: Datacube,
        products: list,
        msg: Callable[[str], Any],
        dataset_filter: Optional[dict] = {},
        temporal_range: Optional[DateTimeRange] = None,
        tiles: Optional[TilesRange2d] = None,
    ):
        """
        This returns a tuple containing:
        - a generator of datasets
        - the number of datasets in the generator
        - a config dictionary containing the product, temporal range, tiles, and the datacube query used
        """

        cfg: Dict[str, Any] = dict(
            grid=self._grid,
            freq=self._frequency,
        )

        # query by a list of products is not a "officially" supported feature
        # but it is embedded in the code everywhere
        # mark it for ref
        query = dict(product=products, **dataset_filter)

        if tiles is not None:
            (x0, x1), (y0, y1) = tiles
            msg(f"Limit search to tiles: x:[{x0}, {x1}) y:[{y0}, {y1})")
            cfg["tiles"] = tiles
            query["geopolygon"] = gs_bounds(self._gridspec, tiles)

        if temporal_range is not None:
            query.update(
                temporal_range.dc_query(pad=0.6)
            )  # pad a bit more than half a day on each side
            cfg["temporal_range"] = temporal_range.short

        cfg["query"] = sanitize_query(query)

        msg("Connecting to the database, streaming datasets")
        dss = ordered_dss(
            dc,
            freq="y",
            key=lambda ds: (ds.center_time, ds.metadata.region_code)
            if hasattr(ds.metadata, "region_code")
            else (ds.center_time,),
            **query,
        )
        return dss, cfg

    def save(
        self,
        dc: Datacube,
        products: str,
        dataset_filter: Optional[dict] = {},
        temporal_range: Union[str, DateTimeRange, None] = None,
        tiles: Optional[TilesRange2d] = None,
        predicate: Optional[Callable[[Dataset], bool]] = None,
        msg: Optional[Callable[[str], Any]] = None,
        debug: bool = False,
    ) -> bool:
        """
        :param product: Product name to consume
        :param dataset_filter: Optionally apply search filter on Datasets
        :param temporal_range: Optionally  limit query in time
        :param tiles: Optionally limit query to a range of tiles
        :param predicate: If supplied filter Datasets as they come in with custom filter, Dataset->Bool
        :param msg: Observe messages if needed via callback
        :param debug: Dump some intermediate state to files for debugging
        """

        if DatasetCache.exists(self._output) and self._overwrite is False:
            raise ValueError(f"File database already exists: {self._output}")

        dt_range = SimpleNamespace(start=None, end=None)

        def _update_start_end(x, out):
            if out.start is None:
                out.start = x
                out.end = x
            else:
                out.start = min(out.start, x)
                out.end = max(out.end, x)

        def persist(ds: Dataset) -> CompressedDataset:
            _ds = compress_ds(ds)
            _update_start_end(_ds.time, dt_range)
            return _ds

        def msg_default(msg):
            pass

        if msg is None:
            msg = msg_default

        if isinstance(temporal_range, str):
            temporal_range = DateTimeRange(temporal_range)

        product_list = re.split(r"\+|-", products)
        product_list = list(filter(None, product_list))
        dss, cfg = self._get_dss(
            dc, product_list, msg, dataset_filter, temporal_range, tiles
        )
        if "+" in products:
            products = [
                dc.index.products.get_by_name(product) for product in product_list
            ]
            fused_product = fuse_products(*products)
            dss = self.ds_align(dss, fused_product, len(products))

        if predicate is not None:
            dss = filter(predicate, dss)

        dss_slice = list(islice(dss, 0, 100))
        if len(dss_slice) == 0:
            msg("found no datasets")
            return True

        if len(dss_slice) >= 100:
            msg("Training compression dictionary")
            samples = dss_slice.copy()
            random.shuffle(samples)
            zdict = DatasetCache.train_dictionary(samples, 8 * 1024)
            msg(".. done")
        else:
            zdict = None

        dss = chain(dss_slice, dss)
        cache = DatasetCache.create(
            self._output,
            zdict=zdict,
            complevel=self._complevel,
            truncate=self._overwrite,
        )
        cache.add_grid(self._gridspec, self._grid)
        cache.append_info_dict("stats/", dict(config=cfg))

        cells: Dict[Tuple[int, int], Any] = {}
        dss = cache.tee(dss)
        dss = bin_dataset_stream(self._gridspec, dss, cells, persist=persist)
        rr = ds_stream_test_func(dss)
        msg(rr.text)

        if tiles is not None:
            # prune out tiles that were not requested
            cells = {
                tidx: cell for tidx, cell in cells.items() if is_tile_in(tidx, tiles)
            }

        if temporal_range is not None:
            # Prune Datasets outside of temporal range (after correcting for UTC offset)
            for cell in cells.values():
                utc_offset = cell.utc_offset
                cell.dss = [
                    ds for ds in cell.dss if (ds.time + utc_offset) in temporal_range
                ]

        n_tiles = len(cells)
        msg(f"Total of {n_tiles:,d} spatial tiles")

        if self._frequency == "all":
            tasks = bin_full_history(cells, start=dt_range.start, end=dt_range.end)
        elif self._frequency == "semiannual":
            tasks = bin_seasonal(cells, months=6, anchor=1)
        elif self._frequency == "seasonal":
            tasks = bin_seasonal(cells, months=3, anchor=12)
        elif self._frequency == "nov-mar":
            tasks = bin_seasonal(cells, months=5, anchor=11, extract_single_season=True)
        elif self._frequency == "apr-oct":
            tasks = bin_seasonal(cells, months=7, anchor=4, extract_single_season=True)
        elif self._frequency == "annual-fy":
            tasks = bin_seasonal(cells, months=12, anchor=7)
        elif self._frequency == "annual":
            tasks = bin_annual(cells)
        elif temporal_range is not None:
            tasks = bin_generic(cells, [temporal_range])
        else:
            tasks = bin_annual(cells)
        # Remove duplicate source uuids.
        # Duplicates occur when queried datasets are captured around UTC midnight
        # and around weekly boundary
        tasks = {k: set(dss) for k, dss in tasks.items()}
        tasks_uuid = {k: [ds.id for ds in dss] for k, dss in tasks.items()}

        all_ids = set()
        for k, dss in tasks_uuid.items():
            all_ids.update(dss)
        msg(f"Total of {len(all_ids):,d} unique dataset IDs after filtering")

        if len(all_ids) == 0:
            return True

        msg(f"Saving tasks to disk ({len(tasks)})")
        cache.add_grid_tiles(self._grid, tasks_uuid)
        msg(".. done")

        self._write_info(tasks, msg, cells, debug)

        return True

    def _write_info(self, tasks, msg, cells, debug):
        csv_path = self.out_path(".csv")
        msg(f"Writing summary to {csv_path}")
        with open(csv_path, "wt") as f:
            f.write('"T","X","Y","datasets","days"\n')

            for p, x, y in sorted(tasks):
                dss = tasks[(p, x, y)]
                n_dss = len(dss)
                n_days = len(set(ds.time.date() for ds in dss))
                line = f'"{p}", {x:+05d}, {y:+05d}, {n_dss:4d}, {n_days:4d}\n'
                f.write(line)

        msg("Dumping GeoJSON(s)")
        grid_info = compute_grid_info(
            cells, resolution=max(self._gridspec.tile_size) / 4
        )
        tasks_geo = gjson_from_tasks(tasks, grid_info)
        for temporal_range, gjson in tasks_geo.items():
            fname = self.out_path(f"-{temporal_range}.geojson")
            msg(f"..writing to {fname}")
            with open(fname, "wt") as f:
                json.dump(gjson, f)

        if debug:
            pkl_path = self.out_path("-cells.pkl")
            msg(f"Saving debug info to: {pkl_path}")
            with open(pkl_path, "wb") as fb:
                pickle.dump(cells, fb)

            pkl_path = self.out_path("-tasks.pkl")
            msg(f"Saving debug info to: {pkl_path}")
            with open(pkl_path, "wb") as fb:
                pickle.dump(tasks, fb)

        return True


class TaskReader:
    def __init__(
        self,
        cache: Union[str, DatasetCache],
        product: Optional[OutputProduct] = None,
        resolution: Optional[Tuple[float, float]] = None,
    ):
        self._cache_path = None

        if len(cache) != 0 and isinstance(cache, str):
            if cache.startswith("s3://"):
                self._cache_path = s3_download(cache)
                cache = self._cache_path
            cache = DatasetCache.open_ro(cache)

            # TODO: verify this things are set in the file
            cfg = cache.get_info_dict("stats/config")
            grid = cfg["grid"]
            gridspec = cache.grids[grid]
        else:  # if read from message, there is no filedb at beginning
            cfg = {}

        self._product = product
        # if not from_sqs, the resolution check can finish before init TaskReader
        self.resolution = resolution
        self._dscache = cache
        self._cfg = cfg
        self._grid = grid if cache else ""
        self._gridspec = gridspec if cache else ""
        self._all_tiles = sorted(idx for idx, _ in cache.tiles(grid)) if cache else []

    def is_compatible_resolution(self, resolution: Tuple[float, float], tol=1e-8):
        for res, sz in zip(resolution, self._gridspec.tile_size):
            res = abs(res)
            npix = int(sz / res)
            if abs(npix * res - sz) > tol:
                return False
        return True

    def change_resolution(self, resolution: Tuple[float, float]):
        """
        Modify GridSpec to have different pixel resolution but still covering same tiles as the original.
        """
        if not self.is_compatible_resolution(resolution):
            raise ValueError(
                "Supplied resolution is not compatible with the current GridSpec"
            )
        gs = self._gridspec
        self._gridspec = GridSpec(
            gs.crs, gs.tile_size, resolution=resolution, origin=gs.origin
        )

    def init_from_sqs(self, local_db_path: str):
        """
        Adding the missing _grid, _gridspec, _gridspec and _all_tiles which skip for sqs task init.
        Upading the cfg which used placeholder filedb path for sqs task init.
        """

        _log = logging.getLogger(__name__)

        cache = DatasetCache.open_ro(local_db_path)

        # TODO: Validate this information. Assumption is the ...
        cfg = cache.get_info_dict("stats/config")
        grid = cfg["grid"]
        gridspec = cache.grids[grid]

        cfg["filedb"] = cache

        # Configure everything on the TaskReader
        self._dscache = cache
        self._cfg = cfg
        self._grid = grid
        self._gridspec = gridspec
        self._all_tiles = sorted(idx for idx, _ in cache.tiles(grid)) if cache else []

        # first time to access the filedb, then it can do the resolution check
        if self.resolution is not None:
            _log.info(
                f"Changing resolution to {self.resolution[0], self.resolution[1]}"
            )
            if self.is_compatible_resolution(self.resolution):
                self.change_resolution(self.resolution)
            else:  # if resolution has issue, stop init
                _log.error(
                    f"Requested resolution is not compatible with GridSpec in '{cfg.filedb}'"
                )
                raise ValueError(
                    f"Requested resolution is not compatible with GridSpec in '{cfg.filedb}'"
                )

    def __del__(self):
        if self._cache_path is not None:
            os.unlink(self._cache_path)

    def __repr__(self) -> str:
        grid, path, n = self._grid, str(self._dscache.path), len(self._all_tiles)
        return f"<{path}> grid:{grid} n:{n:,d}"

    def _resolve_product(self, product: Optional[OutputProduct]) -> OutputProduct:
        if product is None:
            product = self._product

        if product is None:
            raise ValueError("Product is not supplied and default is not set")
        return product

    @property
    def cache(self) -> DatasetCache:
        return self._dscache

    @property
    def grid(self) -> GridSpec:
        return self._grid

    @property
    def product(self) -> OutputProduct:
        return self._resolve_product(None)

    @property
    def all_tiles(self) -> List[TileIdx_txy]:
        return self._all_tiles

    def datasets(self, tile_index: TileIdx_txy) -> Tuple[Dataset, ...]:
        return tuple(
            ds for ds in self._dscache.stream_grid_tile(tile_index, self._grid)
        )

    def load_task(
        self,
        tile_index: TileIdx_txy,
        product: Optional[OutputProduct] = None,
        source: Any = None,
        ds_filters: Optional[str] = None,
    ) -> Task:
        product = self._resolve_product(product)

        dss = self.datasets(tile_index)
        if ds_filters is not None:
            ds_checker = DatasetChecker(ds_filters)
            dss = tuple(ds for ds in dss if ds_checker.check_dataset(ds))
        tidx_xy = _xy(tile_index)

        return Task(
            product=product,
            tile_index=tidx_xy,
            geobox=self._gridspec.tile_geobox(tidx_xy),
            time_range=DateTimeRange(tile_index[0]),
            datasets=dss,
            source=source,
        )

    def stream(
        self,
        tiles: Iterable[TileIdx_txy],
        product: Optional[OutputProduct] = None,
        ds_filters: Optional[str] = None,
    ) -> Iterator[Task]:
        product = self._resolve_product(product)
        for tidx in tiles:
            yield self.load_task(tidx, product, ds_filters=ds_filters)

    def stream_from_sqs(
        self,
        sqs_queue,
        product: Optional[OutputProduct] = None,
        visibility_timeout: int = 300,
        ds_filters: Optional[str] = None,
        **kw,
    ) -> Iterator[Task]:
        from odc.aws.queue import get_messages, get_queue
        from ._sqs import SQSWorkToken

        product = self._resolve_product(product)

        if isinstance(sqs_queue, str):
            sqs_queue = get_queue(sqs_queue)

        for msg in get_messages(sqs_queue, visibility_timeout=visibility_timeout, **kw):
            token = SQSWorkToken(msg, visibility_timeout)
            tidx, filedb = parse_sqs(msg.body)
            local_db_file = None

            # Avoid downloading the file multiple times.
            if urlparse(filedb).scheme == "s3":
                _, key = s3_url_parse(filedb)
                local_db_file = key.split("/")[-1]
                self._cache_path = local_db_file

                # Make sure we have this DB downloaded
                if not os.path.isfile(local_db_file):
                    s3_download(filedb, destination=local_db_file)
            else:
                # Assume it's a local file if it's not an S3 URL
                local_db_file = filedb

            # Initialise TaskReader information from the DB
            self.init_from_sqs(local_db_file)

            # Load the task
            yield self.load_task(tidx, product, source=token, ds_filters=ds_filters)


class DatasetChecker:
    def __init__(self, ds_filters):
        ds_filters = ds_filters.split("|")
        self.ds_filters = tuple(json.loads(ds_filter) for ds_filter in ds_filters)

    @staticmethod
    def check_dt(ds_filter, datetime_str):
        time_range = DateTimeRange(ds_filter["datetime"])
        dt = ciso8601.parse_datetime_as_naive(datetime_str)
        return dt in time_range

    def check_ds_1(self, ds_filter, ds):
        valid = True
        for key in ds_filter.keys():
            if key == "datetime":
                valid &= self.check_dt(ds_filter, ds.metadata_doc["properties"][key])
            else:
                valid &= ds_filter[key] == ds.metadata_doc["properties"][key]

        return valid

    def check_dataset(self, ds):
        valid = False
        for ds_filter in self.ds_filters:
            valid |= self.check_ds_1(ds_filter, ds)

        return valid
