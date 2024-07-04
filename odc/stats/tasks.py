import random
from typing import Optional, Tuple, Union, Callable, Any, Dict, List, Iterable, Iterator
from types import SimpleNamespace
from collections import namedtuple
from datetime import datetime
from itertools import islice, chain, groupby, starmap
from itertools import product as iproduct
from functools import partial
import pickle
import json
import os
from urllib.parse import urlparse
import logging
import ciso8601
import re
import copy
import toolz

from odc.dscache import DatasetCache
from datacube import Datacube
from datacube.model import Dataset, GridSpec
from datacube.utils.geometry import Geometry
from datacube.utils.documents import transform_object_tree
from datacube.utils.dates import normalise_dt

from odc.dscache.tools import bin_dataset_stream, ordered_dss
from odc.dscache.tools.tiling import parse_gridspec_with_name
from odc.dscache.tools.profiling import ds_stream_test_func
from ._text import split_and_check

from odc.aws.s3_client import S3Client, s3_url_parse

from .model import DateTimeRange, Task, OutputProduct, TileIdx, TileIdx_txy, TileIdx_xy
from ._gjson import gs_bounds, compute_grid_info, gjson_from_tasks
from .utils import (
    bin_annual,
    bin_full_history,
    bin_generic,
    bin_seasonal,
    bin_rolling_seasonal,
    fuse_ds,
    fuse_products,
)
from ._stac_fetch import s3_fetch_dss

TilesRange2d = Tuple[Tuple[int, int], Tuple[int, int]]
CompressedDataset = namedtuple("CompressedDataset", ["id", "time"])
_log = logging.getLogger(__name__)


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


def sanitize_products_str(products_str):
    """
    split a string composed by product names and s3 paths into a list of product names
    e.g., ga_ls8-ga_ls7-s3://dea-public-data-dev/derivative/ga_ls_tc_pc_cyear_3/2-0-0/
    -> [ga_ls8, ga_ls7, s3://dea-public-data-dev/derivative/ga_ls_tc_pc_cyear_3/2-0-0/]
    rules:
    1. Any separator (`+/-`) at the start or the end is disregarded
    2. Multiple same separators between the product names are treated as one
    3. Multiple different separators between the product names is respected by left-right order
    e.g., ga_ls8+-ga_ls7 -> separator is `+` as `+` proceeds `-` from left to right
    """
    if re.search(r"s3://", products_str, flags=re.I) is None:
        pattern = re.compile(r"[\+-]{1,}")
    else:
        pattern = re.compile(r"(?<=/)[-\+]{1,}|[-\+]{1,}(?=s3)", flags=re.I)
    product_list = re.split(pattern, products_str)
    product_list = list(filter(None, product_list))
    group_size = len(re.findall(r"[\w/]{1,}\+[-\+]{0,}\w{1}", products_str))

    if len(product_list) == 1:
        # indexed: True, not: False
        return [
            (re.sub(r"^\s+|\s+$", "", p), "s3://" not in p) for p in product_list
        ], group_size
    else:
        final_list = []
        for p in product_list:
            if re.search(r"s3://", p, flags=re.I) is None:
                l, _ = sanitize_products_str(p)
                final_list += l
            else:
                final_list += [(re.sub(r"^\s+|\s+$", "", p), False)]
    return final_list, group_size


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

    @classmethod
    def ds_align(
        cls,
        dss: Iterable,
        group_size: int,
        dss_extra: Optional[Iterable] = None,
        optional_products: Optional[Iterable] = None,
        fuse_dss: bool = True,
    ):
        def pack_dss(grouped_dss, group_size):
            for _, ds in grouped_dss:
                if len(ds) >= group_size:
                    yield ds
                elif optional_products is not None:
                    # make sure only nonobligate datasets missing
                    i = 0
                    if len(ds) >= (group_size - len(optional_products)):
                        for d in ds:
                            i += (
                                1
                                if any(d.product.name in p for p in optional_products)
                                else 0
                            )
                        if (len(ds) - i) >= (group_size - len(optional_products)):
                            yield ds

        def match_dss(ds_grouped, ds_extra_grouped):
            k, ds = ds_grouped
            k_e, ds_extra = ds_extra_grouped
            # ds_extra has no time but same region_code
            if (len(k) > len(k_e)) & (k[1] == k_e[0]):
                d_c = []
                ds = tuple(ds)
                for d in ds_extra:
                    d_c += [copy.deepcopy(d)]
                    d_c[-1].center_time = k[0]
                    d_c[-1].metadata_doc["label"] = d.product.name + ds[
                        0
                    ].metadata_doc.get(
                        "label",
                        toolz.get_in(["properties", "title"], ds[0].metadata_doc),
                    ).replace(
                        ds[0].product.name, ""
                    )
                    d_c[-1].metadata_doc["properties"]["datetime"] = ds[0].metadata_doc[
                        "properties"
                    ]["datetime"]
                return (k, ds + tuple(d_c))
            else:
                return (k, ())

        def sorted_key(ds, keys=("center_time", "region_code")):
            sort_keys = ()
            for k in keys:
                if k == "center_time":
                    sort_keys += (ds.center_time,)
                elif hasattr(ds.metadata, k):
                    sort_keys += (getattr(ds.metadata, k),)
            return sort_keys

        def group_dss(dss, keys=("center_time", "region_code")):
            grouped_dss = sorted(dss, key=partial(sorted_key, keys=keys))
            grouped_dss = groupby(
                grouped_dss,
                key=partial(sorted_key, keys=keys),
            )
            for k, ds in grouped_dss:
                yield (k, tuple(ds))

        grouped_dss = group_dss(dss)

        try:
            ds0 = next(dss_extra)
        except StopIteration:
            grouped_dss = pack_dss(grouped_dss, group_size)
        else:
            # if dss_extra is non-empty
            grouped_dss_extra = chain(iter([ds0]), dss_extra)
            grouped_dss_extra = group_dss(grouped_dss_extra, keys=["region_code"])
            grouped_dss = starmap(match_dss, iproduct(grouped_dss, grouped_dss_extra))
            grouped_dss = pack_dss(grouped_dss, group_size)

        try:
            ds0 = next(grouped_dss)
        except StopIteration:
            return iter([])
        grouped_dss = chain(grouped_dss, iter([ds0]))

        if fuse_dss:
            fused_product = fuse_products(*[d.product for d in ds0])

            def map_fuse_func(x):
                return fuse_ds(*x, product=fused_product)

            dss = map(map_fuse_func, grouped_dss)
            return dss
        else:
            return grouped_dss

    @classmethod
    def _find_dss(
        cls,
        dc: Datacube,
        products: str,
        query: Dict[str, Any],
        cfg: Dict[str, Any],
        dataset_filter=None,
        predicate=None,
        fuse_dss: bool = True,
        ignore_time: Optional[Iterable] = None,
        optional_products: Optional[Iterable] = None,
    ):
        """
        query and filter the datasets with a string composed by products name
        A string joined by `-` implies union of all datasets
        A string joined by `+` implies intersect (filtered and then groupby) against time
        return a generator of datasets
        """
        # pylint:disable=too-many-locals,too-many-branches
        if dataset_filter is None:
            dataset_filter = {}

        # query by a list of products is not a "officially" supported feature
        # but it is embedded in the code everywhere
        # mark it for ref

        product_list, group_size = sanitize_products_str(products)

        indexed_products = []
        non_indexed_products = []
        for p, indexed in product_list:
            if indexed:
                indexed_products += [p]
            else:
                non_indexed_products += [p]

        if ignore_time:
            ignore_time_indexed = list(set(indexed_products) & set(ignore_time))
            indexed_products = list(set(indexed_products) - set(ignore_time))
            ignore_time_nonindexed = set(non_indexed_products) & set(ignore_time)
            non_indexed_products = list(set(non_indexed_products) - set(ignore_time))
        else:
            ignore_time_indexed = []
            ignore_time_nonindexed = []

        if indexed_products:
            query.update({"product": indexed_products, **dataset_filter})
            dss = ordered_dss(
                dc,
                freq="y",
                key=lambda ds: (
                    (ds.center_time, ds.metadata.region_code)
                    if hasattr(ds.metadata, "region_code")
                    else (ds.center_time,)
                ),
                **query,
            )
        else:
            dss = iter([])
        if ignore_time_indexed:
            query.update({"product": list(ignore_time), "time": ("1970", "2038")})
            dss_extra = ordered_dss(
                dc,
                freq="y",
                key=lambda ds: (
                    (ds.center_time, ds.metadata.region_code)
                    if hasattr(ds.metadata, "region_code")
                    else (ds.center_time,)
                ),
                **query,
            )
        else:
            dss_extra = iter([])

        if non_indexed_products:
            dss_stac = cls.create_dss_by_stac(
                non_indexed_products,
                tiles=cfg.get("tiles"),
                temporal_range=cfg.get("temporal_range"),
            )
            dss = chain(dss, dss_stac)
        if ignore_time_nonindexed:
            dss_stac_extra = cls.create_dss_by_stac(
                ignore_time_nonindexed,
                tiles=cfg.get("tiles"),
                temporal_range=None,
            )
            dss_extra = chain(dss_extra, dss_stac_extra)

        if group_size > 0:
            dss = cls.ds_align(
                dss, group_size + 1, dss_extra, optional_products, fuse_dss
            )

        if predicate is not None:
            dss = filter(predicate, dss)

        return dss

    @classmethod
    def create_dss_by_stac(
        cls,
        s3_path: List[str],
        pattern: str = "*.stac-item.json",
        tiles=None,
        temporal_range=None,
    ):

        if tiles is not None:
            glob_path = [
                "x" + str(x) + "/" + "y" + str(y) + "/" + "*"
                for x in range(*tiles[0])
                for y in range(*tiles[1])
            ]
        else:
            glob_path = ["*/*/*"]

        def filter_ds_by_time(dss, temporal_range):
            for ds in dss:
                if temporal_range.start.tzinfo is None:
                    start_time = temporal_range.start.replace(
                        tzinfo=ds.center_time.tzinfo
                    )
                else:
                    start_time = temporal_range.start
                if temporal_range.end.tzinfo is None:
                    end_time = temporal_range.end.replace(tzinfo=ds.center_time.tzinfo)
                else:
                    end_time = temporal_range.end

                if (ds.center_time >= start_time) & (ds.center_time <= end_time):
                    yield ds

        dss_stac = iter([])
        for p in s3_path:
            dss = iter([])
            for x in glob_path:
                input_glob = os.path.join(p, x, pattern)
                fetched_dss = s3_fetch_dss(input_glob)
                if temporal_range is not None:
                    fetched_dss = filter_ds_by_time(fetched_dss, temporal_range)
                dss = chain(dss, fetched_dss)
            dss_stac = chain(dss_stac, dss)

        return dss_stac

    def get_dss_by_grid(
        self,
        dc: Datacube,
        products: str,
        msg: Callable[[str], Any],
        dataset_filter=None,
        predicate=None,
        temporal_range: Optional[DateTimeRange] = None,
        tiles: Optional[TilesRange2d] = None,
        ignore_time: Optional[Iterable] = None,
        optional_products: Optional[Iterable] = None,
    ):
        """
        This returns a tuple containing:
        - a generator of datasets
        - the number of datasets in the generator
        - a config dictionary containing the product, temporal range, tiles, and the datacube query used
        """

        # pylint:disable=too-many-locals
        cfg: Dict[str, Any] = {
            "grid": self._grid,
            "freq": self._frequency,
        }

        query = {}

        if tiles is not None:
            (x0, x1), (y0, y1) = tiles
            msg(f"Limit search to tiles: x:[{x0}, {x1}) y:[{y0}, {y1})")
            cfg["tiles"] = tiles
            query["geopolygon"] = gs_bounds(self._gridspec, tiles)

        if temporal_range is not None:
            query.update(
                temporal_range.dc_query(pad=0.6)
            )  # pad a bit more than half a day on each side
            cfg["temporal_range"] = temporal_range

        msg("Connecting to the database, streaming datasets")
        dss = self._find_dss(
            dc,
            products,
            query,
            cfg,
            dataset_filter,
            predicate,
            ignore_time=ignore_time,
            optional_products=optional_products,
        )
        cfg["query"] = sanitize_query(query)
        if cfg.get("temporal_range"):
            cfg["temporal_range"] = cfg["temporal_range"].short

        return dss, cfg

    # pylint:disable=too-many-locals,too-many-branches,too-many-statements,too-many-arguments
    def save(
        self,
        dc: Datacube,
        products: str,
        dataset_filter=None,
        temporal_range: Union[str, DateTimeRange, None] = None,
        tiles: Optional[TilesRange2d] = None,
        predicate: Optional[Callable[[Dataset], bool]] = None,
        ignore_time: Optional[Iterable] = None,
        optional_products: Optional[Iterable] = None,
        msg: Optional[Callable[[str], Any]] = None,
        debug: bool = False,
    ) -> bool:
        """
        :param products: Product name to consume
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

        dss, cfg = self.get_dss_by_grid(
            dc,
            products,
            msg,
            dataset_filter,
            predicate,
            temporal_range,
            tiles,
            ignore_time,
            optional_products,
        )

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
        cache.append_info_dict("stats/", {"config": cfg})

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
                    ds for ds in cell.dss if ds.time + utc_offset in temporal_range
                ]

        n_tiles = len(cells)
        msg(f"Total of {n_tiles:,d} spatial tiles")

        if self._frequency == "all":
            if temporal_range is None:
                tasks = bin_full_history(cells, start=dt_range.start, end=dt_range.end)
            else:
                tasks = bin_generic(cells, [temporal_range])
        elif self._frequency == "semiannual":
            tasks = bin_seasonal(cells, months=6, anchor=1)
        elif self._frequency == "seasonal":
            tasks = bin_seasonal(cells, months=3, anchor=12)
        elif self._frequency == "quartely":
            tasks = bin_seasonal(cells, months=3, anchor=1)
        elif self._frequency == "3month-seasons":
            anchor = int(temporal_range.start.strftime("%m"))
            tasks = bin_seasonal(cells, months=3, anchor=anchor)
        elif self._frequency == "rolling-3months":
            tasks = bin_rolling_seasonal(
                cells, temporal_range=temporal_range, months=3, interval=1
            )
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
        # pylint:disable=too-many-locals
        csv_path = self.out_path(".csv")
        msg(f"Writing summary to {csv_path}")
        with open(csv_path, "wt", encoding="utf8") as f:
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
            with open(fname, "wt", encoding="utf8") as f:
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
        self.s3_client = S3Client()

        if len(cache) != 0 and isinstance(cache, str):
            if cache.startswith("s3://"):
                self._cache_path = self.s3_client.download(cache)
                cache = self._cache_path
            cache = DatasetCache.open_ro(cache)

            # TODO: verify this things are set in the file
            cfg = cache.get_info_dict("stats/config")
            self._grid = cfg["grid"]
            self._gridspec = cache.grids[self._grid]
            self._all_tiles = sorted(idx for idx, _ in cache.tiles(self._grid))
        else:  # if read from message, there is no filedb at beginning
            cfg = {}
            self._grid = ""
            self._gridspec = ""
            self._all_tiles = []

        self._product = product
        # if not from_sqs, the resolution check can finish before init TaskReader
        self.resolution = resolution
        self._dscache = cache
        self._cfg = cfg

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
                "Changing resolution to %s, %s", self.resolution[0], self.resolution[1]
            )
            if self.is_compatible_resolution(self.resolution):
                self.change_resolution(self.resolution)
            else:  # if resolution has issue, stop init
                _log.error(
                    "Requested resolution is not compatible with GridSpec in '%s'",
                    cfg.filedb,
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
                    self.s3_client.download(filedb, destination=local_db_file)
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
