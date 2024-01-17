import logging
from typing import (
    Iterable,
    Iterator,
    Optional,
    List,
    Any,
    Tuple,
    Union,
)
from dask.distributed import Client, WorkerPlugin
from datetime import datetime
import xarray as xr
import math
import psutil

from odc.aws.s3_client import S3Client  # TODO: move it to datacube
from .model import Task, TaskResult, TaskRunnerConfig, product_for_plugin
from .io import S3COGSink
from ._text import read_int
from .tasks import TaskReader
from .plugins import resolve
from odc.algo import wait_for_future
from datacube.utils.dask import start_local_dask
from datacube.utils.rio import configure_s3_access


Future = Any


class S3ClientPlugin(WorkerPlugin):
    def __init__(self):
        return

    def setup(self, worker):
        # Attach the client to the worker for easy access
        worker.s3_client = S3Client()


class TaskRunner:
    def __init__(
        self,
        cfg: TaskRunnerConfig,
        resolution: Optional[Tuple[float, float]] = None,
        from_sqs: Optional[str] = "",
    ):
        """ """
        _log = logging.getLogger(__name__)
        self._cfg = cfg
        self._log = _log
        self.sink = S3COGSink(
            cog_opts=cfg.cog_opts, acl=cfg.s3_acl, public=cfg.s3_public
        )

        _log.info("Resolving plugin: %s", cfg.plugin)
        mk_proc = resolve(cfg.plugin)
        self.proc = mk_proc(**cfg.plugin_config)
        self.product = product_for_plugin(
            self.proc, location=cfg.output_location, **cfg.product
        )
        _log.info("Output product: %s", self.product)

        if not from_sqs:
            _log.info("Constructing task reader: %s", cfg.filedb)
            self.rdr = TaskReader(cfg.filedb, self.product)
            _log.info("Will read from %s", self.rdr)
            if resolution is not None:
                _log.info("Changing resolution to %s, %s", resolution[0], resolution[1])
                if self.rdr.is_compatible_resolution(resolution):
                    self.rdr.change_resolution(resolution)
                else:
                    _log.error(
                        "Requested resolution is not compatible with GridSpec in '%s'",
                        cfg.filedb,
                    )
                    raise ValueError(
                        f"Requested resolution is not compatible with GridSpec \
                                in '{cfg.filedb}'"
                    )
        else:  # skip rdr and resolution compatible init
            _log.info("Skip rdr init for run from sqs: %s", cfg.filedb)
            self.rdr = TaskReader("", self.product, resolution)

        self._client = None

    def _init_dask(self) -> Client:
        cfg = self._cfg
        _log = self._log

        nthreads = cfg.threads
        if nthreads <= 0:
            nthreads = get_max_cpu()

        memory_limit: Union[str, int] = cfg.memory_limit
        if memory_limit == "":
            mem_1g = 1 << 30
            memory_limit = get_max_mem()
            if memory_limit > 2 * mem_1g:
                # leave at least a gig extra if total mem more than 2G
                memory_limit -= mem_1g

        client = start_local_dask(
            threads_per_worker=nthreads,
            processes=False,
            memory_limit=memory_limit,
        )
        aws_unsigned = self._cfg.aws_unsigned
        for c in (None, client):
            configure_s3_access(
                aws_unsigned=aws_unsigned, cloud_defaults=True, client=c
            )
        plugin = S3ClientPlugin()
        client.register_plugin(plugin)
        _log.info("Started local Dask %s", client)

        return client

    def client(self) -> Client:
        if self._client is None:
            self._client = self._init_dask()
        return self._client

    def verify_setup(self) -> bool:
        _log = self._log

        if self.product.location.startswith("s3://"):
            _log.info("Verifying credentials for output to %s", self.product.location)

            test_uri = None  # TODO: for now just check credentials are present
            if not self.sink.verify_s3_credentials(test_uri):
                _log.error("Failed to obtain S3 credentials for writing")
                return False
        return True

    # pylint: disable=import-outside-toplevel
    def tasks(
        self,
        tasks: List[str],
        ds_filters: Optional[str] = None,
    ) -> Iterator[Task]:
        from ._cli_common import parse_all_tasks

        if len(tasks) == 0:
            tiles = self.rdr.all_tiles
        else:
            # this can throw ValueError
            tiles = parse_all_tasks(tasks, self.rdr.all_tiles)

        return self.rdr.stream(tiles, ds_filters=ds_filters)

    # pylint: enable=import-outside-toplevel
    def dry_run(
        self,
        tasks: List[str],
        check_exists: bool = True,
        ds_filters: Optional[str] = None,
    ) -> Iterator[TaskResult]:
        sink = self.sink
        overwrite = self._cfg.overwrite

        # exists (None|T|F) -> str
        flag_mapping = {
            None: "",
            False: " (new)",
            True: " (recompute)" if overwrite else " (skip)",
        }

        for task in self.tasks(tasks, ds_filters=ds_filters):
            uri = sink.uri(task)
            exists = None
            if check_exists:
                exists = sink.exists(task)

            skipped = (overwrite is False) and (exists is True)
            nds = len(task.datasets)
            # TODO: take care of utc offset for day boundaries when computing ndays
            ndays = len(set(ds.center_time.date() for ds in task.datasets))
            flag = flag_mapping.get(exists, "")
            msg = f"{task.location} days={ndays:03} ds={nds:04} {uri}{flag}"

            yield TaskResult(task, uri, skipped=skipped, meta=msg)

    # pylint: disable=broad-except
    def _safe_result(self, f: Future, task: Task) -> TaskResult:
        _log = self._log
        try:
            rr = f.result()
            if rr.error is None:
                return TaskResult(task, rr.path)
            else:
                error_msg = f"Failed to write: {rr.path}"
                _log.error(error_msg)
                return TaskResult(task, rr.path, error=error_msg)
        except Exception as e:
            _log.error("Error during processing of %s %s", task.location, e)
            return TaskResult(task, error=str(e))

    # pylint: enable=broad-except
    def _register_heartbeat(self, hearbeat_filepath: str):
        """
        Records the timestamp at which a hearbeat was detected

        """
        t_now = datetime.utcnow()
        with open(f"{hearbeat_filepath}", "w", encoding="utf-8") as file_obj:
            file_obj.write(t_now.strftime("%Y-%m-%d %H:%M:%S"))

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    def _run(self, tasks: Iterable[Task], apply_eodatasets3) -> Iterator[TaskResult]:
        cfg = self._cfg
        client = self.client()
        sink = self.sink
        proc = self.proc
        _log = self._log

        for task in tasks:
            _log.info("Starting processing of %s", task.location)
            tk = task.source
            if tk is not None:
                t0 = tk.start_time
            else:
                t0 = datetime.utcnow()
            if not cfg.overwrite:
                path = sink.uri(task)
                _log.debug("Checking if can skip %s", path)
                if sink.exists(task):
                    _log.info("Skipped task @ %s", path)
                    if tk:
                        _log.info("Notifying completion via SQS")
                        tk.done()

                    yield TaskResult(task, path, skipped=True)
                    continue

            _log.debug("Building Dask Graph")
            ds = proc.reduce(
                proc.input_data(
                    task.datasets,
                    task.geobox,
                    transform_code=proc.transform_code,
                    area_of_interest=proc.area_of_interest,
                )
            )

            _log.debug("Submitting to Dask (%s)", task.location)
            ds = client.persist(ds, fifo_timeout="1ms")

            aux: Optional[xr.Dataset] = None

            # if no rgba setting in cog_ops:overrides, no rgba tif as ouput
            if "overrides" in cfg.cog_opts and "rgba" in cfg.cog_opts["overrides"]:
                rgba = proc.rgba(ds)
                if rgba is not None:
                    aux = xr.Dataset(dict(rgba=rgba))

            cog = sink.dump(task, ds, aux, proc, apply_eodatasets3)
            cog = client.compute(cog, fifo_timeout="1ms")

            _log.debug("Waiting for completion")
            cancelled = False

            for dt, _ in wait_for_future(cog, cfg.future_poll_interval, t0=t0):
                if cfg.heartbeat_filepath is not None:
                    self._register_heartbeat(cfg.heartbeat_filepath)
                if tk:
                    tk.extend_if_needed(
                        cfg.job_queue_max_lease, cfg.renew_safety_margin
                    )
                if cfg.max_processing_time > 0 and dt > cfg.max_processing_time:
                    _log.error(
                        "Task {task.location} failed to finish on time: %s>%s",
                        dt,
                        cfg.max_processing_time,
                    )
                    cancelled = True
                    cog.cancel()
                    break

            if cancelled:
                result = TaskResult(task, error="Cancelled due to timeout")
            else:
                result = self._safe_result(cog, task)

            if result:
                _log.info("Finished processing of %s", result.task.location)
                if tk:
                    _log.info("Notifying completion via SQS")
                    tk.done()
            else:
                if tk:
                    tk.cancel()

            yield result

    # pylint: enable=too-many-locals, too-many-branches, too-many-statements
    def run(
        self,
        tasks: Optional[List[str]] = None,
        sqs: Optional[str] = None,
        ds_filters: Optional[str] = None,
        apply_eodatasets3: Optional[bool] = False,
    ) -> Iterator[TaskResult]:
        cfg = self._cfg
        _log = self._log

        if tasks is not None:
            _log.info("Starting processing from task list")
            return self._run(
                self.tasks(tasks, ds_filters=ds_filters), apply_eodatasets3
            )
        if sqs is not None:
            _log.info(
                "Processing from SQS: %s, T:%s M:%s seconds",
                sqs,
                cfg.job_queue_max_lease,
                cfg.renew_safety_margin,
            )
            return self._run(
                self.rdr.stream_from_sqs(
                    sqs,
                    visibility_timeout=cfg.job_queue_max_lease,
                    ds_filters=ds_filters,
                ),
                apply_eodatasets3,
            )
        raise ValueError("Must supply one of tasks= or sqs=")


def get_max_mem() -> int:
    """
    Max available memory, takes into account pod resource allocation
    """
    total = psutil.virtual_memory().total
    mem_quota = get_mem_quota()
    if mem_quota is None:
        return total
    return min(mem_quota, total)


def get_max_cpu() -> int:
    """
    Max available CPU (rounded up if fractional), takes into account pod
    resource allocation
    """
    ncpu = get_cpu_quota()
    if ncpu is not None:
        return int(math.ceil(ncpu))
    return psutil.cpu_count()


def get_cpu_quota() -> Optional[float]:
    """
    :returns: ``None`` if unconstrained or there is an error
    :returns: maximum amount of CPU this pod is allowed to use
    """
    quota = read_int("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    if quota is None:
        return None
    period = read_int("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if period is None:
        return None
    return quota / period


def get_mem_quota() -> Optional[int]:
    """
    :returns: ``None`` if there was some error
    :returns: maximum RAM, in bytes, this pod can use according to Linux cgroups

    Note that number returned can be larger than total available memory.
    """
    return read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
