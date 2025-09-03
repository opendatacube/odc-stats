"""
Various I/O adaptors
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING, cast
from hashlib import sha1
from collections import namedtuple
from collections.abc import Callable, Iterable, Sequence

import json
from urllib.parse import urlparse
import logging
import dask
from dask.delayed import Delayed
from pathlib import Path
import xarray as xr
import io
from rasterio.crs import CRS
from numpy import datetime64

from eodatasets3.assemble import serialise
from eodatasets3.scripts.tostac import json_fallback
from eodatasets3.model import DatasetDoc
from eodatasets3.images import FileWrite, GridSpec
import eodatasets3.stac as eo3stac
from importlib.metadata import version

from datacube import Datacube
from datacube.testutils.io import native_geobox
from pyproj import aoi, transformer

from odc.geo.geobox import GeoBox
from odc.geo.geobox import pad as gbox_pad
from odc.geo.xr import xr_reproject

from ._grouper import group_by_nothing, solar_offset
from odc.algo._masking import (
    _max_fuser,
    _nodata_fuser,
    _or_fuser,
    enum_to_bool,
    mask_cleanup,
)

from odc.aws.s3_client import S3Client
from dask.distributed import get_worker
from datacube.utils.dask import save_blob_to_file
from odc.geo.cog import to_cog

if TYPE_CHECKING:
    from .plugins import StatsPluginInterface
    from .model import Task
    from datacube.model import Dataset

WriteResult = namedtuple("WriteResult", ["path", "sha1", "error"])

_log = logging.getLogger(__name__)
DEFAULT_COG_OPTS = {
    "compress": "deflate",
    "zlevel": 6,
    "blocksize": 512,
}


def dump_json(meta: dict[str, Any]) -> str:
    return json.dumps(meta, separators=(",", ":"))


def mk_sha1(data):
    if isinstance(data, str):
        data = data.encode("utf8")
    return sha1(data).hexdigest()


_dask_sha1 = dask.delayed(mk_sha1, name="sha1")


@dask.delayed
def _pack_write_result(write_out, sha1_data):
    path, ok = write_out
    if ok:
        return WriteResult(path, sha1_data, None)
    else:
        return WriteResult(path, sha1_data, "Failed Write")


@dask.delayed(name="sha1-digest")
def _sha1_digest(*write_results):
    lines = []
    for wr in write_results:
        if wr.error is not None:
            raise OSError(f"Failed to write for: {wr.path}")
        file = wr.path.split("/")[-1]
        lines.append(f"{wr.sha1}  {file}\n")
    return "".join(lines)


@dask.delayed(name="xarray-to-list")
def _xarray_to_list(image, dest_shape):
    # apply the OWS styling, the return image must have red, green, blue and alpha.
    display_pixels = []
    for display_band in ["red", "green", "blue"]:
        display_pixels.append(image[display_band].values.reshape(dest_shape))
    return display_pixels


@dask.delayed(name="save_with_s3_client")
def save_with_s3_client(data, url, with_deps=None, **kw):
    worker = get_worker()
    s3_client = worker.s3_client
    return s3_client.dump(data, url, with_deps=with_deps, **kw)


class S3COGSink:
    def __init__(
        self,
        cog_opts: dict[str, Any] | None = None,
        acl: str | None = None,
        public: bool = False,
        band_ext: str = "tif",
    ):
        """
        :param creds: S3 write credentials
        :param cog_opts: Configure compression settings, globally and per-band
        :param acl: Canned ACL string:
                    private|public-read|public-read-write|authenticated-read|
                    aws-exec-read|bucket-owner-read|bucket-owner-full-control
        :param public: Mark objects as public access, same as `acl="public-read"`

        Example of COG config

        .. code-block:: python

           # - Lossless compression for all bands but rgba
           # - Webp compression for rgba band
           cfg = {
               "compression": "deflate",
               "zlevel": 9,
               "blocksize": 1024,
               "overrides": {"rgba": {"compression": "webp", "webp_level": 80}},
           }


        """

        if cog_opts is None:
            cog_opts = {**DEFAULT_COG_OPTS}
        else:
            tmp = {**DEFAULT_COG_OPTS}
            tmp.update(cog_opts)
            cog_opts = tmp

        cog_opts_per_band = cast(
            dict[str, dict[str, Any]], cog_opts.pop("overrides", {})
        )
        per_band_cfg = {k: v for k, v in cog_opts.items() if isinstance(v, dict)}
        if per_band_cfg:
            for k in per_band_cfg:
                cog_opts.pop(k)
            cog_opts_per_band.update(per_band_cfg)

        if acl is None and public:
            acl = "public-read"

        self.s3_client = S3Client()
        self._cog_opts = cog_opts
        self._cog_opts_per_band = cog_opts_per_band
        self._stac_meta_ext = "stac-item.json"
        self._odc_meta_ext = "odc-metadata.yaml"
        self._proc_info_ext = "proc-info.yaml"
        self._stac_meta_contentype = "application/json"
        self._odc_meta_contentype = "text/yaml"
        self._prod_info_meta_contentype = "text/yaml"
        self._band_ext = band_ext
        self._acl = acl

    def uri(self, task: Task) -> str:
        return task.metadata_path("absolute", ext=self._stac_meta_ext)

    def verify_s3_credentials(self, test_uri: str | None = None) -> bool:
        if test_uri is None:
            return True
        rr = self._write_blob(b"verifying S3 permissions", test_uri).compute()
        assert rr.path == test_uri
        return rr.error is None

    # pylint: disable=invalid-name
    def _write_blob(
        self, data, url: str, ContentType: str | None = None, with_deps=None
    ) -> Delayed:
        """
        Returns Delayed WriteResult[path, sha1, error=None]
        """
        _u = urlparse(url)
        sha1_data = _dask_sha1(data)

        if _u.scheme == "s3":
            kw = {}
            if ContentType is not None:
                kw["ContentType"] = ContentType
            if self._acl is not None:
                kw["ACL"] = self._acl

            return _pack_write_result(
                save_with_s3_client(data, url, with_deps=with_deps, **kw), sha1_data
            )
        elif _u.scheme == "file":
            _dir = Path(_u.path).parent
            if not _dir.exists():
                _dir.mkdir(parents=True, exist_ok=True)
            return _pack_write_result(
                save_blob_to_file(data, _u.path, with_deps=with_deps), sha1_data
            )
        else:
            raise ValueError(f"Don't know how to save to '{url}'")

    # pylint: enable=invalid-name
    def _ds_to_cog(self, ds: xr.Dataset, paths: dict[str, str]) -> list[Delayed]:
        out = []
        for band, dv in ds.data_vars.items():
            band = str(band)
            url = paths.get(band, None)
            if url is None:
                raise ValueError(f"No path for band: '{band}'")
            cog_opts = self.cog_opts(band)
            cog_bytes = to_cog(dv, **cog_opts)
            out.append(self._write_blob(cog_bytes, url, ContentType="image/tiff"))
        return out

    def _apply_color_ramp(
        self, ds: xr.Dataset, ows_style_dict: dict, time: datetime64
    ) -> Delayed:
        from datacube_ows.styles.api import StandaloneStyle
        from datacube_ows.styles.api import apply_ows_style

        ows_style = StandaloneStyle(ows_style_dict)
        # assign the time to xr.Dataset cause ows needs it
        dst = ds.expand_dims(dim={"time": [time]})
        img = dask.delayed(apply_ows_style)(ows_style, dst)
        return img

    def _get_thumbnail(
        self, display_pixels: list, input_geobox: GridSpec, odc_file_path: str
    ) -> Delayed:
        thumbnail_bytes = dask.delayed(FileWrite().create_thumbnail_from_numpy)(
            rgb=display_pixels,
            static_stretch=(0, 255),
            out_scale=10,
            input_geobox=input_geobox,
            nodata=None,
        )

        return self._write_blob(
            thumbnail_bytes,
            odc_file_path.split(".")[0] + "_thumbnail.jpg",
            ContentType="image/jpeg",
        )

    def _ds_to_thumbnail_cog(self, ds: xr.Dataset, task: Task) -> list[Delayed]:
        odc_file_path = task.metadata_path("absolute", ext=self._odc_meta_ext)

        thumbnail_cogs = []

        input_geobox = GridSpec(
            shape=task.geobox.shape,
            transform=task.geobox.transform,
            crs=CRS.from_epsg(task.geobox.crs.to_epsg()),
        )

        if task.product.preview_image_ows_style:
            _log.info("Generate thumbnail")
            try:
                image = self._apply_color_ramp(
                    ds, task.product.preview_image_ows_style, task.time_range.start
                )
            except AttributeError as e:
                _log.error(
                    "%s Cannot parse OWS styling: %s.",
                    e,
                    task.product.preview_image_ows_style,
                )
            except ImportError as e:
                raise type(e)(
                    str(e)
                    + '. Please run python -m pip install "odc-stats[ows]" to \
                    setup environment to generate thumbnail.'
                )
            else:
                display_pixels = _xarray_to_list(image, task.geobox.shape[0:2])
                thumbnail_cog = self._get_thumbnail(
                    display_pixels, input_geobox, odc_file_path
                )
                thumbnail_cogs.append(thumbnail_cog)

        return thumbnail_cogs

    def cog_opts(self, band_name: str = "") -> dict[str, Any]:
        opts = dict(self._cog_opts)
        opts.update(self._cog_opts_per_band.get(band_name, {}))
        return opts

    def write_cog(self, da: xr.DataArray, url: str) -> Delayed:
        cog_bytes = to_cog(da, **self.cog_opts(str(da.name)))
        return self._write_blob(cog_bytes, url, ContentType="image/tiff")

    def exists(self, task: Task | str) -> bool:
        if isinstance(task, str):
            uri = task
        else:
            uri = self.uri(task)
        _u = urlparse(uri)
        if _u.scheme == "s3":
            meta = self.s3_client.head_object(uri)
            return meta is not None
        elif _u.scheme == "file":
            return Path(_u.path).exists()
        else:
            raise ValueError(f"Can't handle url: {uri}")

    def get_eo3_stac_meta(
        self, task: Task, meta: DatasetDoc, stac_file_path: str, odc_file_path: str
    ) -> str:
        """
        Convert the eodatasets3 DatasetDoc to stac meta format string.
        The stac_meta is Python dict, please use json_fallback() to format it.
        Also pass dataset_location to convert all accessories to full url.
        The S3 and local dir will use different ways to extract.
        """
        _u = urlparse(stac_file_path)

        if _u.scheme == "s3":
            dataset_location = f"{_u.scheme}://{_u.netloc}/{_u.path}"
        else:
            dataset_location = str(Path(_u.path).parent)

        stac_meta = eo3stac.to_stac_item(
            dataset=meta,
            stac_item_destination_url=stac_file_path,
            dataset_location=dataset_location,
            odc_dataset_metadata_url=odc_file_path,
            explorer_base_url=task.product.explorer_path,
        )
        return json.dumps(
            stac_meta, default=json_fallback
        )  # stac_meta is Python str, but content is 'Dict format'

    def dump_with_pystac(
        self,
        task: Task,
        proc: StatsPluginInterface,
        ds: xr.Dataset,
        aux: xr.Dataset | None = None,
    ) -> Delayed:
        """
        Dump files with STAC metadata file, which generated from PySTAC
        """
        json_url = task.metadata_path("absolute", ext=self._stac_meta_ext)
        meta = task.render_metadata(
            ext=self._band_ext, use_center_time=getattr(proc, "CENTER_TIMERANGE", False)
        )
        json_data = dump_json(meta).encode("utf8")

        # fake write result for metadata output, we want metadata file to be
        # the last file written, so need to delay it until after sha1 is
        # written.
        meta_sha1 = dask.delayed(WriteResult(json_url, mk_sha1(json_data), None))

        paths = task.paths("absolute", ext=self._band_ext)
        cogs = self._ds_to_cog(ds, paths)

        if aux is not None:
            aux_paths = {
                k: task.aux_path(k, relative_to="absolute", ext=self._band_ext)
                for k in aux.data_vars
            }
            cogs.extend(self._ds_to_cog(aux, aux_paths))

        # this will raise IOError if any write failed, hence preventing json
        # from being written
        sha1_digest = _sha1_digest(meta_sha1, *cogs)
        sha1_url = task.metadata_path("absolute", ext="sha1")
        sha1_done = self._write_blob(sha1_digest, sha1_url, ContentType="text/plain")

        return self._write_blob(
            json_data,
            json_url,
            ContentType=self._stac_meta_contentype,
            with_deps=sha1_done,
        )

    # pylint: disable=too-many-locals,protected-access
    def dump_with_eodatasets3(
        self,
        task: Task,
        proc: StatsPluginInterface,
        ds: xr.Dataset,
        aux: xr.Dataset | None = None,
    ) -> Delayed:
        """
        Dump files with metadata files, which generated from eodatasets3
        """
        stac_file_path = task.metadata_path("absolute", ext=self._stac_meta_ext)
        odc_file_path = task.metadata_path("absolute", ext=self._odc_meta_ext)
        sha1_url = task.metadata_path("absolute", ext="sha1")
        proc_info_url = task.metadata_path("absolute", ext=self._proc_info_ext)
        dataset_assembler = task.render_assembler_metadata(
            ext=self._band_ext,
            output_dataset=ds,
            use_center_time=getattr(proc, "CENTER_TIMERANGE", False),
        )

        dataset_assembler.extend_user_metadata(
            "input-products", sorted({e.product.name for e in task.datasets})
        )

        dataset_assembler.extend_user_metadata("odc-stats-config", vars(task.product))

        dataset_assembler.note_software_version(
            "eodatasets3",
            "https://github.com/GeoscienceAustralia/eo-datasets",
            version("eodatasets3"),
        )

        dataset_assembler.note_software_version(
            "odc-stats",
            "https://github.com/opendatacube/odc-stats",
            version("odc_stats"),
        )

        dataset_assembler.note_software_version(
            proc.NAME, "https://github.com/opendatacube/odc-stats", proc.VERSION
        )

        if task.product.preview_image_ows_style:
            try:
                dataset_assembler._accessories["thumbnail"] = Path(
                    urlparse(odc_file_path.split(".")[0] + "_thumbnail.jpg").path
                ).name

                dataset_assembler.note_software_version(
                    "datacube-ows",
                    "https://github.com/opendatacube/datacube-ows",
                    # Just realized the odc-stats does not have version.
                    version("datacube_ows"),
                )
            except ImportError as e:
                raise type(e)(
                    str(e)
                    + '. Please run python -m pip install "odc-stats[ows]" \
                            to setup environment to generate thumbnail.'
                )

        dataset_assembler._accessories["checksum:sha1"] = Path(
            urlparse(sha1_url).path
        ).name
        dataset_assembler._accessories["metadata:processor"] = Path(
            urlparse(proc_info_url).path
        ).name

        meta = dataset_assembler.to_dataset_doc()
        # already add all information to dataset_assembler,
        # now convert to odc and stac metadata format

        stac_meta = self.get_eo3_stac_meta(task, meta, stac_file_path, odc_file_path)

        meta_stream = io.StringIO("")  # too short, not worth to move to another method.
        serialise.to_stream(meta_stream, meta)
        odc_meta = meta_stream.getvalue()  # odc_meta is Python str

        meta_stream = io.StringIO("")
        serialise._init_yaml().dump(
            {
                **dataset_assembler._user_metadata,
                "software_versions": dataset_assembler._software_versions,
            },
            meta_stream,
        )
        proc_info_meta = meta_stream.getvalue()

        # fake write result for metadata output, we want metadata file to be
        # the last file written, so need to delay it until after sha1 files is
        # written.
        stac_meta_sha1 = dask.delayed(
            WriteResult(stac_file_path, mk_sha1(stac_meta), None)
        )
        odc_meta_sha1 = dask.delayed(
            WriteResult(odc_file_path, mk_sha1(odc_meta), None)
        )
        proc_info_sha1 = dask.delayed(
            WriteResult(proc_info_url, mk_sha1(proc_info_meta), None)
        )

        cogs = self._ds_to_cog(ds, task.paths("absolute", ext=self._band_ext))

        if aux is not None:
            aux_paths = {
                k: task.aux_path(k, relative_to="absolute", ext=self._band_ext)
                for k in aux.data_vars
            }
            cogs.extend(self._ds_to_cog(aux, aux_paths))

        # this will raise IOError if any write failed, hence preventing json
        # from being written
        sha1_digest = _sha1_digest(
            stac_meta_sha1,
            odc_meta_sha1,
            proc_info_sha1,
            *cogs,
            *self._ds_to_thumbnail_cog(ds, task),
        )

        # The uploading DAG is:
        # sha1_done -> proc_info_done -> odc_meta_done -> stac_meta_done

        sha1_done = self._write_blob(sha1_digest, sha1_url, ContentType="text/plain")

        proc_info_done = self._write_blob(
            proc_info_meta,
            proc_info_url,
            ContentType=self._prod_info_meta_contentype,
            with_deps=sha1_done,
        )
        odc_meta_done = self._write_blob(
            odc_meta,
            odc_file_path,
            ContentType=self._odc_meta_contentype,
            with_deps=proc_info_done,
        )
        return self._write_blob(
            stac_meta,
            stac_file_path,
            ContentType=self._stac_meta_contentype,
            with_deps=odc_meta_done,
        )

    # pylint: disable=too-many-positional-arguments
    def dump(
        self,
        task: Task,
        proc: StatsPluginInterface,
        ds: xr.Dataset,
        aux: xr.Dataset | None = None,
        apply_eodatasets3: bool | None = False,
    ) -> Delayed:
        if apply_eodatasets3:
            return self.dump_with_eodatasets3(task, proc, ds, aux)
        else:
            return self.dump_with_pystac(task, proc, ds, aux)


def compute_native_load_geobox(
    dst_geobox: GeoBox, ds: Dataset, band: str, buffer: float | None = None
) -> GeoBox:
    """Compute area of interest for a given Dataset given query.

    Take native projection and resolution from ``ds, band`` pair and compute
    region in that projection that fully encloses footprint of the
    ``dst_geobox`` with some padding. Construct GeoBox that encloses that
    region fully with resolution/pixel alignment copied from supplied band.

    :param dst_geobox:
    :param ds: Sample dataset (only resolution and projection is used, not footprint)
    :param band: Reference band to use
                 (resolution of output GeoBox will match resolution of this band)
    :param buffer: Buffer in units of CRS of ``ds`` (meters usually),
                   default is 10 pixels worth
    """
    native: GeoBox = native_geobox(ds, basis=band)
    if buffer is None:
        buffer = 10 * cast(
            float, max(map(abs, (native.resolution.y, native.resolution.x)))
        )  # type: ignore

    assert native.crs is not None
    return GeoBox.from_geopolygon(
        dst_geobox.extent.to_crs(native.crs).buffer(buffer),
        crs=native.crs,
        resolution=native.resolution,
        align=native.alignment,
    )


def choose_transform_path(
    src_crs: str,
    dst_crs: str,
    transform_code: str | None = None,
    area_of_interest: Sequence[float] | None = None,
) -> str:
    # leave gdal to choose the best option if nothing is specified
    if transform_code is None and area_of_interest is None:
        return {}

    if area_of_interest is not None:
        assert len(area_of_interest) == 4
        area_of_interest = aoi.AreaOfInterest(*area_of_interest)

    transformer_group = transformer.TransformerGroup(
        src_crs, dst_crs, area_of_interest=area_of_interest
    )
    if transform_code is None:
        return {"COORDINATE_OPERATION": transformer_group.transformers[0].to_proj4()}
    for t in transformer_group.transformers:
        for step in json.loads(t.to_json()).get("steps", []):
            if step.get("type", "") == "Transformation":
                authority_code = step.get("id", {})
                if transform_code.split(":")[0].upper() in authority_code.get(
                    "authority", ""
                ) and transform_code.split(":")[1] == str(
                    authority_code.get("code", "")
                ):
                    return {"COORDINATE_OPERATION": t.to_proj4()}
    # raise error if nothing is available
    raise ValueError(f"Not able to find transform path by {transform_code}")


def _split_by_grid(xx: xr.DataArray) -> list[xr.DataArray]:
    def extract(grid_id, ii):
        yy = xx[ii]
        crs = xx.grid2crs[grid_id]
        yy.attrs.update(crs=crs)
        yy.attrs.pop("grid2crs", None)
        return yy

    return [extract(grid_id, ii) for grid_id, ii in xx.groupby(xx.grid).groups.items()]


def _native_load_1(
    sources: xr.DataArray,
    bands: tuple[str, ...],
    geobox: GeoBox,
    *,
    optional_bands: tuple[str, ...] | None = None,
    basis: str | None = None,
    load_chunks: dict[str, int] | None = None,
    pad: int | None = None,
) -> xr.Dataset:
    if basis is None:
        basis = bands[0]
    (ds,) = sources.data[0]
    load_geobox = compute_native_load_geobox(geobox, ds, basis)
    if pad is not None:
        load_geobox = gbox_pad(load_geobox, pad)

    mm = ds.product.lookup_measurements(bands)
    if optional_bands is not None:
        for ob in optional_bands:
            try:
                om = ds.product.lookup_measurements(ob)
            except KeyError:
                continue
            else:
                mm.update(om)

    xx = Datacube.load_data(sources, load_geobox, mm, dask_chunks=load_chunks)
    return xx


def native_load(
    dss: Sequence[Dataset],
    bands: Sequence[str],
    geobox: GeoBox,
    *,
    optional_bands: tuple[str, ...] | None = None,
    basis: str | None = None,
    load_chunks: dict[str, int] | None = None,
    pad: int | None = None,
):
    sources = group_by_nothing(list(dss), solar_offset(geobox.extent))
    for srcs in _split_by_grid(sources):
        _xx = _native_load_1(
            srcs,
            tuple(bands),
            geobox,
            optional_bands=optional_bands,
            basis=basis,
            load_chunks=load_chunks,
            pad=pad,
        )
        yield _xx


def _apply_native_transform_1(
    xx: xr.Dataset,
    native_transform: Callable[[xr.Dataset], xr.Dataset],
    groupby: str | None = None,
    fuser: Callable[[xr.Dataset], xr.Dataset] | None = None,
) -> xr.Dataset:
    xx = native_transform(xx)

    if groupby is not None:
        if fuser is None:
            fuser = _nodata_fuser  # type: ignore
        xx = xx.groupby(groupby).map(fuser)

    return xx


# pylint:disable=too-many-arguments,too-many-locals,too-many-branches
def load_with_native_transform(
    dss: Sequence[Dataset],
    bands: Sequence[str],
    geobox: GeoBox,
    native_transform: Callable[[xr.Dataset], xr.Dataset],
    *,
    optional_bands: tuple[str, ...] | None = None,
    basis: str | None = None,
    groupby: str | None = None,
    fuser: Callable[[xr.Dataset], xr.Dataset] | None = None,
    resampling: str = "nearest",
    chunks: dict[str, int] | None = None,
    load_chunks: dict[str, int] | None = None,
    pad: int | None = None,
    **kw,
) -> xr.Dataset:
    """Load a bunch of datasets with native pixel transform.

    :param dss: A list of datasets to load
    :param bands: Which measurements to load
    :param geobox: GeoBox of the final output
    :param native_transform: ``xr.Dataset -> xr.Dataset`` transform,
                             should support Dask inputs/outputs
    :param basis: Name of the band to use as a reference for what is "native projection"
    :param groupby: One of 'solar_day'|'time'|'idx'|None
    :param fuser: Optional ``xr.Dataset -> xr.Dataset`` transform
    :param resampling: Any resampling mode supported by GDAL as a string:
                       nearest, bilinear, average, mode, cubic, etc...
    :param chunks: If set use Dask, must be in dictionary form
                   ``{'x': 4000, 'y': 4000}``

    :param load_chunks: Defaults to ``chunks`` but can be different if supplied
                        (different chunking for native read vs reproject)

    :param pad: Optional padding in native pixels, if set will load extra
                pixels beyond of what is needed to reproject to final
                destination. This is useful when you plan to apply convolution
                filter or morphological operators on input data.

    :param kw: Used to support old names ``dask_chunks`` and ``group_by``
               also kwargs for reproject ``tranform_code`` in the form of
               "authority:code", e.g., "epsg:9688", and ``area_of_interest``,
               e.g., [-180, -90, 180, 90]

    1. Partition datasets by native Projection
    2. For every group do
       - Load data
       - Apply native_transform
       - [Optional] fuse rasters that happened on the same day/time
       - Reproject to final geobox
    3. Stack output of (2)
    4. [Optional] fuse rasters that happened on the same day/time
    """
    if fuser is None:
        fuser = _nodata_fuser

    if groupby is None:
        groupby = kw.pop("group_by", "idx")

    if chunks is None:
        chunks = kw.pop("dask_chunks", None)

    if load_chunks is None:
        load_chunks = chunks

    _chunks = None
    if chunks is not None:
        _chunks = tuple(
            getattr(geobox.shape, ax) if chunks.get(ax, -1) == -1 else chunks.get(ax)
            for ax in ("y", "x")
        )

    _xx = []
    # fail if the intended transform not available
    # to avoid any unexpected results
    for xx in native_load(
        dss,
        bands,
        geobox,
        optional_bands=optional_bands,
        basis=basis,
        load_chunks=load_chunks,
        pad=pad,
    ):
        extra_args = choose_transform_path(
            xx.crs,
            geobox.crs,
            kw.pop("transform_code", None),
            kw.pop("area_of_interest", None),
        )
        extra_args.update(kw)

        yy = _apply_native_transform_1(
            xx,
            native_transform,
            groupby=groupby,
            fuser=fuser,
        )

        vars_to_scale = False
        if isinstance(yy, xr.DataArray):
            vars_to_scale = True
            if yy.dtype == "bool":
                yy = yy.astype("uint8") << 7
        else:
            vars_to_scale = [var for var in yy.data_vars if yy[var].dtype == "bool"]
            yy = yy.assign(
                **{var: yy[var].astype("uint8") << 7 for var in vars_to_scale}
            )

        _yy = xr_reproject(
            yy,
            geobox,
            resampling=resampling,
            chunks=_chunks,
            **extra_args,
        )

        if isinstance(_yy, xr.DataArray) and vars_to_scale:
            _yy = _yy > 64
        elif vars_to_scale:
            _yy = _yy.assign(**{var: _yy[var] > 64 for var in vars_to_scale})

        _xx += [_yy]

    if len(_xx) == 1:
        xx = _xx[0]
    else:
        xx = xr.concat(_xx, _xx[0].dims[0])  # type: ignore
        if groupby != "idx":
            xx = xx.groupby(groupby).map(fuser)
    # TODO: probably want to replace spec MultiIndex with just `time` component
    return xx


def load_enum_mask(
    dss: list[Dataset],
    band: str,
    geobox: GeoBox,
    *,
    categories: Iterable[str | int],
    invert: bool = False,
    resampling: str = "nearest",
    groupby: str | None = None,
    chunks: dict[str, int] | None = None,
    **kw,
) -> xr.DataArray:
    """Load enumerated mask (like fmask).

    1. Load each mask time slice separately in native projection of the file
    2. Convert enum to Boolean (F:0, T:255)
    3. Optionally (groupby='solar_day') group observations on the same day
       using OR for pixel fusing: T,F->T
    4. Reproject to destination GeoBox (any resampling mode is ok)
    5. Optionally group observations on the same day using OR for pixel fusing T,F->T
    6. Finally convert to real Bool
    """

    def native_op(ds):
        return ds.map(
            enum_to_bool,
            categories=categories,
            invert=invert,
            dtype="uint8",
            value_true=255,
        )

    xx = load_with_native_transform(
        dss,
        (band,),
        geobox,
        native_op,
        basis=band,
        resampling=resampling,
        groupby=groupby,
        chunks=chunks,
        fuser=_max_fuser,
        **kw,
    )
    return xx[band] > 127


def load_enum_filtered(
    dss: Sequence[Dataset],
    band: str,
    geobox: GeoBox,
    *,
    categories: Iterable[str | int],
    filters: Iterable[tuple[str, int]] | None = None,
    groupby: str | None = None,
    resampling: str = "nearest",
    chunks: dict[str, int] | None = None,
    **kw,
) -> xr.DataArray:
    """Load enumerated mask (like fmask/SCL) with native pixel filtering.

    The idea is to load "cloud" classes while adding some padding, then erase
    pixels that were classified as cloud in any of the observations on a given
    day.

    This method converts enum-mask to a boolean image in the native projection
    of the data and then reprojects boolean image to the final
    projections/resolution. This allows one to use any resampling strategy,
    like ``average`` or ``cubic`` and not be limited to a few resampling
    strategies that support operations on categorical data.

    :param dss: A list of datasets to load
    :param band: Which measurement band to load
    :param geobox: GeoBox of the final output
    :param categories: Enum values or names

    :param filters: iterable tuples of morphological operations in the order
                    you want them to perform, e.g., [("opening", 2), ("dilation", 5)]
    :param groupby: One of 'solar_day'|'time'|'idx'|None
    :param resampling: Any resampling mode supported by GDAL as a string:
                       nearest, bilinear, average, mode, cubic, etc...
    :param chunks: If set use Dask, must be in dictionary form
                   ``{'x': 4000, 'y': 4000}``
    :param kw: Passed on to ``load_with_native_transform``


    1. Load each mask time slice separately in native projection of the file
    2. Convert enum to Boolean
    3. Optionally (groupby='solar_day') group observations on the same day
       using OR for pixel fusing: T,F->T
    4. Optionally apply ``mask_cleanup`` in native projection (after fusing)
    4. Reproject to destination GeoBox (any resampling mode is ok)
    5. Optionally group observations on the same day using OR for pixel fusing T,F->T
    """

    def native_op(xx: xr.Dataset) -> xr.Dataset:
        _xx = enum_to_bool(xx[band], categories)
        return xr.Dataset(
            {band: _xx},
            attrs={"native": True},  # <- native flag needed for fuser
        )

    def fuser(xx: xr.Dataset) -> xr.Dataset:
        """Fuse with OR.

        Fuse with OR, and when fusing in native pixel domain apply mask_cleanup if
        requested
        """
        is_native = xx.attrs.get("native", False)
        xx = xx.map(_or_fuser)
        xx.attrs.pop("native", None)

        if is_native and filters is not None:
            _xx = xx[band]
            assert isinstance(_xx, xr.DataArray)
            xx[band] = mask_cleanup(_xx, mask_filters=filters)

        return xx

    # unless set by user to some value use largest filter radius for pad value
    pad: int | None = kw.pop("pad", None)
    if pad is None:
        if filters is not None:
            pad = max(list(zip(*filters, strict=False))[1])  # type: ignore

    xx = load_with_native_transform(
        dss,
        (band,),
        geobox,
        native_op,
        fuser=fuser,
        groupby=groupby,
        resampling=resampling,
        chunks=chunks,
        pad=pad,
        **kw,
    )[band]
    assert isinstance(xx, xr.DataArray)
    return xx
