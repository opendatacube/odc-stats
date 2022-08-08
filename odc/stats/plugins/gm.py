"""
Geomedian
"""
from typing import Optional, Tuple, Iterable, Dict
import xarray as xr
from odc.algo import geomedian_with_mads
from ._registry import StatsPluginInterface, register
from odc.algo import enum_to_bool, keep_good_only
from odc.algo import mask_cleanup
import logging

_log = logging.getLogger(__name__)


class StatsGM(StatsPluginInterface):
    NAME = "gm"
    SHORT_NAME = NAME
    VERSION = "0.0.0"
    PRODUCT_FAMILY = "geomedian"

    def __init__(
        self,
        bands: Tuple[str, ...],
        mask_band: str,
        nodata_classes: Optional[Tuple[str, ...]] = None,
        cloud_filters: Dict[str, Iterable[Tuple[str, int]]] = None,
        basis_band=None,
        aux_names: Dict[str, str] = None,
        resampling: str = "nearest",
        work_chunks: Tuple[int, int] = (400, 400),
        **kwargs,
    ):
        aux_names = (
            dict(smad="smad", emad="emad", bcmad="bcmad", count="count")
            if aux_names is None
            else aux_names
        )
        self.bands = tuple(bands)
        self._mask_band = mask_band
        if nodata_classes is not None:
            nodata_classes = tuple(nodata_classes)
        self._nodata_classes = nodata_classes
        self.resampling = resampling
        input_bands = self.bands
        if self._nodata_classes is not None:
            # NOTE: this ends up loading Mask band twice, once to compute
            # ``.erase`` band and once to compute ``nodata`` mask.
            input_bands = (*input_bands, self._mask_band)

        super().__init__(
            input_bands=input_bands,
            basis=basis_band or self.bands[0],
            resampling=self.resampling,
            **kwargs,
        )

        self.cloud_filters = cloud_filters
        self._renames = aux_names
        self.aux_bands = tuple(
            self._renames.get(k, k)
            for k in (
                "smad",
                "emad",
                "bcmad",
                "count",
            )
        )

        self._work_chunks = work_chunks

    @property
    def measurements(self) -> Tuple[str, ...]:
        return self.bands + self.aux_bands

    def native_transform(self, xx: xr.Dataset) -> xr.Dataset:
        if self._mask_band not in xx.data_vars:
            return xx

        # Apply the contiguity flag
        non_contiguent = xx.get("nbart_contiguity", 1) == 0

        # Erase Data Pixels for which mask == nodata
        mask = xx[self._mask_band]
        bad = enum_to_bool(mask, self._nodata_classes)
        bad = bad | non_contiguent
        if self.cloud_filters is not None:
            for cloud_class, c_filter in self.cloud_filters.items():
                cloud_mask = enum_to_bool(mask, (cloud_class,))
                cloud_mask_buffered = mask_cleanup(cloud_mask, mask_filters=c_filter)
                bad = cloud_mask_buffered | bad
        else:
            cloud_shadow_mask = enum_to_bool(mask, ("cloud", "shadow"))
            bad = cloud_shadow_mask | bad
            _log.info("Applying cloud/shadow mask without buffering.")

        xx = xx.drop_vars([self._mask_band] + ["nbart_contiguity"])
        xx = keep_good_only(xx, ~bad)
        return xx

    def reduce(self, xx: xr.Dataset) -> xr.Dataset:
        scale = 1 / 10_000
        cfg = dict(
            maxiters=1000,
            num_threads=1,
            scale=scale,
            offset=-1 * scale,
            reshape_strategy="mem",
            out_chunks=(-1, -1, -1),
            work_chunks=self._work_chunks,
            compute_count=True,
            compute_mads=True,
        )

        gm = geomedian_with_mads(xx, **cfg)
        gm = gm.rename(self._renames)

        return gm


register("gm-generic", StatsGM)


class StatsGMS2(StatsGM):
    NAME = "gm_s2_annual"
    SHORT_NAME = NAME
    VERSION = "0.0.0"
    PRODUCT_FAMILY = "geomedian"
    DEFAULT_FILTER = [("opening", 2), ("dilation", 5)]

    def __init__(
        self,
        bands: Optional[Tuple[str, ...]] = None,
        mask_band: str = "SCL",
        cloud_filters: Dict[str, Iterable[Tuple[str, int]]] = None,
        aux_names: Dict[str, str] = None,
        rgb_bands=None,
        **kwargs,
    ):
        cloud_filters = (
            {
                "cloud shadows": self.DEFAULT_FILTER,
                "cloud medium probability": self.DEFAULT_FILTER,
                "cloud high probability": self.DEFAULT_FILTER,
                "thin cirrus": self.DEFAULT_FILTER,
            }
            if cloud_filters is None
            else cloud_filters
        )

        aux_names = (
            dict(smad="SMAD", emad="EMAD", bcmad="BCMAD", count="COUNT")
            if aux_names is None
            else aux_names
        )

        if bands is None:
            bands = (
                "B02",
                "B03",
                "B04",
                "B05",
                "B06",
                "B07",
                "B08",
                "B8A",
                "B11",
                "B12",
            )
            if rgb_bands is None:
                rgb_bands = ("B04", "B03", "B02")

        super().__init__(
            bands=bands,
            mask_band=mask_band,
            cloud_filters=cloud_filters,
            aux_names=aux_names,
            rgb_bands=rgb_bands,
            **kwargs,
        )


register("gm-s2", StatsGMS2)


class StatsGMLS(StatsGM):
    NAME = "gm_ls_annual"
    SHORT_NAME = NAME
    VERSION = "3.0.1"
    PRODUCT_FAMILY = "geomedian"

    def __init__(
        self,
        bands: Optional[Tuple[str, ...]] = None,
        mask_band: str = "fmask",
        nodata_classes: Optional[Tuple[str, ...]] = ("nodata",),
        cloud_filters: Dict[str, Iterable[Tuple[str, int]]] = None,
        aux_names: Dict[str, str] = None,
        rgb_bands=None,
        **kwargs,
    ):
        aux_names = (
            dict(
                smad="sdev",
                emad="edev",
                bcmad="bcdev",
                count="count",
            )
            if aux_names is None
            else aux_names
        )

        if bands is None:
            bands = (
                "nbart_red",
                "nbart_green",
                "nbart_blue",
                "nbart_nir",
                "nbart_swir_1",
                "nbart_swir_2",
                "nbart_contiguity",
            )
            if rgb_bands is None:
                rgb_bands = ("nbart_red", "nbart_green", "nbart_blue")

        super().__init__(
            bands=bands,
            mask_band=mask_band,
            cloud_filters=cloud_filters,
            nodata_classes=nodata_classes,
            aux_names=aux_names,
            rgb_bands=rgb_bands,
            **kwargs,
        )

    @property
    def measurements(self) -> Tuple[str, ...]:
        return tuple(b for b in self.bands if b != "nbart_contiguity") + self.aux_bands


register("gm-ls", StatsGMLS)
