"""
Geomedian
"""

from typing import Optional, Union, Tuple, Iterable, Dict
import xarray as xr
from odc.algo import geomedian_with_mads
from ._registry import StatsPluginInterface, register
from odc.algo import enum_to_bool, erase_bad
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
        contiguity_band: Optional[str] = None,
        nodata_classes: Optional[Tuple[str, ...]] = None,
        cloud_filters: Dict[
            Union[str, Tuple[str, ...]], Iterable[Tuple[str, int]]
        ] = None,
        basis_band=None,
        aux_names: Dict[str, str] = None,
        work_chunks: Tuple[int, int] = (400, 400),
        **kwargs,
    ):
        aux_names = (
            {"smad": "smad", "emad": "emad", "bcmad": "bcmad", "count": "count"}
            if aux_names is None
            else aux_names
        )
        self.bands = tuple(bands)
        self._mask_band = mask_band
        self._contiguity_band = contiguity_band
        if nodata_classes is not None:
            nodata_classes = tuple(nodata_classes)
        self._nodata_classes = nodata_classes
        input_bands = self.bands
        if self._nodata_classes is not None:
            # NOTE: this ends up loading Mask band twice, once to compute
            # ``.erase`` band and once to compute ``nodata`` mask.
            input_bands = (*input_bands, self._mask_band)

        super().__init__(
            input_bands=input_bands,
            basis=basis_band or self.bands[0],
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

        # Erase Data Pixels for which mask == nodata
        mask = xx[self._mask_band]
        bad = enum_to_bool(mask, self._nodata_classes)
        # Apply the contiguity flag
        if self._contiguity_band is not None:
            non_contiguent = xx.get(self._contiguity_band, 1) == 0
            bad = bad | non_contiguent

        if self.cloud_filters is not None:
            for cloud_class, c_filter in self.cloud_filters.items():
                if not isinstance(cloud_class, tuple):
                    cloud_class = (cloud_class,)
                cloud_mask = enum_to_bool(mask, cloud_class)
                cloud_mask_buffered = mask_cleanup(cloud_mask, mask_filters=c_filter)
                bad = cloud_mask_buffered | bad
        else:
            cloud_shadow_mask = enum_to_bool(mask, ("cloud", "shadow"))
            bad = cloud_shadow_mask | bad
            _log.info("Applying cloud/shadow mask without buffering.")

        if self._contiguity_band is not None:
            xx = xx.drop_vars([self._mask_band] + [self._contiguity_band])
        else:
            xx = xx.drop_vars([self._mask_band])
        xx = erase_bad(xx, bad)
        return xx

    def reduce(self, xx: xr.Dataset) -> xr.Dataset:
        scale = 1 / 10_000
        cfg = {
            "maxiters": 1000,
            "num_threads": 1,
            "scale": scale,
            "offset": -1 * scale,
            "reshape_strategy": "mem",
            "out_chunks": (-1, -1, -1),
            "work_chunks": self._work_chunks,
            "compute_count": True,
            "compute_mads": True,
        }

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
        nodata_classes: Optional[Tuple[str, ...]] = ("no data",),
        cloud_filters: Dict[
            Union[str, Tuple[str, ...]], Iterable[Tuple[str, int]]
        ] = None,
        aux_names: Dict[str, str] = None,
        rgb_bands=None,
        **kwargs,
    ):
        cloud_filters = (
            {
                (
                    "cloud shadows",
                    "cloud medium probability",
                    "cloud high probability",
                    "thin cirrus",
                ): self.DEFAULT_FILTER,
            }
            if cloud_filters is None
            else cloud_filters
        )

        aux_names = (
            {"smad": "SMAD", "emad": "EMAD", "bcmad": "BCMAD", "count": "COUNT"}
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
            nodata_classes=nodata_classes,
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
        contiguity_band: str = "nbart_contiguity",
        nodata_classes: Optional[Tuple[str, ...]] = ("nodata",),
        cloud_filters: Dict[
            Union[str, Tuple[str, ...]], Iterable[Tuple[str, int]]
        ] = None,
        aux_names: Dict[str, str] = None,
        rgb_bands=None,
        **kwargs,
    ):
        aux_names = (
            {"smad": "sdev", "emad": "edev", "bcmad": "bcdev", "count": "count"}
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

        # ideally it should be read from product def
        self.nodata_defs = kwargs.pop(
            "nodata_defs",
            {
                aux_names["smad"]: float("nan"),
                aux_names["bcmad"]: float("nan"),
                aux_names["emad"]: float("nan"),
            },
        )

        super().__init__(
            bands=bands,
            mask_band=mask_band,
            contiguity_band=contiguity_band,
            cloud_filters=cloud_filters,
            nodata_classes=nodata_classes,
            aux_names=aux_names,
            rgb_bands=rgb_bands,
            **kwargs,
        )

    @property
    def measurements(self) -> Tuple[str, ...]:
        return (
            tuple(b for b in self.bands if b != self._contiguity_band) + self.aux_bands
        )

    def reduce(self, xx: xr.Dataset) -> xr.Dataset:
        gm = super().reduce(xx)
        for key, val in self.nodata_defs.items():
            gm[key].attrs["nodata"] = val

        return gm


register("gm-ls", StatsGMLS)
