from abc import ABC, abstractmethod
from typing import Mapping, Optional, Sequence, Tuple

import xarray as xr
from datacube.model import Dataset
from datacube.utils.geometry import GeoBox
from odc.algo import to_rgba
from odc.algo.io import load_with_native_transform
from odc.algo._masking import _nodata_fuser


class StatsPluginInterface(ABC):
    NAME = "*unset*"
    SHORT_NAME = ""
    VERSION = "0.0.0"
    PRODUCT_FAMILY = "statistics"

    def __init__(
        self,
        resampling: str = "bilinear",
        input_bands: Optional[Sequence[str]] = None,
        chunks: Optional[Mapping[str, int]] = None,
        basis: Optional[str] = None,
        group_by: str = "solar_day",
        rgb_bands: Optional[Sequence[str]] = None,
        rgb_clamp: Tuple[float, float] = (1.0, 3_000.0),
        transform_code: Optional[str] = None,
        area_of_interest: Optional[Sequence[float]] = None,
    ):
        self.resampling = resampling
        self.input_bands = input_bands if input_bands is not None else []
        self.chunks = chunks if chunks is not None else {"y": -1, "x": -1}
        self.basis = basis
        self.group_by = group_by
        self.rgb_bands = rgb_bands
        self.rgb_clamp = rgb_clamp
        self.transform_code = transform_code
        self.area_of_interest = area_of_interest

    @property
    @abstractmethod
    def measurements(self) -> Tuple[str, ...]:
        pass

    def native_transform(self, xx: xr.Dataset) -> xr.Dataset:
        return xx

    def fuser(self, xx: xr.Dataset) -> xr.Dataset:
        return _nodata_fuser(xx)

    def input_data(
        self, datasets: Sequence[Dataset], geobox: GeoBox, **kwargs
    ) -> xr.Dataset:
        xx = load_with_native_transform(
            datasets,
            bands=self.input_bands,
            geobox=geobox,
            native_transform=self.native_transform,
            basis=self.basis,
            groupby=self.group_by,
            fuser=self.fuser,
            resampling=self.resampling,
            chunks=self.chunks,
            **kwargs,
        )
        return xx

    @abstractmethod
    def reduce(self, xx: xr.Dataset) -> xr.Dataset:
        pass

    def rgba(self, xx: xr.Dataset) -> Optional[xr.DataArray]:
        """
        Given result of ``.reduce(..)`` optionally produce RGBA preview image
        """
        if self.rgb_bands is None:
            return None
        return to_rgba(xx, clamp=self.rgb_clamp, bands=self.rgb_bands)
