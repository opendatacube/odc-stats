"""
Simple odc-stats plugin example, with excessive
documentation. Creates temporal mean NDVI
from DEA Landsat Collection 3
    
"""

from typing import Optional, Sequence, Tuple

import numpy as np
import xarray as xr
from datacube.utils.masking import mask_invalid_data
from odc.stats.plugins._registry import register, StatsPluginInterface
from odc.algo._masking import (
    erase_bad,
    enum_to_bool,
    mask_cleanup,
)

NODATA = -999

class StatsNDVI(StatsPluginInterface):
    """
    Define a class for summarising time 
    series of NDVI using the median.
    """
    NAME = "ndvi_median"
    SHORT_NAME = NAME
    VERSION = "1.0"
    PRODUCT_FAMILY = "ndvi"

    def __init__(
        self,
        input_bands: Sequence[str] = None,
        output_bands: Sequence[str] = None,
        mask_band: Sequence[str] = None,
        contiguity_band: Sequence[str] = None,
        group_by: str = "solar_day",
        **kwargs,
    ):
        
        self.input_bands = input_bands
        self.output_bands = output_bands
        self.mask_band = mask_band
        self.contiguity_band = contiguity_band
        self.group_by = group_by

        ## These params get passed to the upstream 
        #  base StatsPluginInterface class
        super().__init__(
            input_bands=tuple(input_bands)+(mask_band,)+(contiguity_band,),
            **kwargs
        )

        
    @property
    def measurements(self) -> Tuple[str, ...]:
        """
        Here we define the output bands, in this example we
        will pass the names of the output bands into the config file,
        but equally we could define the outputs ames within this function.
        For example, by adding a suffix to the input bands.
        """
        
        return (self.output_bands,)

    def native_transform(self, xx):
        """
        This function is passed to an upstream function
        called "odc.algo.io.load_with_native_transform".
        The function decribed here is applied on every time
        step of data and is usually used for things like
        masking clouds, nodata, and contiguity masking.
        """
        #grab the QA band from the Landsat data
        mask = xx[self.mask_band]

        # create boolean arrays from the mask for cloud
        # and cloud shadows, and nodata
        bad = enum_to_bool(mask, ("nodata",))
        non_contiguent = xx.get(self.contiguity_band, 1) == 0
        bad = bad | non_contiguent
        
        cloud_mask = enum_to_bool(mask, ("cloud", "shadow"))
        bad =  cloud_mask | bad

        # drop masking bands
        xx = xx.drop_vars([self.mask_band] + [self.contiguity_band])
        
        ## Mask the bad data (clouds etc)
        xx = erase_bad(xx, bad)

        return xx

    def reduce(self, xx: xr.Dataset) -> xr.Dataset:
        """
        Calculate NDVI and summarise time series with a median.
        """
        # convert to float by and convert nodata to NaN
        # xx = xx.where(xx>1)
        xx = mask_invalid_data(xx)
        
        ndvi = (xx['nbart_nir'] - xx['nbart_red']) / (xx['nbart_nir'] + xx['nbart_red'])

        # calculate temporal median NDVI. 
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!! 
        # Note that we use 'spec' and not 'time', this is an odc-stats thing
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ndvi = ndvi.median('spec').rename(self.output_bands)
        
        return ndvi.to_dataset()

# now lets 'register' the function with odc-stats
register("NDVI-median", StatsNDVI)
