"""
Plugin of Module A3 in LandCover PipeLine
"""

from typing import Tuple, Optional, Dict, List

import numpy as np
import xarray as xr

from odc.stats._algebra import expr_eval
from ._registry import StatsPluginInterface, register

from .l34_utils import l4_water_persistence, l4_veg_cover, lc_level3, l4_cultivated, l4_natural_veg, l4_natural_aquatic, l4_surface, l4_bare_gradation, l4_water


NODATA = 255
water_frequency_nodata = -999

class StatsLccsLevel4(StatsPluginInterface):
    NAME = "ga_ls_lccs_Level34"
    SHORT_NAME = NAME
    VERSION = "0.0.1"
    PRODUCT_FAMILY = "lccs"

    def __init__(
        self,
        veg_threshold: Optional[List] = None,
        bare_threshold: Optional[List] = None,
        watper_threshold: Optional[List] = None,
        veg_mapping: Optional[Dict[int, int]] = None,
        bs_mapping: Optional[Dict[int, int]] = None,
        waterper_wat_mapping: Optional[Dict[int, int]] = None,
        l3_to_l4_mapping: Optional[Dict[int, int]] = None,
        water_seasonality_threshold: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.veg_threshold = (
            veg_threshold if veg_threshold is not None else [1, 4, 15, 40, 65, 100]
        )
        self.bare_threshold = bare_threshold if bare_threshold is not None else [20, 60]
        self.watper_threshold = (
            watper_threshold if watper_threshold is not None else [1, 4, 7, 10]
        )
        self.water_seasonality_threshold = water_seasonality_threshold if water_seasonality_threshold else 3
        
        # The mapping below are from the LC KH page
        # Map vegetation cover classes
        self.veg_mapping = {160: 16, 150: 15, 130: 13, 120: 12, 100: 10}
        # Map bare gradation classes
        self.bs_mapping = {100: 10, 120: 12, 150: 15}
        # Map values to the classes expected in water persistence in land cover Level-4 output
        self.waterper_wat_mapping = {100: 1, 70: 7, 80: 8, 90: 9}
       

    @property
    def measurements(self) -> Tuple[str, ...]:
        _measurements = [
            "level3",
            "level4"
        ]
        return _measurements

    def native_transform(self, xx):
        return xx

    def fuser(self, xx):
        return xx

    @staticmethod
    def apply_mapping(data, class_mapping):
        for o_val, n_val in class_mapping.items():
            data = xr.where(data == o_val, n_val, data)
        return data

    def define_life_form(self, xx: xr.Dataset):
        lifeform = xx.woody_cover.data
        
        # 113 ----> 1 woody
        # 114 ----> 2 herbaceous
        lifeform_mask = expr_eval(
            "where(a==113, 1, a)",
            {"a": xx.woody_cover.data},
            name="mark_lifeform",
            dtype="uint8"
        )
        lifeform_mask = expr_eval(
            "where(a==114, 2, a)",
            {"a": lifeform_mask},
            name="mark_lifeform",
            dtype="uint8"
        )
 
        return lifeform_mask
        
    def define_water_seasonality(self, xx: xr.Dataset):
        # >= 3 months ----> 1  Semi-permanent or permanent
        # < 3 months  ----> 2 Temporary or seasonal

        water_season_mask = expr_eval(
            "where((a>watseas_trh)&(a<=12), 100, a)",
            {"a": xx.water_frequency.data},
            name="mark_water_season",
            dtype="uint8",
            **{"watseas_trh": self.water_seasonality_threshold},
        )
        water_season_mask = expr_eval(
            "where((a<=watseas_trh)&(a<=12), 200, a)",
            {"a": water_season_mask},
            name="mark_water_season",
            dtype="uint8",
            **{"watseas_trh": self.water_seasonality_threshold},
        )
        water_season_mask = expr_eval(
            "where((a==watersea_nodata), 255, a)",
            {"a": water_season_mask},
            name="mark_water_season",
            dtype="uint8",
            **{"watseas_trh": self.water_seasonality_threshold,
               "watersea_nodata": water_frequency_nodata},
        )
        mapping = {100:1, 200:2}
        water_season_mask = self.apply_mapping(water_season_mask, mapping)
 
        return water_season_mask

    
    def reduce(self, xx: xr.Dataset) -> xr.Dataset:

        intertidal_mask, level3 = lc_level3.lc_level3(xx)
    
        # Vegetation cover
        veg_cover = l4_veg_cover.canopyco_veg_con(xx, self.veg_threshold)
        # Define mapping from current output to expected a3 output
        veg_cover = self.apply_mapping(veg_cover, self.veg_mapping)
        # Define life form
        lifeform = self.define_life_form(xx)

        # Apply cultivated Level-4 classes (1-18)
        l4_ctv = l4_cultivated.lc_l4_cultivated(xx.classes_l3_l4, level3, lifeform, veg_cover)
        print("***** CULATIVATED: ", np.unique(l4_ctv.compute()))
        # Apply terrestrial vegetation classes [19-36]
        l4_ctv_ntv = l4_natural_veg.lc_l4_natural_veg(l4_ctv, level3, lifeform, veg_cover)
        print("***** CULATIVATED NTV : ", np.unique(l4_ctv_ntv.compute()))
       
        # Bare gradation
        bare_gradation = l4_bare_gradation.bare_gradation(xx, self.bare_threshold, veg_cover)
        # Apply bare gradation expected output classes
        bare_gradation = self.apply_mapping(bare_gradation, self.bs_mapping)

       
        # Water persistence
        water_persistence = l4_water_persistence.water_persistence(xx, self.watper_threshold)
        # Apply water persistence expcted classes
        water_persistence = self.apply_mapping(water_persistence, self.waterper_wat_mapping)
        
        water_seasonality = self.define_water_seasonality(xx) 

        l4_ctv_ntv_nav = l4_natural_aquatic.natural_auquatic_veg(l4_ctv_ntv, lifeform, veg_cover, water_seasonality)
        print("***** NAV : ", np.unique(l4_ctv_ntv_nav.compute()))
        l4_ctv_ntv_nav_surface = l4_surface.lc_l4_surface(l4_ctv_ntv_nav, level3, bare_gradation)
        print("***** SURFACE : ", np.unique(l4_ctv_ntv_nav_surface.compute()))
        # #TODO WATER (99-104)
        level4 = l4_water.water_classification(l4_ctv_ntv_nav_surface, level3, intertidal_mask, water_persistence)
        print("***** LEVEL3:",  np.unique(level3.compute()))
        attrs = xx.attrs.copy()
        attrs["nodata"] = NODATA
        # l3 = level3.squeeze(dim=["spec"])
        dims = xx.squeeze(dim=["spec"]).dims
   
        attrs = xx.attrs.copy()
        attrs["nodata"] = int(NODATA)

        level3 = level3.astype(np.uint8)
        level4 = level4.astype(np.uint8)
        data_vars = {
            "level3": xr.DataArray(
                level3, dims=xx["pv_pc_50"].dims, attrs=attrs
            ),
            "level4": xr.DataArray(
                level4, dims=xx["pv_pc_50"].dims, attrs=attrs
            )
        }
        
        coords = dict((dim, xx.coords[dim]) for dim in dims)
        
        print(xr.Dataset(data_vars=data_vars, coords=coords, attrs=xx.attrs))
        return xr.Dataset(data_vars=data_vars, coords=coords, attrs=xx.attrs)


register("lc_l3_l4", StatsLccsLevel4)