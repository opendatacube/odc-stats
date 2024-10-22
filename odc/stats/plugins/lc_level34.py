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

class StatsL4(StatsPluginInterface):
    NAME = "ga_ls_lccs_veg_bare_class_a3"
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
        # Level-3 to level-4 class map
        self.l3_to_l4_mapping = {
            111: 1,  # Cultivated Terrestrial Vegetated
            112: 19, # Natural Terrestrial Vegetated
            124: 55, # Natural Aquatic Vegetated
            215: 93, # Artificial Surface:
            216: 94, # Natural Surface
            220: 98, # Water
            223: 3,
        }

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
        
    def level3_to_level4_mapping(self, xx: xr.Dataset):
        l3_data = xx.level3_class.data
        l3_data = self.apply_mapping(l3_data, self.l3_to_l4_mapping)

        return l3_data

    
    def reduce(self, xx: xr.Dataset) -> xr.Dataset:
        intertidal_mask, level3 = lc_level3.lc_level3(xx, NODATA)
        # l4 = np.zeros(xx.level3_class.data.shape)
  
        # Vegetation cover
        fc_nodata = -9999
        veg_cover = l4_veg_cover.canopyco_veg_con(xx, self.veg_threshold, NODATA, fc_nodata)
        # Define mapping from current output to expected a3 output
        veg_cover = self.apply_mapping(veg_cover, self.veg_mapping)
        # Define life form
        lifeform = self.define_life_form(xx)

        # Apply cultivated Level-4 classes (1-18)
        l4_ctv = l4_cultivated.lc_l4_cultivated(level3, lifeform, veg_cover)

        # Apply terrestrial vegetation classes [19-36]
        l4_ctv_ntv = l4_natural_veg.lc_l4_natural_veg(xx, l4_ctv, lifeform, veg_cover)
        # level_3 = self.level3_to_level4_mapping(xx)

        # Bare gradation
        bare_gradation = l4_bare_gradation.bare_gradation(xx, self.bare_threshold, veg_cover, NODATA)
        # Apply bare gradation expected output classes
        bare_gradation = self.apply_mapping(bare_gradation, self.bs_mapping)

       
        # Water persistence
        water_persistence = l4_water_persistence.water_persistence(xx, self.watper_threshold, NODATA)
        # Apply water persistence expcted classes
        water_persistence = self.apply_mapping(water_persistence, self.waterper_wat_mapping)
        
        water_seasonality = self.define_water_seasonality(xx) 
        # xx.water_seasonality.data

        l4_ctv_ntv_nav = l4_natural_aquatic.natural_auquatic_veg(l4_ctv_ntv, lifeform, veg_cover, water_seasonality)

        l4_ctv_ntv_nav_surface = l4_surface.lc_l4_surface(l4_ctv_ntv_nav, level3, bare_gradation)
        
        #TODO WATER (99-104)
        l4_ctv_ntv_nav_surface_water = l4_water.water_classification(l4_ctv_ntv_nav_surface, intertidal_mask, water_persistence)
        
        attrs = xx.attrs.copy()
        attrs["nodata"] = NODATA
        l3 = level3.squeeze(dim=["spec"])
        dims = level3.dims

        
        attrs = xx.attrs.copy()
        attrs["nodata"] = int(NODATA)

        data_vars = {
            "level3": xr.DataArray(
                level3.data, dims=xx["pv_pc_50"].dims, attrs=attrs
            ),
            "level4": xr.DataArray(
                l4_ctv_ntv_nav_surface_water, dims=xx["pv_pc_50"].dims, attrs=attrs
            )
        }

        coords = dict((dim, xx.coords[dim]) for dim in dims)
        return xr.Dataset(data_vars=data_vars, coords=coords, attrs=xx.attrs)


# register("lccs_level3", StatsLccsLevel3)
register("lc_l3_l4", StatsL4)