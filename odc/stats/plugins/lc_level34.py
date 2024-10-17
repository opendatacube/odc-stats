"""
Plugin of Module A3 in LandCover PipeLine
"""

from typing import Tuple, Optional, Dict, List

import numpy as np
import xarray as xr

from odc.stats._algebra import expr_eval
from ._registry import StatsPluginInterface, register

from .l34_utils import l4_water_persistence, l4_veg_cover, lc_level3, l4_cultivated, l4_natural_veg, l4_natural_aquatic, l4_surface, l4_bare_gradation


NODATA = 255

# class StatsLccsLevel3(StatsPluginInterface):
#     NAME = "ga_ls_lccs_level3"
#     SHORT_NAME = NAME
#     VERSION = "0.0.1"
#     PRODUCT_FAMILY = "lccs"

#     @property
#     def measurements(self) -> Tuple[str, ...]:
#         _measurements = ["level3_class"]
#         return _measurements

#     def reduce(self, xx: xr.Dataset) -> xr.Dataset:

#         l34_dss = xx.classes_l3_l4
#         urban_dss = xx.urban_classes
#         cultivated_dss = xx.cultivated_class
    
#         # Map intertidal areas to water
#         intertidal = l34_dss == 223
#         l34_dss = xr.where(intertidal, 220, l34_dss)
        
#         # Cultivated pipeline applies a mask which feeds only terrestrial veg (110) to the model
#         # Just exclude no data (255) and apply the cultivated results
#         cultivated_mask = cultivated_dss != int(NODATA)
#         l34_cultivated_masked = xr.where(cultivated_mask, cultivated_dss, l34_dss)

#         # Urban is classified on l3/4 surface output (210)
#         urban_mask = l34_dss == 210
#         l34_urban_cultivated_masked = xr.where(
#             urban_mask, urban_dss, l34_cultivated_masked
#         )

#         attrs = xx.attrs.copy()
#         attrs["nodata"] = NODATA
#         l34_urban_cultivated_masked = l34_urban_cultivated_masked.squeeze(dim=["spec"])
#         dims = l34_urban_cultivated_masked.dims

#         data_vars = {
#             "level3_class": xr.DataArray(
#                 l34_urban_cultivated_masked.data, dims=dims, attrs=attrs
#             )
#         }

#         coords = dict((dim, xx.coords[dim]) for dim in dims)
#         level3 = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

#         return level3
    
class StatsVegDryClassA3(StatsPluginInterface):
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
            "where(a==113, 1, 2)",
            {"a": xx.woody_cover.data},
            name="mark_lifeform",
            dtype="uint8"
        )
        return lifeform_mask
        
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
        lifeform = self.define_life_form(xx).compute()

        # Apply cultivated Level-4 classes (1-18)
        l4_ctv = l4_cultivated.lc_l4_cultivated(level3, lifeform, veg_cover)

        # Apply terrestrial vegetation classes [19-36]
        l4_ctv_ntv = l4_natural_veg.lc_l4_natural_veg(l4_ctv, lifeform, veg_cover)
        # level_3 = self.level3_to_level4_mapping(xx)

        # Bare gradation
        bare_gradation = l4_bare_gradation.bare_gradation(xx, self.bare_threshold, veg_cover, NODATA)
        # Apply bare gradation expected output classes
        bare_gradation = self.apply_mapping(bare_gradation, self.bs_mapping)

       
        # Water persistence
        water_persistence = l4_water_persistence.water_persistence(xx, self.watper_threshold, NODATA)
        # Apply water persistence expcted classes
        water_persistence = self.apply_mapping(water_persistence, self.waterper_wat_mapping)
        
        water_seasonality = xx.water_seasonality.data

        l4_ctv_ntv_nav = l4_natural_aquatic.natural_auquatic_veg(l4_ctv_ntv, level3, lifeform, veg_cover, water_seasonality)


        l4_ctv_ntv_nav_surface = l4_surface.lc_l4_surface(l4_ctv_ntv_nav, bare_gradation)
        #TODO WATER (99-105) if not added 
        
        attrs = xx.attrs.copy()
        attrs["nodata"] = NODATA
        l3 = level3.squeeze(dim=["spec"])
        dims = level3.dims

        # data_vars = {
        #     "level3_class": xr.DataArray(
        #         l3.data, dims=dims, attrs=attrs
        #     )
        # }

        # coords = dict((dim, xx.coords[dim]) for dim in dims)
        # level3 = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
        
        attrs = xx.attrs.copy()
        attrs["nodata"] = int(NODATA)

        data_vars = {
            "level3": xr.DataArray(
                level3.data, dims=xx["pv_pc_50"].dims, attrs=attrs
            ),
            "level4": xr.DataArray(
                l4_ctv_ntv_nav_surface, dims=xx["pv_pc_50"].dims, attrs=attrs
            )
            # "waterper_wat_cin": xr.DataArray(
            #     a3[2], dims=xx["water_frequency"].dims, attrs=attrs
            # ),
            # "level3": xr.DataArray(a3[3], dims=xx["level3_class"].dims, attrs=attrs),
        }

        coords = dict((dim, xx.coords[dim]) for dim in dims)
        return xr.Dataset(data_vars=data_vars, coords=coords, attrs=xx.attrs)


# register("lccs_level3", StatsLccsLevel3)
register("lc_l3_l4", StatsVegDryClassA3)