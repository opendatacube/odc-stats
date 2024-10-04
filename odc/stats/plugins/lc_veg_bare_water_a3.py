"""
Plugin of Module A3 in LandCover PipeLine
"""

from typing import Tuple, Optional, Dict, List

import xarray as xr
from odc.stats._algebra import expr_eval

from ._registry import StatsPluginInterface, register

NODATA = 255


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
            111: 1,
            112: 19,
            124: 55,
            215: 93,
            216: 94,
            220: 98,
            223: 3,
        }

    @property
    def measurements(self) -> Tuple[str, ...]:
        _measurements = [
            "canopyco_veg_con",
            "baregrad_phy_con",
            "waterper_wat_cin",
            "level3",
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

    def a3_veg_cover(self, xx: xr.Dataset):
        # Mask NODATA
        fcp_nodata = -999
        veg_mask = expr_eval(
            "where(a!=nodata, a, NODATA)",
            {"a": xx.pv_pc_50.data},
            name="mark_nodata",
            dtype="uint8",
            **{"nodata": fcp_nodata, "NODATA": NODATA},
        )

        # ## data<1 ---> 0
        veg_mask = expr_eval(
            "where(a<m, 0, a)",
            {"a": veg_mask},
            name="mark_veg",
            dtype="uint8",
            **{"m": self.veg_threshold[0]},
        )

        # Map any data > 100 ---> 100
        veg_mask = expr_eval(
            "where((a>100) & (a!=nodata), 100, a)",
            {"a": veg_mask},
            name="mark_veg",
            dtype="uint8",
            **{"nodata": NODATA},
        )

        # [1-4) --> 16
        veg_mask = expr_eval(
            "where((a>=m)&(a<n), 160, a)",
            {"a": veg_mask},
            name="mark_veg",
            dtype="uint8",
            **{"m": self.veg_threshold[0], "n": self.veg_threshold[1]},
        )

        # [4-15) --> 15(0)
        veg_mask = expr_eval(
            "where((a>=m)&(a<n), 150, a)",
            {"a": veg_mask},
            name="mark_veg",
            dtype="uint8",
            **{"m": self.veg_threshold[1], "n": self.veg_threshold[2]},
        )

        # [15-40) --> 13(0)
        veg_mask = expr_eval(
            "where((a>=m)&(a<n), 130, a)",
            {"a": veg_mask},
            name="mark_veg",
            dtype="uint8",
            **{"m": self.veg_threshold[2], "n": self.veg_threshold[3]},
        )

        # [40-65) --> 12(0)
        veg_mask = expr_eval(
            "where((a>=m)&(a<n), 120, a)",
            {"a": veg_mask},
            name="mark_veg",
            dtype="uint8",
            **{"m": self.veg_threshold[3], "n": self.veg_threshold[4]},
        )

        # 65-100 --> 10
        veg_mask = expr_eval(
            "where((a>=m)&(a<n), 100, a)",
            {"a": veg_mask},
            name="mark_veg",
            dtype="uint8",
            **{"m": self.veg_threshold[4], "n": self.veg_threshold[5]},
        )

        # Define mapping from current output to expected a3 output
        veg_mask = self.apply_mapping(veg_mask, self.veg_mapping)

        return veg_mask

    def bare_gradation(self, xx: xr.Dataset):

        # Now add the bare gradation
        fcp_nodaata = -999
        bs_mask = expr_eval(
            "where(a!=nodata, a, NODATA)",
            {"a": xx.bs_pc_50.data},
            name="mark_nodata",
            dtype="uint8",
            **{"nodata": fcp_nodaata, "NODATA": NODATA},
        )

        # Map any data > 100 ---> 100
        bs_mask = expr_eval(
            "where((a>100)&(a!=nodata), 100, a)",
            {"a": bs_mask},
            name="mark_veg",
            dtype="uint8",
            **{"nodata": NODATA},
        )

        # 60% <= data  --> 15(0)
        bs_mask = expr_eval(
            "where((a>=m)&(a!=nodata), 150, a)",
            {"a": bs_mask},
            name="mark_veg",
            dtype="uint8",
            **{"m": self.bare_threshold[1], "nodata": NODATA},
        )

        # 20% <= data < 60% --> 12(0)
        bs_mask = expr_eval(
            "where((a>=m)&(a<n), 120, a)",
            {"a": bs_mask},
            name="mark_veg",
            dtype="uint8",
            **{"m": self.bare_threshold[0], "n": self.bare_threshold[1]},
        )

        # data < 20% --> 10(0)
        bs_mask = expr_eval(
            "where(a<m, 100, a)",
            {"a": bs_mask},
            name="mark_veg",
            dtype="uint8",
            **{"m": self.bare_threshold[0]},
        )

        # Apply bare gradation expected output classes
        bs_mask = self.apply_mapping(bs_mask, self.bs_mapping)
        return bs_mask

    def water_persistence(self, xx: xr.Dataset):
        # Now add water persistence
        water_mask = expr_eval(
            "where(a!=a, nodata, a)",
            {"a": xx.water_frequency.data},
            name="mark_water",
            dtype="uint8",
            **{"nodata": NODATA},
        )

        #  10 <= water_frequency < 1 --> 1(0)
        water_mask = expr_eval(
            "where((a>=m)&(a!=nodata), 100, a)",
            {"a": water_mask},
            name="mark_water",
            dtype="uint8",
            **{"m": self.watper_threshold[3], "nodata": NODATA},
        )

        #  7 <= water_frequency < 10 --> 7(0)
        water_mask = expr_eval(
            "where((a>=m)&(a<n), 70, a)",
            {"a": water_mask},
            name="mark_water",
            dtype="uint8",
            **{"m": self.watper_threshold[2], "n": self.watper_threshold[3]},
        )

        #  4 <= water_frequency < 7 --> 8(00)
        water_mask = expr_eval(
            "where((a>=m)&(a<n), 80, a)",
            {"a": water_mask},
            name="mark_water",
            dtype="uint8",
            **{"m": self.watper_threshold[1], "n": self.watper_threshold[2]},
        )

        #  1 <= water_frequency < 4 --> 9(00)
        water_mask = expr_eval(
            "where((a>=m)&(a<n), 90, a)",
            {"a": water_mask},
            name="mark_water",
            dtype="uint8",
            **{"m": self.watper_threshold[0], "n": self.watper_threshold[1]},
        )

        # water_frequency < 1 --> 0
        water_mask = expr_eval(
            "where(a<1, 0, a)",
            {"a": water_mask},
            name="mark_water",
            dtype="uint8",
            **{"m": self.watper_threshold[0]},
        )

        # Apply water persistence expcted classes
        water_mask = self.apply_mapping(water_mask, self.waterper_wat_mapping)

        return water_mask

    def level3_to_level4_mapping(self, xx: xr.Dataset):
        l3_data = xx.level3_class.data
        l3_data = self.apply_mapping(l3_data, self.l3_to_l4_mapping)
        # l3_to_l4_mapping = {111: 1, 112: 19, 124: 55, 215: 93, 216: 94, 220: 98, 223: 3}
        # for o_val, n_val in l3_to_l4_mapping.items():
        #     l3_data = xr.where(l3_data == o_val, n_val, l3_data)
        return l3_data

    def a3_classes(self, xx: xr.Dataset):
        veg_cover = self.a3_veg_cover(xx).compute()
        bare_cover = self.bare_gradation(xx).compute()
        water_persistence = self.water_persistence(xx).compute()
        level_3 = self.level3_to_level4_mapping(xx)

        return veg_cover, bare_cover, water_persistence, level_3

    def reduce(self, xx: xr.Dataset) -> xr.Dataset:
        xx = xx.squeeze(dim=["spec"])
        dims = xx.dims
        a3 = self.a3_classes(xx)

        attrs = xx.attrs.copy()
        attrs["nodata"] = int(NODATA)

        data_vars = {
            "canopyco_veg_con": xr.DataArray(
                a3[0], dims=xx["pv_pc_50"].dims, attrs=attrs
            ),
            "baregrad_phy_con": xr.DataArray(
                a3[1], dims=xx["pv_pc_50"].dims, attrs=attrs
            ),
            "waterper_wat_cin": xr.DataArray(
                a3[2], dims=xx["water_frequency"].dims, attrs=attrs
            ),
            "level3": xr.DataArray(a3[3], dims=xx["level3_class"].dims, attrs=attrs),
        }

        coords = dict((dim, xx.coords[dim]) for dim in dims)
        return xr.Dataset(data_vars=data_vars, coords=coords, attrs=xx.attrs)


register("lc_a3", StatsVegDryClassA3)
