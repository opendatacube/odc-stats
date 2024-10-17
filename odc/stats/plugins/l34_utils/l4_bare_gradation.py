
import xarray as xr
from odc.stats._algebra import expr_eval

def bare_gradation(xx: xr.Dataset, bare_threshold, veg_cover, NODATA):

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
            **{"m": bare_threshold[1], "nodata": NODATA},
        )

        # 20% <= data < 60% --> 12(0)
        bs_mask = expr_eval(
            "where((a>=m)&(a<n), 120, a)",
            {"a": bs_mask},
            name="mark_veg",
            dtype="uint8",
            **{"m": bare_threshold[0], "n": bare_threshold[1]},
        )

        # data < 20% --> 10(0)
        bs_mask = expr_eval(
            "where(a<m, 100, a)",
            {"a": bs_mask},
            name="mark_veg",
            dtype="uint8",
            **{"m": bare_threshold[0]},
        )
        
        return bs_mask