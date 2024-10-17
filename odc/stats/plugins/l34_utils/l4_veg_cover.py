
# from typing import Tuple, Optional, Dict, List
import xarray as xr
from odc.stats._algebra import expr_eval


def canopyco_veg_con(xx: xr.Dataset, veg_threshold, NODATA, fcp_nodata):

    # Mask NODATA
    veg_mask = expr_eval(
        "where(a!=nodata, a, NODATA)",
        {"a": xx.pv_pc_50.data},
        name="mark_nodata",
        dtype="uint8",
        **{"nodata": fcp_nodata, "NODATA": NODATA},
    )

    # ## data < 1 ---> 0
    veg_mask = expr_eval(
        "where(a<m, 0, a)",
        {"a": veg_mask},
        name="mark_veg",
        dtype="uint8",
        **{"m": veg_threshold[0]},
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
        **{"m": veg_threshold[0], "n": veg_threshold[1]},
    )

    # [4-15) --> 15(0)
    veg_mask = expr_eval(
        "where((a>=m)&(a<n), 150, a)",
        {"a": veg_mask},
        name="mark_veg",
        dtype="uint8",
        **{"m": veg_threshold[1], "n": veg_threshold[2]},
    )

    # [15-40) --> 13(0)
    veg_mask = expr_eval(
        "where((a>=m)&(a<n), 130, a)",
        {"a": veg_mask},
        name="mark_veg",
        dtype="uint8",
        **{"m": veg_threshold[2], "n": veg_threshold[3]},
    )

    # [40-65) --> 12(0)
    veg_mask = expr_eval(
        "where((a>=m)&(a<n), 120, a)",
        {"a": veg_mask},
        name="mark_veg",
        dtype="uint8",
        **{"m": veg_threshold[3], "n": veg_threshold[4]},
    )

    # 65-100 --> 10
    veg_mask = expr_eval(
        "where((a>=m)&(a<n), 100, a)",
        {"a": veg_mask},
        name="mark_veg",
        dtype="uint8",
        **{"m": veg_threshold[4], "n": veg_threshold[5]},
    )


    return veg_mask

