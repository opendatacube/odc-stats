# from typing import Tuple, Optional, Dict, List
import xarray as xr
from odc.stats._algebra import expr_eval

NODATA = 255


def canopyco_veg_con(xx: xr.Dataset, veg_threshold):

    # Mask NODATA
    pv_pc_50 = expr_eval(
        "where(a==a, a, nodata)",
        {"a": xx.pv_pc_50.data},
        name="mark_nodata",
        dtype="float32",
        **{"nodata": NODATA},
    )

    # Map any data > 100 ---> 100
    pv_pc_50 = expr_eval(
        "where((a>100) & (a!=nodata), 100, a)",
        {
            "a": pv_pc_50,
        },
        name="mark_veg",
        dtype="uint8",
        **{"nodata": NODATA},
    )

    # ## data < 1 ---> 0
    veg_mask = expr_eval(
        "where(a<m, 0, a)",
        {
            "a": pv_pc_50,
        },
        name="mark_veg",
        dtype="uint8",
        **{"m": veg_threshold[0]},
    )

    # [1-4) --> 16
    veg_mask = expr_eval(
        "where((a>=m)&(a<n), 16, b)",
        {
            "a": pv_pc_50,
            "b": veg_mask,
        },
        name="mark_veg",
        dtype="uint8",
        **{"m": veg_threshold[0], "n": veg_threshold[1]},
    )

    # [4-15) --> 15
    veg_mask = expr_eval(
        "where((a>=m)&(a<n), 15, b)",
        {
            "a": pv_pc_50,
            "b": veg_mask,
        },
        name="mark_veg",
        dtype="uint8",
        **{"m": veg_threshold[1], "n": veg_threshold[2]},
    )

    # [15-40) --> 13
    veg_mask = expr_eval(
        "where((a>=m)&(a<n), 13, b)",
        {
            "a": pv_pc_50,
            "b": veg_mask,
        },
        name="mark_veg",
        dtype="uint8",
        **{"m": veg_threshold[2], "n": veg_threshold[3]},
    )

    # [40-65) --> 12
    veg_mask = expr_eval(
        "where((a>=m)&(a<n), 12, b)",
        {
            "a": pv_pc_50,
            "b": veg_mask,
        },
        name="mark_veg",
        dtype="uint8",
        **{"m": veg_threshold[3], "n": veg_threshold[4]},
    )

    # 65-100 --> 10
    veg_mask = expr_eval(
        "where((a>=m)&(a<n), 10, b)",
        {
            "a": pv_pc_50,
            "b": veg_mask,
        },
        name="mark_veg",
        dtype="uint8",
        **{"m": veg_threshold[4], "n": veg_threshold[5]},
    )

    return veg_mask
