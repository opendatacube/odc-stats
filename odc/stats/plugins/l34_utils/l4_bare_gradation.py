import xarray as xr
from odc.stats._algebra import expr_eval


NODATA = 255


def bare_gradation(xx: xr.Dataset, bare_threshold, veg_cover):

    # Map any data > 100 ---> 100
    bs_pc_50 = expr_eval(
        "where((a>100)&(a!=nodata), 100, a)",
        {"a": xx.bs_pc_50.data},
        name="mark_veg",
        dtype="uint8",
        **{"nodata": NODATA},
    )

    # 60% <= data  --> 15
    bs_mask = expr_eval(
        "where((a>=m)&(a!=nodata), 15, a)",
        {"a": bs_pc_50},
        name="mark_veg",
        dtype="uint8",
        **{"m": bare_threshold[1], "nodata": NODATA},
    )

    # 20% <= data < 60% --> 12
    bs_mask = expr_eval(
        "where((a>=m)&(a<n), 12, b)",
        {"a": bs_pc_50, "b": bs_mask},
        name="mark_veg",
        dtype="uint8",
        **{"m": bare_threshold[0], "n": bare_threshold[1]},
    )

    # data < 20% --> 10
    bs_mask = expr_eval(
        "where(a<m, 10, b)",
        {"a": bs_pc_50, "b": bs_mask},
        name="mark_veg",
        dtype="uint8",
        **{"m": bare_threshold[0]},
    )

    return bs_mask
