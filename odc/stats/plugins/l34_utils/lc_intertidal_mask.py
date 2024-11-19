import xarray as xr

from odc.stats._algebra import expr_eval

NODATA = 255


def intertidal_mask(xx: xr.Dataset):

    res = expr_eval(
        "where(a==_u, 1, 0)",
        {"a": xx.level_3_4.data},
        name="mask_intertidal",
        dtype="uint8",
        **{"_u": 223},
    )

    return res
