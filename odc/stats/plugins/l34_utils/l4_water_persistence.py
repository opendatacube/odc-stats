import xarray as xr

from odc.stats._algebra import expr_eval
from . import utils

NODATA = 255


def water_persistence(xx: xr.Dataset, watper_threshold):
    # Now add water persistence
    # water_mask = expr_eval(
    #     "where(a!=a, nodata, a)",
    #     {"a": xx.water_frequency.data},
    #     name="mark_water",
    #     dtype="uint8",
    #     **{"nodata": NODATA},
    # )

    #  10 <= water_frequency < 1 --> 1(0)
    water_mask = expr_eval(
        "where((a>=m)&(a!=nodata), 100, a)",
        {"a": xx.water_frequency.data},
        name="mark_water",
        dtype="uint8",
        **{"m": watper_threshold[3], "nodata": NODATA},
    )

    #  7 <= water_frequency < 10 --> 7(0)
    water_mask = expr_eval(
        "where((a>=m)&(a<n), 70, a)",
        {"a": water_mask},
        name="mark_water",
        dtype="uint8",
        **{"m": watper_threshold[2], "n": watper_threshold[3]},
    )

    #  4 <= water_frequency < 7 --> 8(00)
    water_mask = expr_eval(
        "where((a>=m)&(a<n), 80, a)",
        {"a": water_mask},
        name="mark_water",
        dtype="uint8",
        **{"m": watper_threshold[1], "n": watper_threshold[2]},
    )

    #  1 <= water_frequency < 4 --> 9(00)
    water_mask = expr_eval(
        "where((a>=m)&(a<n), 90, a)",
        {"a": water_mask},
        name="mark_water",
        dtype="uint8",
        **{"m": watper_threshold[0], "n": watper_threshold[1]},
    )

    # Apply water persistence expcted classes
    # Map values to the classes expected in water persistence in land cover Level-4 output
    waterper_wat_mapping = {100: 1, 70: 7, 80: 8, 90: 9}
    water_mask = utils.apply_mapping(water_mask, waterper_wat_mapping)

    # # water_frequency < 1 --> 0
    # water_mask = expr_eval(
    #     "where(a<1, 0, a)",
    #     {"a": water_mask},
    #     name="mark_water",
    #     dtype="uint8",
    #     **{"m": watper_threshold[0]},
    # )

    return water_mask
