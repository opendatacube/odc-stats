import xarray as xr

from odc.stats._algebra import expr_eval

NODATA = 255
WATER_FREQ_NODATA = -999


def water_persistence(xx: xr.Dataset, watper_threshold):

    # Apply nodata
    water_frequency = expr_eval(
        "where((a==water_freq_nodata), nodata, a)",
        {"a": xx.water_frequency.data},
        name="mark_water",
        dtype="uint8",
        **{
            "m": watper_threshold[3],
            "nodata": NODATA,
            "water_freq_nodata": WATER_FREQ_NODATA,
        },
    )

    #  10 <= water_frequency < 1 --> 1
    water_mask = expr_eval(
        "where((a>=m)&(a!=nodata), 1, a)",
        {"a": water_frequency},
        name="mark_water",
        dtype="uint8",
        **{"m": watper_threshold[3], "nodata": NODATA},
    )

    #  7 <= water_frequency < 10 --> 7
    water_mask = expr_eval(
        "where((a>=m)&(a<n), 7, b)",
        {"a": water_frequency, "b": water_mask},
        name="mark_water",
        dtype="uint8",
        **{"m": watper_threshold[2], "n": watper_threshold[3]},
    )

    #  4 <= water_frequency < 7 --> 8
    water_mask = expr_eval(
        "where((a>=m)&(a<n), 8, b)",
        {"a": water_frequency, "b": water_mask},
        name="mark_water",
        dtype="uint8",
        **{"m": watper_threshold[1], "n": watper_threshold[2]},
    )

    #  1 <= water_frequency < 4 --> 9
    water_mask = expr_eval(
        "where((a>=m)&(a<n), 9, b)",
        {"a": water_frequency, "b": water_mask},
        name="mark_water",
        dtype="uint8",
        **{"m": watper_threshold[0], "n": watper_threshold[1]},
    )

    return water_mask
