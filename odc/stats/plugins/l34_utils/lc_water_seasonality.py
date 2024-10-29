import xarray as xr

from odc.stats._algebra import expr_eval

WATER_FREQ_NODATA = -999
NODATA = 255


def water_seasonality(xx: xr.Dataset, water_seasonality_threshold):
    # >= 3 months ----> 1  Semi-permanent or permanent
    # < 3 months  ----> 2 Temporary or seasonal

    # Apply nodata
    water_frequency = expr_eval(
        "where((a==watersea_nodata), nodata, a)",
        {"a": xx.water_frequency.data},
        name="mark_water_season",
        dtype="uint8",
        **{
            "watersea_nodata": WATER_FREQ_NODATA,
             "nodata": NODATA
        },
    )

    water_season_mask = expr_eval(
        "where((a>watseas_trh)&(a<=12), 1, a)",
        {"a": water_frequency},
        name="mark_water_season",
        dtype="uint8",
        **{"watseas_trh": water_seasonality_threshold,
           "nodata": NODATA},
    )
    water_season_mask = expr_eval(
        "where((a<=watseas_trh)&(a<=12), 2, b)",
        {"a": water_frequency,
         "b": water_season_mask},
        name="mark_water_season",
        dtype="uint8",
        **{"watseas_trh": water_seasonality_threshold},
    )
   
    # mapping = {100: 1, 200: 2}
    # water_season_mask = utils.apply_mapping(water_season_mask, mapping)

    return water_season_mask
