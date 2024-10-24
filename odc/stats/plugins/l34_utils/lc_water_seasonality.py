import xarray as xr

from odc.stats._algebra import expr_eval
from . import utils

WATER_FREQ_NODATA = -999


def water_seasonality(self, xx: xr.Dataset):
    # >= 3 months ----> 1  Semi-permanent or permanent
    # < 3 months  ----> 2 Temporary or seasonal

    water_season_mask = expr_eval(
        "where((a>watseas_trh)&(a<=12), 100, a)",
        {"a": xx.water_frequency.data},
        name="mark_water_season",
        dtype="uint8",
        **{"watseas_trh": self.water_seasonality_threshold},
    )
    water_season_mask = expr_eval(
        "where((a<=watseas_trh)&(a<=12), 200, a)",
        {"a": water_season_mask},
        name="mark_water_season",
        dtype="uint8",
        **{"watseas_trh": self.water_seasonality_threshold},
    )
    water_season_mask = expr_eval(
        "where((a==watersea_nodata), 255, a)",
        {"a": water_season_mask},
        name="mark_water_season",
        dtype="uint8",
        **{
            "watseas_trh": self.water_seasonality_threshold,
            "watersea_nodata": WATER_FREQ_NODATA,
        },
    )
    mapping = {100: 1, 200: 2}
    water_season_mask = utils.apply_mapping(water_season_mask, mapping)

    return water_season_mask
