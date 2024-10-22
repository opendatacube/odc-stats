
import xarray as xr
 
from odc.stats._algebra import expr_eval

def water_persistence(xx: xr.Dataset, watper_threshold, NODATA):
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
    
    # # water_frequency < 1 --> 0
    # water_mask = expr_eval(
    #     "where(a<1, 0, a)",
    #     {"a": water_mask},
    #     name="mark_water",
    #     dtype="uint8",
    #     **{"m": watper_threshold[0]},
    # )
    
    return water_mask