import xarray as xr

from odc.stats._algebra import expr_eval

NODATA = 255


def lc_level3(xx: xr.Dataset):

    # Cultivated pipeline applies a mask which feeds only terrestrial veg (110) to the model
    # Just exclude no data (255 or nan) and apply the cultivated results
    # 255: load with product definition; nan: load without
    # hence accormmodate both

    res = expr_eval(
        "where((a!=a)|(a>=nodata), b, a)",
        {"a": xx.cultivated.data, "b": xx.classes_l3_l4.data},
        name="mask_cultivated",
        dtype="float32",
        **{"nodata": xx.cultivated.attrs.get("nodata")},
    )

    # Mask urban results with bare sfc (210)

    res = expr_eval(
        "where(a==_u, b, a)",
        {
            "a": res,
            "b": xx.urban_classes.data,
        },
        name="mark_urban",
        dtype="float32",
        **{"_u": 210},
    )

    # Mark nodata to 255 in case any nan
    res = expr_eval(
        "where(a==a, a, nodata)",
        {
            "a": res,
        },
        name="mark_nodata",
        dtype="uint8",
        **{"nodata": NODATA},
    )
    # Add intertidal as water
    res = expr_eval(
        "where((a==223)|(a==221), 220, b)",
        {"a": xx.classes_l3_l4.data, "b": res},
        name="mark_urban",
        dtype="uint8",
    )

    return res
