import xarray as xr

from odc.stats._algebra import expr_eval

NODATA = 255


def lc_level3(xx: xr.Dataset, urban_mask):

    # Cultivated pipeline applies a mask which feeds only terrestrial veg (110) to the model
    # Just exclude no data (255 or nan) and apply the cultivated results
    # 255: load with product definition; nan: load without
    # hence accormmodate both

    res = expr_eval(
        "where((a!=a)|(a>=nodata), b, a)",
        {"a": xx.cultivated.data, "b": xx.level_3_4.data},
        name="mask_cultivated",
        dtype="float32",
        **{"nodata": xx.cultivated.attrs.get("nodata")},
    )

    # Mask urban results with bare sfc (210)

    res = expr_eval(
        "where((a==_u), b, a)",
        {
            "a": res,
            "b": xx.artificial_surface.data,
        },
        name="mark_urban",
        dtype="float32",
        **{"_u": 210},
    )

    # Enforce non-urban mask area to be n/artificial (216)

    res = expr_eval(
        "where((b<=0)&(a==_u), _nu, a)",
        {
            "a": res,
            "b": urban_mask,
        },
        name="mask_non_urban",
        dtype="float32",
        **{"_u": 215, "_nu": 216},
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
        {"a": xx.level_3_4.data, "b": res},
        name="mark_urban",
        dtype="uint8",
    )

    # Combine woody and herbaceous aquatic vegetation
    res = expr_eval(
        "where((a==124)|(a==125), 124, b)",
        {"a": xx.level_3_4.data, "b": res},
        name="mark_aquatic_veg",
        dtype="uint8",
    )

    return res
