import xarray as xr

from odc.stats._algebra import expr_eval

NODATA = 255


def lc_level3(xx: xr.Dataset):

    # Cultivated pipeline applies a mask which feeds only terrestrial veg (110) to the model
    # Just exclude no data (255) and apply the cultivated results
    res = expr_eval(
        "where(a<nodata, a, b)",
        {"a": xx.cultivated_class.data, "b": xx.classes_l3_l4.data},
        name="mask_cultivated",
        dtype="float32",
        **{"nodata": NODATA},
    )

    # Mask urban results with bare sfc (210)
    res = expr_eval(
        "where(a==_u, b, a)",
        {
            "a": res,
            "b": xx.urban_classes.data,
        },
        name="mark_urban",
        dtype="uint8",
        **{"_u": 210},
    )

    intertidal_mask = expr_eval(
        "where(a==_u, 1, 0)",
        {
            "a": res,
            "b": xx.urban_classes.data,
        },
        name="mark_urban",
        dtype="uint8",
        **{"_u": 223},
    )
    # Add intertidal as water
    res = expr_eval(
        "where((a==223)|(a==221)|(c==1), 220, b)",
        {"a": xx.classes_l3_l4.data, "b": res, "c": intertidal_mask},
        name="mark_urban",
        dtype="uint8",
    )

    return intertidal_mask, res
