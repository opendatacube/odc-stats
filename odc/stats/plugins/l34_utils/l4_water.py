from odc.stats._algebra import expr_eval

NODATA = 255


def water_classification(xx, water_persistence):

    # Replace nan with nodata
    l4 = expr_eval(
        "where((a==a), a, nodata)",
        {"a": xx.level_3_4.data},
        name="mark_water",
        dtype="uint8",
        **{"nodata": NODATA},
    )

    l4 = expr_eval(
        "where((a==223)|(a==221), 98, a)", {"a": l4}, name="mark_water", dtype="uint8"
    )

    l4 = expr_eval(
        "where((a==98)&(b!=_u), 99, a)",
        {"a": l4, "b": xx.level_3_4.data},
        name="mark_water",
        dtype="uint8",
        **{"_u": 223},
    )

    l4 = expr_eval(
        "where((a==98)&(b==_u), 100, a)",
        {"a": l4, "b": xx.level_3_4.data},
        name="mark_water",
        dtype="uint8",
        **{"_u": 223},
    )

    l4 = expr_eval(
        "where((a==99)&(b==1), 101, a)",
        {"a": l4, "b": water_persistence},
        name="mark_water",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==99)&(b==7), 102, a)",
        {"a": l4, "b": water_persistence},
        name="mark_water",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==99)&(b==8), 103, a)",
        {"a": l4, "b": water_persistence},
        name="mark_water",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==99)&(b==9), 104, a)",
        {"a": l4, "b": water_persistence},
        name="mark_water",
        dtype="uint8",
    )

    return l4
