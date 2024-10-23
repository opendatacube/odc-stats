from odc.stats._algebra import expr_eval


def water_classification(l4, level3, intertidal_mask, water_persistence, NODATA):

    l4 = expr_eval(
        "where((c==223)&(a==220)&b, 100, c)",
        {"a": level3, "b": intertidal_mask, "c": l4},
        name="mark_water",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((c==221)&(a==220)&(b==1), 101, c)",
        {"a": level3, "b": water_persistence, "c": l4},
        name="mark_water",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((c==221)&(a==220)&(b==7), 102, c)",
        {"a": level3, "b": water_persistence, "c": l4},
        name="mark_water",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((c==221)&(a==220)&(b==8), 103, c)",
        {"a": level3, "b": water_persistence, "c": l4},
        name="mark_water",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((c==221)&(a==220)&(b==9), 104, c)",
        {"a": level3, "b": water_persistence, "c": l4},
        name="mark_water",
        dtype="uint8",
    )

    # L34 water:  (water_freq >= 0.2)
    l4 = expr_eval(
        "where((a==221)&(b==220)&(c==nodata), 99, a)",
        {"a": l4, "b": level3, "c": water_persistence},
        name="mark_water",
        dtype="uint8",
        **{"nodata": NODATA},
    )

    l4 = expr_eval(
        "where((a==221), 98, a)", {"a": l4}, name="mark_water", dtype="uint8"
    )

    return l4
