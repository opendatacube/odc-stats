from odc.stats._algebra import expr_eval

NODATA = 255


def water_classification(xx, intertidal_mask, water_persistence):

    l4 = expr_eval(
        "where(((a==223)|(a==221))&(b==1), 101, a)",
        {"a": xx.classes_l3_l4.data, "b": water_persistence},
        name="mark_water",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where(((a==223)|(a==221))&(b==7), 102, a)",
        {"a": l4, "b": water_persistence},
        name="mark_water",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where(((a==223)|(a==221))&(b==8), 103, a)",
        {"a": l4, "b": water_persistence},
        name="mark_water",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where(((a==223)|(a==221))&(b==9), 104, a)",
        {"a": l4, "b": water_persistence},
        name="mark_water",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where(((a==223)|(a==221))&(b==1), 100, a)",
        {"a": l4, "b": intertidal_mask},
        name="mark_water",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==223)|(a==221), 98, a)", {"a": l4}, name="mark_water", dtype="uint8"
    )

    return l4
