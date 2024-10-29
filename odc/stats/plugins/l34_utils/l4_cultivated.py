from odc.stats._algebra import expr_eval

NODATA = 255


def lc_l4_cultivated(l34, level3, lifeform, veg_cover):

    l4 = expr_eval(
        "where((d==110)&(a==111)&(b==10)&(c==1), 9, d)",
        {"a": level3, "b": veg_cover, "c": lifeform, "d": l34},
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((d==110)&(a==111)&(b==12)&(c==1), 10, d)",
        {"a": level3, "b": veg_cover, "c": lifeform, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((d==110)&(a==111)&(b==13)&(c==1), 11, d)",
        {"a": level3, "b": veg_cover, "c": lifeform, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((d==110)&(a==111)&(b==15)&(c==1), 12, d)",
        {"a": level3, "b": veg_cover, "c": lifeform, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((d==110)&(a==111)&(b==16)&(c==1), 13, d)",
        {"a": level3, "b": veg_cover, "c": lifeform, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((d==110)&(a==111)&(b==10)&(c==2), 14, d)",
        {"a": level3, "b": veg_cover, "c": lifeform, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((d==110)&(a==111)&(b==12)&(c==2), 15, d)",
        {"a": level3, "b": veg_cover, "c": lifeform, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((d==110)&(a==111)&(b==13)&(c==2), 16, d)",
        {"a": level3, "b": veg_cover, "c": lifeform, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((d==110)&(a==111)&(b==15)&(c==2), 17, d)",
        {"a": level3, "b": veg_cover, "c": lifeform, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((d==110)&(a==111)&(b==16)&(c==2), 18, d)",
        {"a": level3, "b": veg_cover, "c": lifeform, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((d==110)&(a==111)&(b==1), 2, d)",
        {"a": level3, "b": lifeform, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((d==110)&(a==111)&(b==2), 3, d)",
        {"a": level3, "b": lifeform, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )

    # the 4-8 classes can't happen in LC since cultivated class will not be classified if vegetation doesn't exist.
    # skip these classes in level4

    l4 = expr_eval(
        "where((d==110)&(a==111), 1, d)",
        {"a": level3, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )

    return l4
