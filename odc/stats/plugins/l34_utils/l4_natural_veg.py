from odc.stats._algebra import expr_eval

NODATA = 255


def lc_l4_natural_veg(l4, l3, lifeform, veg_cover):

    l4 = expr_eval(
        "where((a==110)&(b==nodata), nodata, a)",
        {"a": l4, "b": l3},
        name="mark_cultivated",
        dtype="uint8",
        **{"nodata": NODATA},
    )
    l4 = expr_eval(
        "where((d==110)&(a==112)&(c==10)&(b==1), 27, d)",
        {"a": l3, "b": lifeform, "c": veg_cover, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((d==110)&(a==112)&(c==12)&(b==1), 28, d)",
        {"a": l3, "b": lifeform, "c": veg_cover, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((d==110)&(a==112)&(c==13)&(b==1), 29, d)",
        {"a": l3, "b": lifeform, "c": veg_cover, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((d==110)&(a==112)&(c==15)&(b==1), 30, d)",
        {"a": l3, "b": lifeform, "c": veg_cover, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((d==110)&(a==112)&(c==16)&(b==1), 31, d)",
        {"a": l3, "b": lifeform, "c": veg_cover, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((d==110)&(a==112)&(c==10)&(b==2), 32, d)",
        {"a": l3, "b": lifeform, "c": veg_cover, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((d==110)&(a==112)&(c==12)&(b==2), 33, d)",
        {"a": l3, "b": lifeform, "c": veg_cover, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((d==110)&(a==112)&(c==13)&(b==2), 34, d)",
        {"a": l3, "b": lifeform, "c": veg_cover, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((d==110)&(a==112)&(c==15)&(b==2), 35, d)",
        {"a": l3, "b": lifeform, "c": veg_cover, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((d==110)&(a==112)&(c==16)&(b==2), 36, d)",
        {"a": l3, "b": lifeform, "c": veg_cover, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((d==110)&(a==112)&(b==1), 20, d)",
        {"a": l3, "b": lifeform, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((d==110)&(a==112)&(b==2), 21, d)",
        {"a": l3, "b": lifeform, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((d==110)&(a==112)&(c==10), 22, d)",
        {"a": l3, "c": veg_cover, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((d==110)&(a==112)&(c==12), 23, d)",
        {"a": l3, "c": veg_cover, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((d==110)&(a==112)&(c==13), 24, d)",
        {"a": l3, "c": veg_cover, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((d==110)&(a==112)&(c==15), 25, d)",
        {"a": l3, "c": veg_cover, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((d==110)&(a==112)&(c==16), 26, d)",
        {"a": l3, "c": veg_cover, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((d==110)&(a==112), 19, d)",
        {"a": l3, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )

    return l4
