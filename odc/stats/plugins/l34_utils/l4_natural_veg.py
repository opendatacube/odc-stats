from odc.stats._algebra import expr_eval

NODATA = 255


def lc_l4_natural_veg(l4, l3, woody, veg_cover):

    woody = expr_eval(
        "where((a!=a), nodata, a)",
        {"a": woody.data},
        name="mask_woody_nodata",
        dtype="float32",
        **{"nodata": NODATA},
    )

    l4 = expr_eval(
        "where((b==nodata), nodata, a)",
        {"a": l4, "b": l3},
        name="mark_cultivated",
        dtype="uint8",
        **{"nodata": NODATA},
    )

    l4 = expr_eval(
        "where((a==112)&(b==113), 20, d)",
        {"a": l3, "b": woody, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==112)&(b==114), 21, d)",
        {"a": l3, "b": woody, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==112)&(c==10), 22, d)",
        {"a": l3, "c": veg_cover, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((a==112)&(c==12), 23, d)",
        {"a": l3, "c": veg_cover, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((a==112)&(c==13), 24, d)",
        {"a": l3, "c": veg_cover, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((a==112)&(c==15), 25, d)",
        {"a": l3, "c": veg_cover, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((a==112)&(c==16), 26, d)",
        {"a": l3, "c": veg_cover, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==112)&(c==10)&(b==113), 27, d)",
        {"a": l3, "b": woody, "c": veg_cover, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((a==112)&(c==12)&(b==113), 28, d)",
        {"a": l3, "b": woody, "c": veg_cover, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((a==112)&(c==13)&(b==113), 29, d)",
        {"a": l3, "b": woody, "c": veg_cover, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((a==112)&(c==15)&(b==113), 30, d)",
        {"a": l3, "b": woody, "c": veg_cover, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==112)&(c==16)&(b==113), 31, d)",
        {"a": l3, "b": woody, "c": veg_cover, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==112)&(c==10)&(b==114), 32, d)",
        {"a": l3, "b": woody, "c": veg_cover, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((a==112)&(c==12)&(b==114), 33, d)",
        {"a": l3, "b": woody, "c": veg_cover, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((a==112)&(c==13)&(b==114), 34, d)",
        {"a": l3, "b": woody, "c": veg_cover, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((a==112)&(c==15)&(b==114), 35, d)",
        {"a": l3, "b": woody, "c": veg_cover, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==112)&(c==16)&(b==114), 36, d)",
        {"a": l3, "b": woody, "c": veg_cover, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )

    return l4
