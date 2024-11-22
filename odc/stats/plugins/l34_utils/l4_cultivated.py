from odc.stats._algebra import expr_eval

NODATA = 255


def lc_l4_cultivated(l34, level3, woody, veg_cover):

    woody = expr_eval(
        "where((a!=a), nodata, a)",
        {"a": woody.data},
        name="mask_woody_nodata",
        dtype="float32",
        **{"nodata": NODATA},
    )

    l4 = expr_eval(
        "where((a==111)&(b==113), 2, d)",
        {"a": level3, "b": woody, "d": l34},
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==111)&(b==114), 3, d)",
        {"a": level3, "b": woody, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )

    # the 4-8 classes can't happen in LC since cultivated class will not be classified if vegetation doesn't exist.
    # skip these classes in level4

    l4 = expr_eval(
        "where((a==111)&(b==10)&(c==113), 9, d)",
        {"a": level3, "b": veg_cover, "c": woody, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==111)&(b==12)&(c==113), 10, d)",
        {"a": level3, "b": veg_cover, "c": woody, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((a==111)&(b==13)&(c==113), 11, d)",
        {"a": level3, "b": veg_cover, "c": woody, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((a==111)&(b==15)&(c==113), 12, d)",
        {"a": level3, "b": veg_cover, "c": woody, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==111)&(b==16)&(c==113), 13, d)",
        {"a": level3, "b": veg_cover, "c": woody, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==111)&(b==10)&(c==114), 14, d)",
        {"a": level3, "b": veg_cover, "c": woody, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((a==111)&(b==12)&(c==114), 15, d)",
        {"a": level3, "b": veg_cover, "c": woody, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((a==111)&(b==13)&(c==114), 16, d)",
        {"a": level3, "b": veg_cover, "c": woody, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((a==111)&(b==15)&(c==114), 17, d)",
        {"a": level3, "b": veg_cover, "c": woody, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==111)&(b==16)&(c==114), 18, d)",
        {"a": level3, "b": veg_cover, "c": woody, "d": l4},
        name="mark_cultivated",
        dtype="uint8",
    )

    return l4
