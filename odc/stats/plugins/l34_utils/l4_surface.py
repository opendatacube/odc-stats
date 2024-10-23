from odc.stats._algebra import expr_eval


def lc_l4_surface(l4, level3, bare_gradation):

    l4 = expr_eval(
        "where((a==210)&(b==10)&(c==216), 95, a)",
        {"a": l4, "b": bare_gradation, "c": level3},
        name="mark_surface",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((a==210)&(b==12)&(c==216), 96, a)",
        {"a": l4, "b": bare_gradation, "c": level3},
        name="mark_surface",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((a==210)&(b==15)&(c==216), 97, a)",
        {"a": l4, "b": bare_gradation, "c": level3},
        name="mark_surface",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((a==210)&(c==215), 93, a)",
        {"a": l4, "c": level3},
        name="mark_surface",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((a==210)&(c==216), 94, a)",
        {"a": l4, "c": level3},
        name="mark_surface",
        dtype="uint8",
    )
    return l4
