from odc.stats._algebra import expr_eval

def lc_l4_cultivated(level3, lifeform_ds, veg_cover_ds):
            
    l4 = expr_eval(
        "where((a==111)&(b==1), 1, a)",
        {"a": level3,
         "b": lifeform_ds},
        name="mark_cultivated",
        dtype="uint8"
    )

    l4 = expr_eval(
        "where((a==111)&(b==2), 2, a)",
        {"a": l4,
         "b": lifeform_ds},
        name="mark_cultivated",
        dtype="uint8"
    )

    l4 = expr_eval(
        "where((a==111)&(b==10), 4, a)",
        {"a": l4,
        "b": veg_cover_ds},
        name="mark_cultivated",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==111)&(b==12), 5, a)",
        {"a": l4,
        "b": veg_cover_ds},
        name="mark_cultivated",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==111)&(b==13), 6, a)",
        {"a": l4,
        "b": veg_cover_ds},
        name="mark_cultivated",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==111)&(b==15), 7, a)",
        {"a": l4,
        "b": veg_cover_ds},
        name="mark_cultivated",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==111)&(b==16), 8, a)",
        {"a": l4,
        "b": veg_cover_ds},
        name="mark_cultivated",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==111)&(b==10)&(c==1), 9, a)",
        {"a": l4,
        "b": veg_cover_ds,
        "c": lifeform_ds},
        name="mark_cultivated",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==111)&(b==12)&(c==1), 10, a)",
        {"a": l4,
        "b": veg_cover_ds,
        "c": lifeform_ds},
        name="mark_cultivated",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==111)&(b==13)&(c==1), 11, a)",
        {"a": l4,
        "b": veg_cover_ds,
        "c": lifeform_ds},
        name="mark_cultivated",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==111)&(b==15)&(c==1), 12, a)",
        {"a": l4,
        "b": veg_cover_ds,
        "c": lifeform_ds},
        name="mark_cultivated",
        dtype="uint8"
    )

    l4 = expr_eval(
        "where((a==111)&(b==16)&(c==1), 13, a)",
        {"a": l4,
        "b": veg_cover_ds,
        "c": lifeform_ds},
        name="mark_cultivated",
        dtype="uint8"
    )

    l4 = expr_eval(
        "where((a==111)&(b==10)&(c==2), 14, a)",
        {"a": l4,
        "b": veg_cover_ds,
        "c": lifeform_ds},
        name="mark_cultivated",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==111)&(b==12)&(c==2), 15, a)",
        {"a": l4,
        "b": veg_cover_ds,
        "c": lifeform_ds},
        name="mark_cultivated",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==111)&(b==13)&(c==2), 16, a)",
        {"a": l4,
        "b": veg_cover_ds,
        "c": lifeform_ds},
        name="mark_cultivated",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==111)&(b==15)&(c==2), 17, a)",
        {"a": l4,
        "b": veg_cover_ds,
        "c": lifeform_ds},
        name="mark_cultivated",
        dtype="uint8"
    )

    l4 = expr_eval(
        "where((a==111)&(b==16)&(c==2), 18, a)",
        {"a": l4,
        "b": veg_cover_ds,
        "c": lifeform_ds},
        name="mark_cultivated",
        dtype="uint8"
    )

    return l4

