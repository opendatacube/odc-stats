from odc.stats._algebra import expr_eval

def lc_l4_cultivated(level3, lifeform, veg_cover):

    
    l4 = expr_eval(
        "where((a==111)&(b==10)&(c==1), 9, a)",
        {"a": level3,
        "b": veg_cover,
        "c": lifeform},
        name="mark_cultivated",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==111)&(b==12)&(c==1), 10, a)",
        {"a": l4,
        "b": veg_cover,
        "c": lifeform},
        name="mark_cultivated",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==111)&(b==13)&(c==1), 11, a)",
        {"a": l4,
        "b": veg_cover,
        "c": lifeform},
        name="mark_cultivated",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==111)&(b==15)&(c==1), 12, a)",
        {"a": l4,
        "b": veg_cover,
        "c": lifeform},
        name="mark_cultivated",
        dtype="uint8"
    )

    l4 = expr_eval(
        "where((a==111)&(b==16)&(c==1), 13, a)",
        {"a": l4,
        "b": veg_cover,
        "c": lifeform},
        name="mark_cultivated",
        dtype="uint8"
    )

    l4 = expr_eval(
        "where((a==111)&(b==10)&(c==2), 14, a)",
        {"a": l4,
        "b": veg_cover,
        "c": lifeform},
        name="mark_cultivated",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==111)&(b==12)&(c==2), 15, a)",
        {"a": l4,
        "b": veg_cover,
        "c": lifeform},
        name="mark_cultivated",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==111)&(b==13)&(c==2), 16, a)",
        {"a": l4,
        "b": veg_cover,
        "c": lifeform},
        name="mark_cultivated",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==111)&(b==15)&(c==2), 17, a)",
        {"a": l4,
        "b": veg_cover,
        "c": lifeform},
        name="mark_cultivated",
        dtype="uint8"
    )

    l4 = expr_eval(
        "where((a==111)&(b==16)&(c==2), 18, a)",
        {"a": l4,
        "b": veg_cover,
        "c": lifeform},
        name="mark_cultivated",
        dtype="uint8"
    )

    l4 = expr_eval(
        "where((a==111)&(b==1), 2, a)",
        {"a": l4,
         "b": lifeform},
        name="mark_cultivated",
        dtype="uint8"
    )
    
    l4 = expr_eval(
        "where((a==111)&(b==2), 3, a)",
        {"a": l4,
         "b": lifeform},
        name="mark_cultivated",
        dtype="uint8"
    )

    # the 4-8 classes can't happen in LC since cultivated class will not be classified if vegetation doesn't exist.
    # skip these classes in level4

    l4 = expr_eval(
        "where((a==111), 1, a)",
        {"a": l4},
        name="mark_cultivated",
        dtype="uint8"
    )
  

    return l4

