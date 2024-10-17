 
from odc.stats._algebra import expr_eval

def lc_l4_natural_veg(l4, lifeform, veg_cover):
    l4 = expr_eval(
        "where((a==112), 19, a)",
        {"a": l4},
        name="mark_cultivated",
        dtype="uint8"
    )

    l4 = expr_eval(
        "where((a==112)&(b==1), 20, a)",
        {"a": l4,
         "b": lifeform},
        name="mark_cultivated",
        dtype="uint8"
    )

    l4 = expr_eval(
        "where((a==112)&(b==2), 21, a)",
        {"a": l4,
         "b": lifeform},
        name="mark_cultivated",
        dtype="uint8"
    )

    l4 = expr_eval(
        "where((a==112)&(c==10), 22, a)",
        {"a": l4,
         "c": veg_cover},
        name="mark_cultivated",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==112)&(c==12), 23, a)",
        {"a": l4,
         "c": veg_cover},
        name="mark_cultivated",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==112)&(c==13), 24, a)",
        {"a": l4,
         "c": veg_cover},
        name="mark_cultivated",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==112)&(c==15), 25, a)",
        {"a": l4,
         "c": veg_cover},
        name="mark_cultivated",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==112)&(c==16), 26, a)",
        {"a": l4,
         "c": veg_cover},
        name="mark_cultivated",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==112)&(c==10)&(b==1), 27, a)",
        {"a": l4,
         "b": lifeform,
         "c": veg_cover,},
        name="mark_cultivated",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==112)&(c==12)&(b==1), 28, a)",
        {"a": l4,
         "b": lifeform,
         "c": veg_cover,},
        name="mark_cultivated",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==112)&(c==13)&(b==1), 29, a)",
        {"a": l4,
         "b": lifeform,
         "c": veg_cover,},
        name="mark_cultivated",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==112)&(c==15)&(b==1), 30, a)",
        {"a": l4,
         "b": lifeform,
         "c": veg_cover,},
        name="mark_cultivated",
        dtype="uint8"
    )

    l4 = expr_eval(
        "where((a==112)&(c==16)&(b==1), 31, a)",
        {"a": l4,
         "b": lifeform,
         "c": veg_cover,},
        name="mark_cultivated",
        dtype="uint8"
    )

    l4 = expr_eval(
        "where((a==112)&(c==10)&(b==2), 32, a)",
        {"a": l4,
         "b": lifeform,
         "c": veg_cover,},
        name="mark_cultivated",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==112)&(c==12)&(b==2), 33, a)",
        {"a": l4,
         "b": lifeform,
         "c": veg_cover,},
        name="mark_cultivated",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==112)&(c==13)&(b==2), 34, a)",
        {"a": l4,
         "b": lifeform,
         "c": veg_cover,},
        name="mark_cultivated",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==112)&(c==15)&(b==2), 35, a)",
        {"a": l4,
         "b": lifeform,
         "c": veg_cover,},
        name="mark_cultivated",
        dtype="uint8"
    )

    l4 = expr_eval(
        "where((a==112)&(c==16)&(b==2), 36, a)",
        {"a": l4,
         "b": lifeform,
         "c": veg_cover,},
        name="mark_cultivated",
        dtype="uint8"
    )

    return l4
