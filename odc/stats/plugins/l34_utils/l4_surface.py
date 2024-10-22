from odc.stats._algebra import expr_eval

def lc_l4_surface(l4, level3, bare_gradation):
    surface_mask = level3 == 216   
    bare_mask = level3 == 215
    l4 = expr_eval(
        "where((a==216)&(b==10)&(c==216), 95, a)",
        {"a": l4,
         "b": bare_gradation,
         "c": level3},
        name="mark_surface",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==216)&(b==12)&(c==216), 96, a)",
        {"a": l4,
         "b": bare_gradation,
         "c": level3},
        name="mark_surface",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==216)&(b==15)&(c==216), 97, a)",
        {"a": l4,
         "b": bare_gradation,
         "c": level3},
        name="mark_surface",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==215)&(c==215), 93, a)",
        {"a": l4,
         "c": level3},
        name="mark_surface",
        dtype="uint8"
    ) 
    l4 = expr_eval(
        "where((a==216)&(c==216), 94, a)",
        {"a": l4,
         "c": level3},
        name="mark_surface",
        dtype="uint8"
    )
    return l4