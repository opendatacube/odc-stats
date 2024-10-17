from odc.stats._algebra import expr_eval

def lc_l4_surface(l4, bare_gradation):
            
    l4 = expr_eval(
        "where((a==215), 93, a)",
        {"a": l4},
        name="mark_surface",
        dtype="uint8"
    ) 
    l4 = expr_eval(
        "where((a==216), 94, a)",
        {"a": l4},
        name="mark_surface",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==216)&(b==10), 95, a)",
        {"a": l4,
         "b": bare_gradation},
        name="mark_surface",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==216)&(b==12), 96, a)",
        {"a": l4,
         "b": bare_gradation},
        name="mark_surface",
        dtype="uint8"
    )
    l4 = expr_eval(
        "where((a==216)&(b==10), 97, a)",
        {"a": l4,
         "b": bare_gradation},
        name="mark_surface",
        dtype="uint8"
    )
    
    return l4