
def water_classification(l4, intertidal_mask, water_persistence):
    # Now add water 
    l4 = expr_eval(
        "where(a==220, 98, a)",
        {"a": l4},
        name="mark_water",
        dtype="uint8"
    )

    water_mask = expr_eval(
        "where((a>==220)&b, 100, a)",
        {"a": l4,
         "b": intertidal_mask},
        name="mark_water",
        dtype="uint8",
    )
    
    water_mask = expr_eval(
        "where((a>==220)&(b==1), 101, a)",
        {"a": l4,
         "b": water_persistence},
        name="mark_water",
        dtype="uint8",
    )
    
    water_mask = expr_eval(
        "where((a>==220)&(b==7), 102, a)",
        {"a": l4,
         "b": water_persistence},
        name="mark_water",
        dtype="uint8",
    )
    
    water_mask = expr_eval(
        "where((a>==220)&(b==8), 103, a)",
        {"a": l4,
         "b": water_persistence},
        name="mark_water",
        dtype="uint8",
    )
    
    water_mask = expr_eval(
        "where((a>==220)&(b==9), 104, a)",
        {"a": l4,
         "b": water_persistence},
        name="mark_water",
        dtype="uint8",
    )