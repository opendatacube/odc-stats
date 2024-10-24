"""
  Define Natural Aquatic Classes in Level-4
"""

from odc.stats._algebra import expr_eval


def natural_auquatic_veg(l4, lifeform, veg_cover, water_seasonality):

    l4 = expr_eval(
        "where((a==124)&(b==10)&(c==1)&(d==1), 64, a)",
        {
            "a": l4,
            "b": veg_cover,
            "c": lifeform,
            "d": water_seasonality,
        },
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((a==124)&(b==10)&(c==1)&(d==2), 65, a)",
        {
            "a": l4,
            "b": veg_cover,
            "c": lifeform,
            "d": water_seasonality,
        },
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==124)&(b==12)&(c==1)&(d==1), 67, a)",
        {
            "a": l4,
            "b": veg_cover,
            "c": lifeform,
            "d": water_seasonality,
        },
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==124)&(b==12)&(c==1)&(d==2), 68, a)",
        {
            "a": l4,
            "b": veg_cover,
            "c": lifeform,
            "d": water_seasonality,
        },
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==124)&(b==13)&(c==1)&(d==1), 70, a)",
        {
            "a": l4,
            "b": veg_cover,
            "c": lifeform,
            "d": water_seasonality,
        },
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==124)&(b==13)&(c==1)&(d==2), 71, a)",
        {
            "a": l4,
            "b": veg_cover,
            "c": lifeform,
            "d": water_seasonality,
        },
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==124)&(b==15)&(c==1)&(d==1), 73, a)",
        {
            "a": l4,
            "b": veg_cover,
            "c": lifeform,
            "d": water_seasonality,
        },
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==124)&(b==15)&(c==1)&(d==2), 74, a)",
        {
            "a": l4,
            "b": veg_cover,
            "c": lifeform,
            "d": water_seasonality,
        },
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==124)&(b==16)&(c==1)&(d==1), 76, a)",
        {
            "a": l4,
            "b": veg_cover,
            "c": lifeform,
            "d": water_seasonality,
        },
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==124)&(b==16)&(c==1)&(d==2), 77, a)",
        {
            "a": l4,
            "b": veg_cover,
            "c": lifeform,
            "d": water_seasonality,
        },
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==124)&(b==10)&(c==2)&(d==1), 79, a)",
        {
            "a": l4,
            "b": veg_cover,
            "c": lifeform,
            "d": water_seasonality,
        },
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==124)&(b==10)&(c==2)&(d==2), 80, a)",
        {
            "a": l4,
            "b": veg_cover,
            "c": lifeform,
            "d": water_seasonality,
        },
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==124)&(b==12)&(c==2)&(d==1), 82, a)",
        {
            "a": l4,
            "b": veg_cover,
            "c": lifeform,
            "d": water_seasonality,
        },
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==124)&(b==12)&(c==2)&(d==2), 83, a)",
        {
            "a": l4,
            "b": veg_cover,
            "c": lifeform,
            "d": water_seasonality,
        },
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==124)&(b==13)&(c==2)&(d==1), 85, a)",
        {
            "a": l4,
            "b": veg_cover,
            "c": lifeform,
            "d": water_seasonality,
        },
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==124)&(b==13)&(c==2)&(d==2), 86, a)",
        {
            "a": l4,
            "b": veg_cover,
            "c": lifeform,
            "d": water_seasonality,
        },
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==124)&(b==15)&(c==2)&(d==1), 88, a)",
        {
            "a": l4,
            "b": veg_cover,
            "c": lifeform,
            "d": water_seasonality,
        },
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==124)&(b==15)&(c==2)&(d==2), 89, a)",
        {
            "a": l4,
            "b": veg_cover,
            "c": lifeform,
            "d": water_seasonality,
        },
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==124)&(b==16)&(c==2)&(d==1), 91, a)",
        {
            "a": l4,
            "b": veg_cover,
            "c": lifeform,
            "d": water_seasonality,
        },
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==124)&(b==16)&(c==2)&(d==2), 92, a)",
        {
            "a": l4,
            "b": veg_cover,
            "c": lifeform,
            "d": water_seasonality,
        },
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==124)&(b==15)&(c==1), 72, a)",
        {
            "a": l4,
            "b": veg_cover,
            "c": lifeform,
        },
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==124)&(b==16)&(c==1), 75, a)",
        {
            "a": l4,
            "b": veg_cover,
            "c": lifeform,
        },
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((a==124)&(b==10)&(c==2), 78, a)",
        {
            "a": l4,
            "b": veg_cover,
            "c": lifeform,
        },
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((a==124)&(b==12)&(c==2), 81, a)",
        {
            "a": l4,
            "b": veg_cover,
            "c": lifeform,
        },
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((a==124)&(b==13)&(c==2), 84, a)",
        {
            "a": l4,
            "b": veg_cover,
            "c": lifeform,
        },
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==124)&(b==15)&(c==2), 87, a)",
        {
            "a": l4,
            "b": veg_cover,
            "c": lifeform,
        },
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==124)&(b==16)&(c==2), 90, a)",
        {
            "a": l4,
            "b": veg_cover,
            "c": lifeform,
        },
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==124)&(b==10)&(c==1), 63, a)",
        {
            "a": l4,
            "b": veg_cover,
            "c": lifeform,
        },
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((a==124)&(b==12)&(c==1), 66, a)",
        {
            "a": l4,
            "b": veg_cover,
            "c": lifeform,
        },
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((a==124)&(b==13)&(c==1), 69, a)",
        {
            "a": l4,
            "b": veg_cover,
            "c": lifeform,
        },
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==124)&(b==1), 56, a)",
        {"a": l4, "b": lifeform},
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==124)&(c==2), 57, a)",
        {"a": l4, "c": lifeform},
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==124)&(b==10), 58, a)",
        {"a": l4, "b": veg_cover},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((a==124)&(b==12), 59, a)",
        {"a": l4, "b": veg_cover},
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==124)&(b==13), 60, a)",
        {"a": l4, "b": veg_cover},
        name="mark_cultivated",
        dtype="uint8",
    )

    l4 = expr_eval(
        "where((a==124)&(b==15), 61, a)",
        {"a": l4, "b": veg_cover},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((a==124)&(b==16), 62, a)",
        {"a": l4, "b": veg_cover},
        name="mark_cultivated",
        dtype="uint8",
    )
    l4 = expr_eval(
        "where((a==124), 55, a)", {"a": l4}, name="mark_mangroves", dtype="uint8"
    )

    return l4
