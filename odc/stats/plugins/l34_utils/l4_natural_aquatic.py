"""
  Define Natural Aquatic Classes in Level-4
"""

from odc.stats._algebra import expr_eval


def natural_auquatic_veg(l4, veg_cover, water_seasonality):

    # mark woody/herbaceous
    # mangroves -> woody
    # everything else -> herbaceous
    res = expr_eval(
        "where((a==124), 56, a)",
        {
            "a": l4,
        },
        name="mark_woody",
        dtype="uint8",
    )

    res = expr_eval(
        "where((a==125), 57, a)",
        {
            "a": res,
        },
        name="mark_herbaceous",
        dtype="uint8",
    )

    # res = expr_eval(
    #     "where((a!=124)|(a!=125), 255, a)",
    #     {
    #         "a": res,
    #     },
    #     name="mark_nodata",
    #     dtype="uint8",
    # )

    # mark water season
    # use some value not used in final class
    res = expr_eval(
        "where((a==56)&(b==1), 254, a)",
        {
            "a": res,
            "b": water_seasonality,
        },
        name="mark_water_season",
        dtype="uint8",
    )

    res = expr_eval(
        "where((a==56)&(b==2), 253, a)",
        {
            "a": res,
            "b": water_seasonality,
        },
        name="mark_water_season",
        dtype="uint8",
    )

    res = expr_eval(
        "where((a==57)&(b==1), 252, a)",
        {
            "a": res,
            "b": water_seasonality,
        },
        name="mark_water_season",
        dtype="uint8",
    )

    res = expr_eval(
        "where((a==57)&(b==2), 251, a)",
        {
            "a": res,
            "b": water_seasonality,
        },
        name="mark_water_season",
        dtype="uint8",
    )

    # mark final

    res = expr_eval(
        "where((a==254)&(b==10), 64, a)",
        {
            "a": res,
            "b": veg_cover,
        },
        name="mark_final",
        dtype="uint8",
    )

    res = expr_eval(
        "where((a==253)&(b==10), 65, a)",
        {
            "a": res,
            "b": veg_cover,
        },
        name="mark_final",
        dtype="uint8",
    )

    res = expr_eval(
        "where((a==252)&(b==10), 80, a)",
        {
            "a": res,
            "b": veg_cover,
        },
        name="mark_final",
        dtype="uint8",
    )

    res = expr_eval(
        "where((a==251)&(b==10), 81, a)",
        {
            "a": res,
            "b": veg_cover,
        },
        name="mark_final",
        dtype="uint8",
    )
    #########################################
    res = expr_eval(
        "where((a==254)&(b==12), 67, a)",
        {
            "a": res,
            "b": veg_cover,
        },
        name="mark_final",
        dtype="uint8",
    )

    res = expr_eval(
        "where((a==253)&(b==12), 68, a)",
        {
            "a": res,
            "b": veg_cover,
        },
        name="mark_final",
        dtype="uint8",
    )

    res = expr_eval(
        "where((a==252)&(b==12), 82, a)",
        {
            "a": res,
            "b": veg_cover,
        },
        name="mark_final",
        dtype="uint8",
    )

    res = expr_eval(
        "where((a==251)&(b==12), 83, a)",
        {
            "a": res,
            "b": veg_cover,
        },
        name="mark_final",
        dtype="uint8",
    )
    ##########################################
    res = expr_eval(
        "where((a==254)&(b==13), 70, a)",
        {
            "a": res,
            "b": veg_cover,
        },
        name="mark_final",
        dtype="uint8",
    )

    res = expr_eval(
        "where((a==253)&(b==13), 71, a)",
        {
            "a": res,
            "b": veg_cover,
        },
        name="mark_final",
        dtype="uint8",
    )

    res = expr_eval(
        "where((a==252)&(b==13), 85, a)",
        {
            "a": res,
            "b": veg_cover,
        },
        name="mark_final",
        dtype="uint8",
    )

    res = expr_eval(
        "where((a==251)&(b==13), 86, a)",
        {
            "a": res,
            "b": veg_cover,
        },
        name="mark_final",
        dtype="uint8",
    )
    #########################################

    res = expr_eval(
        "where((a==254)&(b==15), 73, a)",
        {
            "a": res,
            "b": veg_cover,
        },
        name="mark_final",
        dtype="uint8",
    )

    res = expr_eval(
        "where((a==253)&(b==15), 74, a)",
        {
            "a": res,
            "b": veg_cover,
        },
        name="mark_final",
        dtype="uint8",
    )

    res = expr_eval(
        "where((a==252)&(b==15), 88, a)",
        {
            "a": res,
            "b": veg_cover,
        },
        name="mark_final",
        dtype="uint8",
    )

    res = expr_eval(
        "where((a==251)&(b==15), 89, a)",
        {
            "a": res,
            "b": veg_cover,
        },
        name="mark_final",
        dtype="uint8",
    )
    ##########################################
    res = expr_eval(
        "where((a==254)&(b==16), 76, a)",
        {
            "a": res,
            "b": veg_cover,
        },
        name="mark_final",
        dtype="uint8",
    )

    res = expr_eval(
        "where((a==253)&(b==16), 77, a)",
        {
            "a": res,
            "b": veg_cover,
        },
        name="mark_final",
        dtype="uint8",
    )

    res = expr_eval(
        "where((a==252)&(b==16), 91, a)",
        {
            "a": res,
            "b": veg_cover,
        },
        name="mark_final",
        dtype="uint8",
    )

    res = expr_eval(
        "where((a==251)&(b==16), 92, a)",
        {
            "a": res,
            "b": veg_cover,
        },
        name="mark_final",
        dtype="uint8",
    )

    return res
