from odc.stats._algebra import expr_eval
import xarray as xr

NODATA = 255


def lifeform(xx: xr.Dataset):

    # 113 ----> 1 woody
    # 114 ----> 2 herbaceous

    lifeform_mask = expr_eval(
        "where((a!=a)|(a>=nodata), nodata, a)",
        {"a": xx.woody.data},
        name="mark_lifeform",
        dtype="float32",
        **{"nodata": NODATA},
    )

    lifeform_mask = expr_eval(
        "where(a==113, 1, a)",
        {"a": lifeform_mask},
        name="mark_lifeform",
        dtype="uint8",
    )
    lifeform_mask = expr_eval(
        "where(a==114, 2, a)",
        {"a": lifeform_mask},
        name="mark_lifeform",
        dtype="uint8",
    )

    return lifeform_mask
