# The content in this file will be moved to odc.algo after refactor.
# These are functionalities required to perform basic algebra in LandCover plugins
# Keep here for time being and an easier development/experiment

import dask.array as da
import numpy as np
import numexpr as ne
import functools
from dask.base import tokenize
from typing import Any, Dict, Optional
from odc.algo._dask import flatten_kv, unflatten_kv


def apply_numexpr_np(
    expr: str,
    data: Optional[Dict[str, Any]] = None,
    dtype=None,
    casting="safe",
    order="K",
    **params,
) -> np.ndarray:
    """
    Apply numexpr to numpy arrays
    """

    if data is None:
        data = params
    else:
        data.update(params)

    out = ne.evaluate(expr, local_dict=data, casting=casting, order=order)
    if dtype is None:
        return out
    else:
        return out.astype(dtype)


def expr_eval(expr, data, dtype="float32", name="expr_eval", **kwargs):
    tk = tokenize(apply_numexpr_np, *flatten_kv(data))
    op = functools.partial(
        apply_numexpr_np, expr, dtype=dtype, casting="unsafe", order="K", **kwargs
    )

    return da.map_blocks(
        lambda op, *data: op(unflatten_kv(data)),
        op,
        *flatten_kv(data),
        name=f"{name}_{tk}",
        dtype=dtype,
        meta=np.array((), dtype=dtype),
    )
