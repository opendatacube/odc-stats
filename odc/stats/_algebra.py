# The content in this file will be moved to odc.algo after refactor.
# These are functionalities required to perform basic algebra in LandCover plugins
# Keep here for time being and an easier development/experiment

import dask.array as da
import functools
from dask.base import tokenize
from odc.algo._numexpr import apply_numexpr_np
from odc.algo._dask import flatten_kv, unflatten_kv


def expr_eval(expr, data, dtype="float32", **kwargs):
    name = kwargs.pop("name", "expr_eval")
    tk = tokenize(apply_numexpr_np, *flatten_kv(data))
    op = functools.partial(
        apply_numexpr_np, expr, dtype=None, casting="unsafe", order="K", **kwargs
    )

    return da.map_blocks(
        lambda op, *data: op(unflatten_kv(data)),
        op,
        *flatten_kv(data),
        name=f"{name}_{tk}",
        dtype=dtype,
    )
