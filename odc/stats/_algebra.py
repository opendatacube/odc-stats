# The content in this file will be moved to odc.algo after refactor.
# These are functionalities required to perform basic algebra in LandCover plugins
# Keep here for time being and an easier development/experiment

import numexpr as ne
import dask.array as da
from dask.base import tokenize


def _expr_eval(expr, params, i_v, m_v_1=None, m_v_2=None, **kwargs):
    local_dict = {}
    i = 0
    local_dict[params[i]] = i_v
    if m_v_1 is not None:
        i += 1
        local_dict[params[i]] = m_v_1
    if m_v_2 is not None:
        i += 1
        local_dict[params[i]] = m_v_2
    local_dict.update(kwargs)
    return ne.evaluate(expr, local_dict=local_dict, casting="unsafe")


def expr_eval(expr, params, i_v, m_v_1=None, m_v_2=None, dtype="float32", **kwargs):
    name = kwargs.pop("name", "expr_eval")
    tk = tokenize(_expr_eval, expr, i_v, m_v_1, m_v_2)
    return da.map_blocks(
        _expr_eval,
        expr,
        params,
        i_v,
        m_v_1,
        m_v_2,
        name=f"{name}_{tk}",
        dtype=dtype,
        **kwargs,
    )
