# The content in this file will be moved to odc.algo after refactor.
# These are functionalities required to perform basic algebra in LandCover plugins
# Keep here for time being and an easier development/experiment

import dask.array as da
import xarray as xr
import numpy as np
from dask.base import tokenize


def _median_by_ind(a):
    d = np.sort(a, axis=0).reshape(a.shape[0], -1)
    valid_count = d.shape[0] - np.isnan(d).sum(axis=0)
    indices = [valid_count // 2 - (1 - valid_count % 2), valid_count // 2]
    d = d.flatten(order="F")
    e = (
        d[indices[0] + np.arange(0, d.shape[0], a.shape[0])]
        + d[indices[1] + np.arange(0, d.shape[0], a.shape[0])]
    ) / 2
    return e.reshape(a.shape[1:])


def median_by_ind(xr_da, dim, dtype="float32", name="median_by_ind"):
    if xr_da.dims[0] != dim:
        raise ValueError(f"{dim} has to be on dimension 0")
    tk = tokenize(_median_by_ind, xr_da.data)
    res = da.map_blocks(
        _median_by_ind,
        xr_da.data,
        name=f"{name}_{tk}",
        dtype=dtype,
        meta=np.array((), dtype=dtype),
        drop_axis=0,
    )
    coords = {dim: xr_da.coords[dim] for dim in xr_da.dims[1:]}

    return xr.DataArray(
        res, dims=xr_da.dims[1:], coords=coords, attrs=xr_da.attrs.copy()
    )


def median_ds(xr_ds, dim, dtype="float32", name="median_ds"):
    res = {}
    for var, data in xr_ds.data_vars.items():
        res[var] = median_by_ind(data, dim, dtype, name)
    # pylint: disable=undefined-loop-variable
    coords = {dim: xr_ds.coords[dim] for dim in data.dims[1:]}
    return xr.Dataset(res, coords=coords, attrs=xr_ds.attrs.copy())
