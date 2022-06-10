import numpy as np
import xarray as xr
import dask.array as da
from odc.stats.plugins.gm import StatsGMLS
from .test_utils import usgs_ls8_sr_definition
import pytest
import pandas as pd

@pytest.fixture
def dataset():
    cloud_mask = 3
    no_data = 0
    band_red = np.array([[[255, 57], [20, 0]],
                         [[30, 10], [70, 80]],
                         [[25, 1], [120, 0]],])
    band_fmask = np.array([
        [[0, 0], [0, no_data]],
        [[3, no_data], [3, 3]],
        [[0, 0], [no_data, 0]],
    ])

    band_red = da.from_array(band_red, chunks=(3, -1, -1))
    times = [np.datetime64(f"2000-01-01T0{i}") for i in range(3)]

    coords = {
        "x": np.linspace(10, 20, band_red.shape[2]),
        "y": np.linspace(0, 5, band_red.shape[1]),
        "time": times,
    }

    data_vars = {"band_red": (("time", "y", "x"), band_red),
                'fmask': (("time", "y", "x"), band_fmask)}

    xx = xr.Dataset(data_vars=data_vars, coords=coords)
    xx['fmask'] = xx.fmask.assign_attrs(units=1, nodata=0,
                                        flags_definition={'fmask': {'bits': [0, 1, 2, 3, 4, 5, 6, 7], 'values': {'0': 'nodata', '1': 'valid', '2': 'cloud', '3': 'shadow', '4': 'snow', '5': 'water'}, 'description': 'Fmask'}},
                                        crs='EPSG:3577',grid_mapping='spatial_ref')

    return xx

def test_native_transform(dataset):

    _ = pytest.importorskip("hdstats")
    mask_filters = {'cloud': [("closing", 0), ("dilation",0)],
                    'shadow': [("closing", 0), ("dilation",0)]
    }

    dataset = dataset.copy()
    stats_gmls = StatsGMLS(cloud_filters=mask_filters,
                        nodata_classes=(-999, ))
    xx = stats_gmls.native_transform(dataset)
    result = xx.compute()

    expected_result = np.array([
                            [[255, 57], [20, -999]],
                            [[30, -999], [70, 80]],
                            [[25, 1], [-999, 0]]
                                ])
    assert (result == expected_result).all()

def test_result_bands_to_match_inputs(dataset):
    _ = pytest.importorskip("hdstats")
    mask_filters = {'cloud': [("closing", 2), ("dilation",1)],
                    'shadow': [("closing", 0), ("dilation",5)]
    }

    dataset = dataset.copy()
    stats_gmls = StatsGMLS(cloud_filters=mask_filters,
                        nodata_classes=(-999, ))
    xx = stats_gmls.native_transform(dataset)
    result = stats_gmls.reduce(xx)

    assert set(result.data_vars.keys()) == set(
            ["band_red", "sdev", "edev", "bcdev", "count"]
    )

def test_result_aux_bands_to_match_inputs(dataset):
    _ = pytest.importorskip("hdstats")
    mask_filters = {'cloud': [("closing", 2), ("dilation",1)],
                    'shadow': [("closing", 0), ("dilation",5)]
    }

    dataset = dataset.copy()
    aux_names=dict(smad="SDEV", emad="EDEV", bcmad="BCDEV", count="COUNT")
    stats_gmls = StatsGMLS(cloud_filters=mask_filters,
                        nodata_classes=(-999, ),
                        aux_names=aux_names)

    xx = stats_gmls.native_transform(dataset)
    result = stats_gmls.reduce(xx)


    assert set(result.data_vars.keys()) == set(
            ["band_red", "SDEV", "EDEV", "BCDEV", "COUNT"]
    )
