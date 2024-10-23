import numpy as np
import xarray as xr
import dask.array as da
from odc.stats.plugins.lc_level34 import StatsLccsLevel4
from odc.stats.plugins.l34_utils import l4_cultivated, lc_level3, l4_veg_cover

import pytest
import pandas as pd

NODATA = 255

def image_groups(l34, urban, cultivated, woody, pv_pc_50):
   
    tuples = [
        (np.datetime64("2000-01-01T00"), np.datetime64("2000-01-01")),
    ]
    index = pd.MultiIndex.from_tuples(tuples, names=["time", "solar_day"])
    coords = {
        "x": np.linspace(10, 20, l34.shape[2]),
        "y": np.linspace(0, 5, l34.shape[1]),
        "spec": index,
    }

    data_vars = {
        "classes_l3_l4": xr.DataArray(
            l34, dims=("spec", "y", "x"), attrs={"nodata": 255}
        ),
        "urban_classes": xr.DataArray(
            urban, dims=("spec", "y", "x"), attrs={"nodata": 255}
        ),
        "cultivated_class": xr.DataArray(
            cultivated, dims=("spec", "y", "x"), attrs={"nodata": 255}
        ),
        "woody_cover": xr.DataArray(
            woody, dims=("spec", "y", "x"), attrs={"nodata": 255}
        ),
        "pv_pc_50": xr.DataArray(
            pv_pc_50, dims=("spec", "y", "x"), attrs={"nodata": 255}
        ),
    }
    xx = xr.Dataset(data_vars=data_vars, coords=coords)
    return xx


def test_ctv_classes_woody():

    expected_cultivated_classes = [
        [13, 10, 9],
        [110, 10, 10],
        [13, 11, 11],
        [12, 13, 10],
    ]

    l34 = np.array(
        [
            [
                [110, 110, 110],
                [110, 110, 110],
                [110, 110, 110],
                [110, 110, 110],
            ]
        ],
        dtype="int",
    )

    urban = np.array(
        [
            [
                [216, 216, 216],
                [216, 216, 216],
                [216, 216, 216],
                [216, 216, 216],
            ]
        ],
        dtype="int",
    )
    # 111 --> cultivated
    cultivated = np.array(
        [
            [
                [111, 111, 111],
                [255, 111, 111],
                [111, 111, 111],
                [111, 111, 111],
            ]
        ],
        dtype="int",
    )

    woody = np.array(
        [
            [
                [113, 113, 113],
                [113, 113, 113],
                [113, 113, 113],
                [113, 113, 113],
            ]
        ],
        dtype="int",
    )
    
    pv_pc_50 = np.array(
        [
            [
                [1, 64, 65],
                [66, 40, 41],
                [3, 16, 15],
                [4, 1, 42],
            ]
        ],
        dtype="int",
    )
    xx = image_groups(l34, urban, cultivated, woody, pv_pc_50)
    
    stats_l4 = StatsLccsLevel4()
    intertidal_mask, level3 = lc_level3.lc_level3(xx)
    lifeform = stats_l4.define_life_form(xx)
    veg_cover = l4_veg_cover.canopyco_veg_con(xx, stats_l4.veg_threshold)
    veg_cover = stats_l4.apply_mapping(veg_cover, stats_l4.veg_mapping)
    
    l4_ctv = l4_cultivated.lc_l4_cultivated(xx.classes_l3_l4, level3, lifeform, veg_cover)
    assert (l4_ctv.compute() == expected_cultivated_classes).all()

def test_ctv_classes_herbaceous():

    expected_cultivated_classes = [
        [18, 15, 14],
        [110, 15, 15],
        [18, 16, 16],
        [17, 18, 15],
    ]

    l34 = np.array(
        [
            [
                [110, 110, 110],
                [110, 110, 110],
                [110, 110, 110],
                [110, 110, 110],
            ]
        ],
        dtype="int",
    )

    urban = np.array(
        [
            [
                [216, 216, 216],
                [216, 216, 216],
                [216, 216, 216],
                [216, 216, 216],
            ]
        ],
        dtype="int",
    )

    cultivated = np.array(
        [
            [
                [111, 111, 111],
                [255, 111, 111],
                [111, 111, 111],
                [111, 111, 111],
            ]
        ],
        dtype="int",
    )

    woody = np.array(
        [
            [
                [114, 114, 114],
                [114, 114, 114],
                [114, 114, 114],
                [114, 114, 114],
            ]
        ],
        dtype="int",
    )
    
    pv_pc_50 = np.array(
        [
            [
                [1, 64, 65],
                [66, 40, 41],
                [3, 16, 15],
                [4, 1, 42],
            ]
        ],
        dtype="int",
    )
    xx = image_groups(l34, urban, cultivated, woody, pv_pc_50)
    
    stats_l4 = StatsLccsLevel4()
    intertidal_mask, level3 = lc_level3.lc_level3(xx)
    lifeform = stats_l4.define_life_form(xx)
    veg_cover = l4_veg_cover.canopyco_veg_con(xx, stats_l4.veg_threshold)
    veg_cover = stats_l4.apply_mapping(veg_cover, stats_l4.veg_mapping)
    
    l4_ctv = l4_cultivated.lc_l4_cultivated(xx.classes_l3_l4, level3, lifeform, veg_cover)
    assert (l4_ctv.compute() == expected_cultivated_classes).all()


def test_ctv_classes_woody_herbaceous():

    expected_cultivated_classes = [
        [13, 10, 9],
        [110, 15, 15],
        [13, 11, 11],
        [17, 18, 15],
    ]
    
    l34 = np.array(
        [
            [
                [110, 110, 110],
                [110, 110, 110],
                [110, 110, 110],
                [110, 110, 110],
            ]
        ],
        dtype="int",
    )

    urban = np.array(
        [
            [
                [216, 216, 216],
                [216, 216, 216],
                [216, 216, 216],
                [216, 216, 216],
            ]
        ],
        dtype="int",
    )

    cultivated = np.array(
        [
            [
                [111, 111, 111],
                [255, 111, 111],
                [111, 111, 111],
                [111, 111, 111],
            ]
        ],
        dtype="int",
    )

    woody = np.array(
        [
            [
                [113, 113, 113],
                [114, 114, 114],
                [113, 113, 113],
                [114, 114, 114],
            ]
        ],
        dtype="int",
    )
    
    pv_pc_50 = np.array(
        [
            [
                [1, 64, 65],
                [66, 40, 41],
                [3, 16, 15],
                [4, 1, 42],
            ]
        ],
        dtype="int",
    )
    xx = image_groups(l34, urban, cultivated, woody, pv_pc_50)
    
    stats_l4 = StatsLccsLevel4()
    intertidal_mask, level3 = lc_level3.lc_level3(xx)
    lifeform = stats_l4.define_life_form(xx)
    veg_cover = l4_veg_cover.canopyco_veg_con(xx, stats_l4.veg_threshold)
    veg_cover = stats_l4.apply_mapping(veg_cover, stats_l4.veg_mapping)
    
    l4_ctv = l4_cultivated.lc_l4_cultivated(xx.classes_l3_l4, level3, lifeform, veg_cover)
    assert (l4_ctv.compute() == expected_cultivated_classes).all()


def test_ctv_classes_no_vegcover():

    expected_cultivated_classes = [
        [2, 2, 2],
        [110, 3, 3],
        [2, 2, 2],
        [3, 3, 3],
    ]
    
    l34 = np.array(
        [
            [
                [110, 110, 110],
                [110, 110, 110],
                [110, 110, 110],
                [110, 110, 110],
            ]
        ],
        dtype="int",
    )

    urban = np.array(
        [
            [
                [216, 216, 216],
                [216, 216, 216],
                [216, 216, 216],
                [216, 216, 216],
            ]
        ],
        dtype="int",
    )

    cultivated = np.array(
        [
            [
                [111, 111, 111],
                [255, 111, 111],
                [111, 111, 111],
                [111, 111, 111],
            ]
        ],
        dtype="int",
    )

    woody = np.array(
        [
            [
                [113, 113, 113],
                [114, 114, 114],
                [113, 113, 113],
                [114, 114, 114],
            ]
        ],
        dtype="int",
    )
    
    pv_pc_50 = np.array(
        [
            [
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
            ]
        ],
        dtype="int",
    )
    xx = image_groups(l34, urban, cultivated, woody, pv_pc_50)
    
    stats_l4 = StatsLccsLevel4()
    intertidal_mask, level3 = lc_level3.lc_level3(xx)
    lifeform = stats_l4.define_life_form(xx)
    veg_cover = l4_veg_cover.canopyco_veg_con(xx, stats_l4.veg_threshold)
    veg_cover = stats_l4.apply_mapping(veg_cover, stats_l4.veg_mapping)
    
    l4_ctv = l4_cultivated.lc_l4_cultivated(xx.classes_l3_l4, level3, lifeform, veg_cover)
    assert (l4_ctv.compute() == expected_cultivated_classes).all()