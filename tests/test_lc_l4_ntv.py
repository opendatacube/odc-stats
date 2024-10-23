"""
 Unit tests for LandCover Natural Terrestrial Vegetated classes
"""

import numpy as np
import xarray as xr
import dask.array as da
from odc.stats.plugins.lc_level34 import StatsLccsLevel4
from odc.stats.plugins.l34_utils import l4_cultivated, lc_level3, l4_veg_cover, l4_natural_veg

import pytest
import pandas as pd

NODATA = 255
FC_NODATA = -9999

# @pytest.fixture(scope="module")
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


def test_ntv_classes_herbaceous():
    
    expected_natural_terrestrial_veg_classes = [
        [36, 33, 32],
        [110, 33, 33],
        [36, 34, 34],
        [35, 36, 33]
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
    # 112 --> natural veg
    cultivated = np.array(
        [
            [
                [112, 112, 112],
                [255, 112, 112],
                [112, 112, 112],
                [112, 112, 112],
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
    intertidal_mask, level3 = lc_level3.lc_level3(xx, NODATA)
    lifeform = stats_l4.define_life_form(xx)
    veg_cover = l4_veg_cover.canopyco_veg_con(xx, stats_l4.veg_threshold, NODATA, FC_NODATA)
    veg_cover = stats_l4.apply_mapping(veg_cover, stats_l4.veg_mapping)

    l4_ntv = l4_natural_veg.lc_l4_natural_veg(xx.classes_l3_l4, level3, lifeform, veg_cover)
    assert (l4_ntv.compute() == expected_natural_terrestrial_veg_classes).all()


def test_ntv_classes_woody():
    
    expected_natural_terrestrial_veg_classes = [
        [31, 28, 27], 
        [110, 28, 28],
        [31, 29, 29],
        [30, 26, 28]
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
    # 112 --> natural veg
    cultivated = np.array(
        [
            [
                [112, 112, 112],
                [255, 112, 112],
                [112, 112, 112],
                [112, 112, 112],
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
                [113, 255, 113],
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
    intertidal_mask, level3 = lc_level3.lc_level3(xx, NODATA)
    lifeform = stats_l4.define_life_form(xx)
    veg_cover = l4_veg_cover.canopyco_veg_con(xx, stats_l4.veg_threshold, NODATA, FC_NODATA)
    veg_cover = stats_l4.apply_mapping(veg_cover, stats_l4.veg_mapping)

    l4_ntv = l4_natural_veg.lc_l4_natural_veg(xx.classes_l3_l4, level3, lifeform, veg_cover)
    assert (l4_ntv.compute() == expected_natural_terrestrial_veg_classes).all()

def test_ntv_classes_no_veg():
    
    expected_natural_terrestrial_veg_classes = [
        [20, 20, 20],
        [110, 21, 21],
        [20, 20, 20],
        [21, 21, 21]
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
    # 112 --> natural veg
    cultivated = np.array(
        [
            [
                [112, 112, 112],
                [255, 112, 112],
                [112, 112, 112],
                [112, 112, 112],
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
    intertidal_mask, level3 = lc_level3.lc_level3(xx, NODATA)
    lifeform = stats_l4.define_life_form(xx)
    veg_cover = l4_veg_cover.canopyco_veg_con(xx, stats_l4.veg_threshold, NODATA, FC_NODATA)
    veg_cover = stats_l4.apply_mapping(veg_cover, stats_l4.veg_mapping)

    l4_ntv = l4_natural_veg.lc_l4_natural_veg(xx.classes_l3_l4, level3, lifeform, veg_cover)
    assert (l4_ntv.compute() == expected_natural_terrestrial_veg_classes).all()

def test_ntv_classes_no_lifeform():
    
    expected_natural_terrestrial_veg_classes = [
        [26, 23, 22],
        [22, 23, 23],
        [26, 24, 24],
        [25, 26, 23]
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
    # 112 --> natural veg
    cultivated = np.array(
        [
            [
                [112, 112, 112],
                [112, 112, 112],
                [112, 112, 112],
                [112, 112, 112],
            ]
        ],
        dtype="int",
    )

    woody = np.array(
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
    intertidal_mask, level3 = lc_level3.lc_level3(xx, NODATA)
    lifeform = stats_l4.define_life_form(xx)
    veg_cover = l4_veg_cover.canopyco_veg_con(xx, stats_l4.veg_threshold, NODATA, FC_NODATA)
    veg_cover = stats_l4.apply_mapping(veg_cover, stats_l4.veg_mapping)

    l4_ntv = l4_natural_veg.lc_l4_natural_veg(xx.classes_l3_l4, level3, lifeform, veg_cover)
    assert (l4_ntv.compute() == expected_natural_terrestrial_veg_classes).all()