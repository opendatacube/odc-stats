"""
 Unit tests for LandCover Natural Aquatic Vegetation classes
"""

import numpy as np
import xarray as xr
import dask.array as da
from odc.stats.plugins.lc_level34 import StatsLccsLevel4
from odc.stats.plugins.l34_utils import (
    l4_cultivated,
    lc_level3,
    l4_veg_cover,
    l4_natural_veg,
    l4_water_persistence,
    l4_natural_aquatic,
)

import pytest
import pandas as pd

NODATA = 255


def image_groups(l34, urban, cultivated, woody, pv_pc_50, water_frequency):

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
        "water_frequency": xr.DataArray(
            water_frequency, dims=("spec", "y", "x"), attrs={"nodata": 255}
        ),
    }
    xx = xr.Dataset(data_vars=data_vars, coords=coords)
    return xx


def test_ntv_classes_woody_herbaceous():
    expected_l4_ntv_classes = [[56, 56, 56], [56, 56, 55], [57, 57, 57], [57, 57, 55]]

    l34 = np.array(
        [
            [
                [124, 124, 124],
                [124, 124, 124],
                [124, 124, 124],
                [124, 124, 124],
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
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
            ]
        ],
        dtype="int",
    )

    woody = np.array(
        [
            [
                [113, 113, 113],
                [113, 113, 255],
                [114, 114, 114],
                [114, 114, 255],
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

    water_frequency = np.array(
        [
            [
                [1, 3, 2],
                [4, 5, 6],
                [9, 2, 11],
                [10, 11, 12],
            ]
        ],
        dtype="int",
    )

    xx = image_groups(l34, urban, cultivated, woody, pv_pc_50, water_frequency)

    stats_l4 = StatsLccsLevel4()
    intertidal_mask, level3 = lc_level3.lc_level3(xx)
    lifeform = stats_l4.define_life_form(xx)
    veg_cover = l4_veg_cover.canopyco_veg_con(xx, stats_l4.veg_threshold)
    veg_cover = stats_l4.apply_mapping(veg_cover, stats_l4.veg_mapping)

    # Apply cultivated to match the code in Level4 processing
    l4_ctv = l4_cultivated.lc_l4_cultivated(
        xx.classes_l3_l4, level3, lifeform, veg_cover
    )
    l4_ctv_ntv = l4_natural_veg.lc_l4_natural_veg(l4_ctv, level3, lifeform, veg_cover)

    water_seasonality = stats_l4.define_water_seasonality(xx)
    l4_ctv_ntv_nav = l4_natural_aquatic.natural_auquatic_veg(
        l4_ctv_ntv, lifeform, veg_cover, water_seasonality
    )

    assert (l4_ctv_ntv_nav.compute() == expected_l4_ntv_classes).all()


def test_ntv_veg_cover():
    expected_l4_ntv_classes = [
        [62, 59, 58],
        [58, 59, 59],
        [62, 60, 60],
        [61, 62, 59],
    ]

    l34 = np.array(
        [
            [
                [124, 124, 124],
                [124, 124, 124],
                [124, 124, 124],
                [124, 124, 124],
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
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
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
    water_frequency = np.array(
        [
            [
                [-9999, -9999, -9999],
                [-9999, -9999, -9999],
                [-9999, -9999, -9999],
                [-9999, -9999, -9999],
            ]
        ],
        dtype="int",
    )

    xx = image_groups(l34, urban, cultivated, woody, pv_pc_50, water_frequency)

    stats_l4 = StatsLccsLevel4()
    intertidal_mask, level3 = lc_level3.lc_level3(xx)
    lifeform = stats_l4.define_life_form(xx)
    veg_cover = l4_veg_cover.canopyco_veg_con(xx, stats_l4.veg_threshold)
    veg_cover = stats_l4.apply_mapping(veg_cover, stats_l4.veg_mapping)

    # Apply cultivated to match the code in Level4 processing
    l4_ctv = l4_cultivated.lc_l4_cultivated(
        xx.classes_l3_l4, level3, lifeform, veg_cover
    )
    l4_ctv_ntv = l4_natural_veg.lc_l4_natural_veg(l4_ctv, level3, lifeform, veg_cover)

    water_seasonality = stats_l4.define_water_seasonality(xx)
    l4_ctv_ntv_nav = l4_natural_aquatic.natural_auquatic_veg(
        l4_ctv_ntv, lifeform, veg_cover, water_seasonality
    )
    assert (l4_ctv_ntv_nav.compute() == expected_l4_ntv_classes).all()


def test_ntv_woody_veg_cover():
    expected_l4_ntv_classes = [
        [75, 66, 63],
        [63, 66, 66],
        [75, 69, 69],
        [72, 75, 66],
    ]

    l34 = np.array(
        [
            [
                [124, 124, 124],
                [124, 124, 124],
                [124, 124, 124],
                [124, 124, 124],
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
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
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
    water_frequency = np.array(
        [
            [
                [-9999, -9999, -9999],
                [-9999, -9999, -9999],
                [-9999, -9999, -9999],
                [-9999, -9999, -9999],
            ]
        ],
        dtype="int",
    )

    xx = image_groups(l34, urban, cultivated, woody, pv_pc_50, water_frequency)

    stats_l4 = StatsLccsLevel4()
    intertidal_mask, level3 = lc_level3.lc_level3(xx)
    lifeform = stats_l4.define_life_form(xx)
    veg_cover = l4_veg_cover.canopyco_veg_con(xx, stats_l4.veg_threshold)
    veg_cover = stats_l4.apply_mapping(veg_cover, stats_l4.veg_mapping)

    # Apply cultivated to match the code in Level4 processing
    l4_ctv = l4_cultivated.lc_l4_cultivated(
        xx.classes_l3_l4, level3, lifeform, veg_cover
    )
    l4_ctv_ntv = l4_natural_veg.lc_l4_natural_veg(l4_ctv, level3, lifeform, veg_cover)

    water_seasonality = stats_l4.define_water_seasonality(xx)
    l4_ctv_ntv_nav = l4_natural_aquatic.natural_auquatic_veg(
        l4_ctv_ntv, lifeform, veg_cover, water_seasonality
    )
    assert (l4_ctv_ntv_nav.compute() == expected_l4_ntv_classes).all()


def test_ntv_woody_seasonal_water_veg_cover():
    expected_l4_ntv_classes = [
        [77, 68, 65],
        [65, 68, 68],
        [77, 71, 71],
        [74, 77, 68],
    ]

    l34 = np.array(
        [
            [
                [124, 124, 124],
                [124, 124, 124],
                [124, 124, 124],
                [124, 124, 124],
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
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
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
    water_frequency = np.array(
        [
            [
                [1, 2, 3],
                [1, 2, 3],
                [1, 2, 3],
                [1, 2, 3],
            ]
        ],
        dtype="int",
    )

    xx = image_groups(l34, urban, cultivated, woody, pv_pc_50, water_frequency)

    stats_l4 = StatsLccsLevel4()
    intertidal_mask, level3 = lc_level3.lc_level3(xx)
    lifeform = stats_l4.define_life_form(xx)
    veg_cover = l4_veg_cover.canopyco_veg_con(xx, stats_l4.veg_threshold)
    veg_cover = stats_l4.apply_mapping(veg_cover, stats_l4.veg_mapping)

    # Apply cultivated to match the code in Level4 processing
    l4_ctv = l4_cultivated.lc_l4_cultivated(
        xx.classes_l3_l4, level3, lifeform, veg_cover
    )
    l4_ctv_ntv = l4_natural_veg.lc_l4_natural_veg(l4_ctv, level3, lifeform, veg_cover)

    water_seasonality = stats_l4.define_water_seasonality(xx)
    l4_ctv_ntv_nav = l4_natural_aquatic.natural_auquatic_veg(
        l4_ctv_ntv, lifeform, veg_cover, water_seasonality
    )

    assert (l4_ctv_ntv_nav.compute() == expected_l4_ntv_classes).all()


def test_ntv_woody_permanent_water_veg_cover():
    expected_l4_ntv_classes = [
        [76, 67, 64],
        [64, 67, 67],
        [76, 70, 70],
        [73, 76, 67],
    ]

    l34 = np.array(
        [
            [
                [124, 124, 124],
                [124, 124, 124],
                [124, 124, 124],
                [124, 124, 124],
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
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
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
    water_frequency = np.array(
        [
            [
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
                [11, 10, 8],
            ]
        ],
        dtype="int",
    )

    xx = image_groups(l34, urban, cultivated, woody, pv_pc_50, water_frequency)

    stats_l4 = StatsLccsLevel4()
    intertidal_mask, level3 = lc_level3.lc_level3(xx)
    lifeform = stats_l4.define_life_form(xx)
    veg_cover = l4_veg_cover.canopyco_veg_con(xx, stats_l4.veg_threshold)
    veg_cover = stats_l4.apply_mapping(veg_cover, stats_l4.veg_mapping)

    # Apply cultivated to match the code in Level4 processing
    l4_ctv = l4_cultivated.lc_l4_cultivated(
        xx.classes_l3_l4, level3, lifeform, veg_cover
    )
    l4_ctv_ntv = l4_natural_veg.lc_l4_natural_veg(l4_ctv, level3, lifeform, veg_cover)

    water_seasonality = stats_l4.define_water_seasonality(xx)
    l4_ctv_ntv_nav = l4_natural_aquatic.natural_auquatic_veg(
        l4_ctv_ntv, lifeform, veg_cover, water_seasonality
    )
    assert (l4_ctv_ntv_nav.compute() == expected_l4_ntv_classes).all()


def test_ntv_herbaceous_veg_cover():
    expected_l4_ntv_classes = [[90, 81, 78], [78, 81, 81], [90, 84, 84], [87, 90, 81]]

    l34 = np.array(
        [
            [
                [124, 124, 124],
                [124, 124, 124],
                [124, 124, 124],
                [124, 124, 124],
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
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
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
    water_frequency = np.array(
        [
            [
                [-9999, -9999, -9999],
                [-9999, -9999, -9999],
                [-9999, -9999, -9999],
                [-9999, -9999, -9999],
            ]
        ],
        dtype="int",
    )

    xx = image_groups(l34, urban, cultivated, woody, pv_pc_50, water_frequency)

    stats_l4 = StatsLccsLevel4()
    intertidal_mask, level3 = lc_level3.lc_level3(xx)
    lifeform = stats_l4.define_life_form(xx)
    veg_cover = l4_veg_cover.canopyco_veg_con(xx, stats_l4.veg_threshold)
    veg_cover = stats_l4.apply_mapping(veg_cover, stats_l4.veg_mapping)

    # Apply cultivated to match the code in Level4 processing
    l4_ctv = l4_cultivated.lc_l4_cultivated(
        xx.classes_l3_l4, level3, lifeform, veg_cover
    )
    l4_ctv_ntv = l4_natural_veg.lc_l4_natural_veg(l4_ctv, level3, lifeform, veg_cover)

    water_seasonality = stats_l4.define_water_seasonality(xx)
    l4_ctv_ntv_nav = l4_natural_aquatic.natural_auquatic_veg(
        l4_ctv_ntv, lifeform, veg_cover, water_seasonality
    )
    assert (l4_ctv_ntv_nav.compute() == expected_l4_ntv_classes).all()


def test_ntv_herbaceous_seasonal_water_veg_cover():
    expected_l4_ntv_classes = [
        [92, 83, 80],
        [80, 83, 83],
        [92, 86, 86],
        [89, 92, 83],
    ]

    l34 = np.array(
        [
            [
                [124, 124, 124],
                [124, 124, 124],
                [124, 124, 124],
                [124, 124, 124],
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
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
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
    water_frequency = np.array(
        [
            [
                [1, 2, 3],
                [1, 2, 3],
                [1, 2, 3],
                [1, 2, 3],
            ]
        ],
        dtype="int",
    )

    xx = image_groups(l34, urban, cultivated, woody, pv_pc_50, water_frequency)

    stats_l4 = StatsLccsLevel4()
    intertidal_mask, level3 = lc_level3.lc_level3(xx)
    lifeform = stats_l4.define_life_form(xx)
    veg_cover = l4_veg_cover.canopyco_veg_con(xx, stats_l4.veg_threshold)
    veg_cover = stats_l4.apply_mapping(veg_cover, stats_l4.veg_mapping)

    # Apply cultivated to match the code in Level4 processing
    l4_ctv = l4_cultivated.lc_l4_cultivated(
        xx.classes_l3_l4, level3, lifeform, veg_cover
    )
    l4_ctv_ntv = l4_natural_veg.lc_l4_natural_veg(l4_ctv, level3, lifeform, veg_cover)

    water_seasonality = stats_l4.define_water_seasonality(xx)
    l4_ctv_ntv_nav = l4_natural_aquatic.natural_auquatic_veg(
        l4_ctv_ntv, lifeform, veg_cover, water_seasonality
    )

    assert (l4_ctv_ntv_nav.compute() == expected_l4_ntv_classes).all()


def test_ntv_herbaceous_permanent_water_veg_cover():
    expected_l4_ntv_classes = [
        [91, 82, 79],
        [79, 82, 82],
        [91, 85, 85],
        [88, 91, 82],
    ]

    l34 = np.array(
        [
            [
                [124, 124, 124],
                [124, 124, 124],
                [124, 124, 124],
                [124, 124, 124],
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
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
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
    water_frequency = np.array(
        [
            [
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
                [11, 10, 8],
            ]
        ],
        dtype="int",
    )

    xx = image_groups(l34, urban, cultivated, woody, pv_pc_50, water_frequency)

    stats_l4 = StatsLccsLevel4()
    intertidal_mask, level3 = lc_level3.lc_level3(xx)
    lifeform = stats_l4.define_life_form(xx)
    veg_cover = l4_veg_cover.canopyco_veg_con(xx, stats_l4.veg_threshold)
    veg_cover = stats_l4.apply_mapping(veg_cover, stats_l4.veg_mapping)

    # Apply cultivated to match the code in Level4 processing
    l4_ctv = l4_cultivated.lc_l4_cultivated(
        xx.classes_l3_l4, level3, lifeform, veg_cover
    )
    l4_ctv_ntv = l4_natural_veg.lc_l4_natural_veg(l4_ctv, level3, lifeform, veg_cover)

    water_seasonality = stats_l4.define_water_seasonality(xx)
    l4_ctv_ntv_nav = l4_natural_aquatic.natural_auquatic_veg(
        l4_ctv_ntv, lifeform, veg_cover, water_seasonality
    )
    assert (l4_ctv_ntv_nav.compute() == expected_l4_ntv_classes).all()
