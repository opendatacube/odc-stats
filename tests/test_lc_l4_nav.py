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
    l4_natural_aquatic,
    lc_lifeform,
)

import pandas as pd

NODATA = 255


def image_groups(l34, urban, cultivated, woody, pv_pc_50, water_frequency, water_season):

    tuples = [
        (np.datetime64("2000-01-01T00"), np.datetime64("2000-01-01")),
    ]
    index = pd.MultiIndex.from_tuples(tuples, names=["time", "solar_day"])
    coords = {
        "x": np.linspace(10, 20, l34.shape[2]),
        "y": np.linspace(0, 5, l34.shape[1]),
    }

    data_vars = {
        "level_3_4": xr.DataArray(
            da.from_array(l34, chunks=(1, -1, -1)),
            dims=("spec", "y", "x"),
            attrs={"nodata": 255},
        ),
        "urban_classes": xr.DataArray(
            da.from_array(urban, chunks=(1, -1, -1)),
            dims=("spec", "y", "x"),
            attrs={"nodata": 255},
        ),
        "cultivated": xr.DataArray(
            da.from_array(cultivated, chunks=(1, -1, -1)),
            dims=("spec", "y", "x"),
            attrs={"nodata": 255},
        ),
        "woody": xr.DataArray(
            da.from_array(woody, chunks=(1, -1, -1)),
            dims=("spec", "y", "x"),
            attrs={"nodata": 255},
        ),
        "pv_pc_50": xr.DataArray(
            da.from_array(pv_pc_50, chunks=(1, -1, -1)),
            dims=("spec", "y", "x"),
            attrs={"nodata": 255},
        ),
        "water_frequency": xr.DataArray(
            da.from_array(water_frequency, chunks=(1, -1, -1)),
            dims=("spec", "y", "x"),
            attrs={"nodata": 255},
        ),
        "water_season":xr.DataArray(
            da.from_array(water_season, chunks=(1, -1, -1)),
            dims=("spec", "y", "x"),
            attrs={"nodata": 255},
        ),
    }
    xx = xr.Dataset(data_vars=data_vars, coords=coords)
    xx = xx.assign_coords(xr.Coordinates.from_pandas_multiindex(index, "spec"))
    return xx


def test_ntv_classes_woody_herbaceous():
    expected_l4_ntv_classes = [[56, 56, 56], [57, 57, 57], [56, 56, 56], [57, 57, 57]]

    l34 = np.array(
        [
            [
                [124, 124, 124],
                [125, 125, 125],
                [124, 124, 124],
                [125, 125, 125],
            ]
        ],
        dtype="uint8",
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
        dtype="uint8",
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
        dtype="uint8",
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
        dtype="uint8",
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
        dtype="uint8",
    )

    water_frequency = np.array(
        [
            [
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
            ]
        ],
        dtype="uint8",
    )
 
    water_season = np.array(
        [
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        ],
        dtype="uint8",
    )

    xx = image_groups(l34, urban, cultivated, woody, pv_pc_50, water_frequency, water_season)

    stats_l4 = StatsLccsLevel4()
    level3 = lc_level3.lc_level3(xx)
    lifeform = lc_lifeform.lifeform(xx)
    veg_cover = l4_veg_cover.canopyco_veg_con(xx, stats_l4.veg_threshold)

    # Apply cultivated to match the code in Level4 processing
    l4_ctv = l4_cultivated.lc_l4_cultivated(xx.level_3_4, level3, lifeform, veg_cover)
    l4_ctv_ntv = l4_natural_veg.lc_l4_natural_veg(l4_ctv, level3, lifeform, veg_cover)

    l4_ctv_ntv_nav = l4_natural_aquatic.natural_auquatic_veg(
        l4_ctv_ntv, veg_cover, xx.water_season
    )

    assert (l4_ctv_ntv_nav.compute() == expected_l4_ntv_classes).all()


def test_ntv_herbaceous_seasonal_water_veg_cover():
    expected_l4_ntv_classes = [
        [91, 83, 79],
        [80, 82, 83],
        [91, 85, 86],
        [89, 92, 82],
    ]

    l34 = np.array(
        [
            [
                [125, 125, 125],
                [125, 125, 125],
                [125, 125, 125],
                [125, 125, 125],
            ]
        ],
        dtype="uint8",
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
        dtype="uint8",
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
        dtype="uint8",
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
        dtype="uint8",
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
        dtype="uint8",
    
    )
    water_frequency = np.array(
        [
            [
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
            ]
        ],
        dtype="uint8",
    )
  
    water_season = np.array(
        [
            [
                [1, 2, 1],
                [2, 1, 2],
                [1, 1, 2],
                [2, 2, 1],
            ]
        ],
        dtype="uint8",
    )

    xx = image_groups(l34, urban, cultivated, woody, pv_pc_50, water_frequency, water_season)

    stats_l4 = StatsLccsLevel4()
    level3 = lc_level3.lc_level3(xx)
    lifeform = lc_lifeform.lifeform(xx)
    veg_cover = l4_veg_cover.canopyco_veg_con(xx, stats_l4.veg_threshold)

    # Apply cultivated to match the code in Level4 processing
    l4_ctv = l4_cultivated.lc_l4_cultivated(xx.level_3_4, level3, lifeform, veg_cover)
    l4_ctv_ntv = l4_natural_veg.lc_l4_natural_veg(l4_ctv, level3, lifeform, veg_cover)

    l4_ctv_ntv_nav = l4_natural_aquatic.natural_auquatic_veg(
        l4_ctv_ntv, veg_cover, xx.water_season
    )
 
    assert (l4_ctv_ntv_nav.compute() == expected_l4_ntv_classes).all()


def test_ntv_woody_seasonal_water_veg_cover():
    expected_l4_ntv_classes = [
        [76, 68, 64],
        [65, 67, 68],
        [76, 70, 71],
        [74, 77, 67],
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
        dtype="uint8",
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
        dtype="uint8",
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
        dtype="uint8",
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
        dtype="uint8",
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
        dtype="uint8",
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
        dtype="uint8",
    )
   
    water_season = np.array(
        [
            [
                [1, 2, 1],
                [2, 1, 2],
                [1, 1, 2],
                [2, 2, 1],
            ]
        ],
        dtype="uint8",
    )
    xx = image_groups(l34, urban, cultivated, woody, pv_pc_50, water_frequency, water_season)

    stats_l4 = StatsLccsLevel4()
    level3 = lc_level3.lc_level3(xx)
    lifeform = lc_lifeform.lifeform(xx)
    veg_cover = l4_veg_cover.canopyco_veg_con(xx, stats_l4.veg_threshold)

    # Apply cultivated to match the code in Level4 processing
    l4_ctv = l4_cultivated.lc_l4_cultivated(xx.level_3_4, level3, lifeform, veg_cover)
    l4_ctv_ntv = l4_natural_veg.lc_l4_natural_veg(l4_ctv, level3, lifeform, veg_cover)

    l4_ctv_ntv_nav = l4_natural_aquatic.natural_auquatic_veg(
        l4_ctv_ntv, veg_cover, xx.water_season
    )

    assert (l4_ctv_ntv_nav.compute() == expected_l4_ntv_classes).all()
