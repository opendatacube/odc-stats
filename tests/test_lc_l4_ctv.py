import numpy as np
import xarray as xr
import dask.array as da

from odc.stats.plugins.lc_level34 import StatsLccsLevel4
from odc.stats.plugins.l34_utils import (
    l4_cultivated,
    lc_level3,
    l4_veg_cover,
    lc_lifeform,
)

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
    }
    xx = xr.Dataset(data_vars=data_vars, coords=coords)
    xx = xx.assign_coords(xr.Coordinates.from_pandas_multiindex(index, "spec"))
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
        dtype="uint8",
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
    xx = image_groups(l34, urban, cultivated, woody, pv_pc_50)

    stats_l4 = StatsLccsLevel4()
    level3 = lc_level3.lc_level3(xx)

    lifeform = lc_lifeform.lifeform(xx)
    veg_cover = l4_veg_cover.canopyco_veg_con(xx, stats_l4.veg_threshold)

    l4_ctv = l4_cultivated.lc_l4_cultivated(
        xx.level_3_4, level3, lifeform, veg_cover
    )

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
                [111, 111, 111],
                [255, 111, 111],
                [111, 111, 111],
                [111, 111, 111],
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
    xx = image_groups(l34, urban, cultivated, woody, pv_pc_50)

    stats_l4 = StatsLccsLevel4()
    level3 = lc_level3.lc_level3(xx)
    lifeform = lc_lifeform.lifeform(xx)
    veg_cover = l4_veg_cover.canopyco_veg_con(xx, stats_l4.veg_threshold)

    l4_ctv = l4_cultivated.lc_l4_cultivated(
        xx.level_3_4, level3, lifeform, veg_cover
    )
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
    level3 = lc_level3.lc_level3(xx)
    lifeform = lc_lifeform.lifeform(xx)
    veg_cover = l4_veg_cover.canopyco_veg_con(xx, stats_l4.veg_threshold)

    l4_ctv = l4_cultivated.lc_l4_cultivated(
        xx.level_3_4, level3, lifeform, veg_cover
    )

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
    level3 = lc_level3.lc_level3(xx)
    lifeform = lc_lifeform.lifeform(xx)
    veg_cover = l4_veg_cover.canopyco_veg_con(xx, stats_l4.veg_threshold)

    l4_ctv = l4_cultivated.lc_l4_cultivated(
        xx.level_3_4, level3, lifeform, veg_cover
    )
    assert (l4_ctv.compute() == expected_cultivated_classes).all()
