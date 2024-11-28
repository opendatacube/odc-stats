from odc.stats.plugins.lc_level34 import StatsLccsLevel4
from odc.stats.plugins._utils import generate_numexpr_expressions

import re
import os
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
from datacube.utils.geometry import GeoBox
from affine import Affine
from unittest.mock import patch


import pytest


NODATA = 255


@pytest.fixture(scope="module")
def image_groups():
    l34 = np.array(
        [
            [
                [210, 210, 210],
                [210, 210, 210],
                [223, 210, 210],
                [221, 221, 221],
            ]
        ],
        dtype="uint8",
    )

    urban = np.array(
        [
            [
                [216, 216, 215],
                [216, 216, 216],
                [215, 215, 215],
                [215, 215, 215],
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
                [1, 64, 65],
                [66, 40, 41],
                [3, 61, 78],
                [4, 23, 42],
            ]
        ],
        dtype="uint8",
    )

    bs_pc_50 = np.array(
        [
            [
                [1, 64, NODATA],
                [66, 40, 41],
                [1, 40, 66],
                [NODATA, 1, 42],
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

    water_frequency = np.array(
        [
            [
                [1, 3, 2],
                [4, 5, 6],
                [2, 2, 11],
                [10, 11, 12],
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

    tuples = [
        (np.datetime64("2000-01-01T00"), np.datetime64("2000-01-01")),
    ]
    index = pd.MultiIndex.from_tuples(tuples, names=["time", "solar_day"])

    affine = Affine.translation(10, 0) * Affine.scale(
        (20 - 10) / l34.shape[2], (5 - 0) / l34.shape[1]
    )
    geobox = GeoBox(
        crs="epsg:3577", affine=affine, width=l34.shape[2], height=l34.shape[1]
    )
    coords = geobox.xr_coords()

    data_vars = {
        "level_3_4": xr.DataArray(
            da.from_array(l34, chunks=(1, -1, -1)),
            dims=("spec", "y", "x"),
            attrs={"nodata": 255},
        ),
        "artificial_surface": xr.DataArray(
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
        "bs_pc_50": xr.DataArray(
            da.from_array(bs_pc_50, chunks=(1, -1, -1)),
            dims=("spec", "y", "x"),
            attrs={"nodata": 255},
        ),
        "water_frequency": xr.DataArray(
            da.from_array(water_frequency, chunks=(1, -1, -1)),
            dims=("spec", "y", "x"),
            attrs={"nodata": 255},
        ),
        "water_season": xr.DataArray(
            da.from_array(water_season, chunks=(1, -1, -1)),
            dims=("spec", "y", "x"),
            attrs={"nodata": 255},
        ),
    }

    xx = xr.Dataset(data_vars=data_vars, coords=coords)
    xx = xx.assign_coords(xr.Coordinates.from_pandas_multiindex(index, "spec"))
    return xx


def test_l4_classes(image_groups, urban_shape):
    expected_l3 = [[216, 216, 215], [216, 216, 216], [220, 215, 215], [220, 220, 220]]

    expected_l4 = [[95, 97, 93], [97, 96, 96], [100, 93, 93], [101, 101, 101]]
    with patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "fake-access-key",
            "AWS_SECRET_ACCESS_KEY": "fake-secret-key",
            "AWS_SESSION_TOKEN": "fake-session-token",  # Optional
        },
    ):
        stats_l4 = StatsLccsLevel4(
            measurements=["level3", "level4"],
            class_def_path="s3://dea-public-data-dev/lccs_validation/c3/data_to_plot/"
            "lccs_colour_scheme_golden_dark_au_c3.csv",
            class_condition={
                "level3": ["level1", "artificial_surface", "cultivated"],
                "level4": [
                    "level1",
                    "level3",
                    "woody",
                    "water_season",
                    "water_frequency",
                    "pv_pc_50",
                    "bs_pc_50",
                ],
            },
            data_var_condition={"level1": "level_3_4"},
            urban_mask=urban_shape,
            filter_expression="mock > 9",
            mask_threshold=0.3,
        )
    ds = stats_l4.reduce(image_groups)

    assert (ds.level3.compute() == expected_l3).all()
    assert (ds.level4.compute() == expected_l4).all()


@pytest.mark.parametrize(
    "rules_df, expected_expressions",
    [
        # Test with range conditions
        # when condition numbers are the same the order doesn't matter
        (
            pd.DataFrame(
                {
                    "condition_1": ["[5, 10)", "(1, 4]"],
                    "condition_2": ["==2", "!=2"],
                    "final_class": [1, 2],
                }
            ),
            [
                "where((condition_1>1.0)&(condition_1<=4.0)&(condition_2!=2.0), 2, previous)",
                "where((condition_1>=5.0)&(condition_1<10.0)&(condition_2==2.0), 1, previous)",
            ],
        ),
        # Test with NaN
        # when clause with smaller number of conditions always takes precedence
        (
            pd.DataFrame(
                {
                    "condition_1": ["[5, 10)", "nan"],
                    "condition_2": ["==2", "!=2"],
                    "final_class": [1, 2],
                }
            ),
            [
                "where((condition_2!=2.0), 2, previous)",
                "where((condition_1>=5.0)&(condition_1<10.0)&(condition_2==2.0), 1, previous)",
            ],
        ),
        # Test with single value implying "==" and "255"
        (
            pd.DataFrame(
                {
                    "condition_1": ["3", "255"],
                    "condition_2": ["==2", "!=2"],
                    "final_class": [1, 2],
                }
            ),
            [
                "where((condition_2!=2.0), 2, previous)",
                "where((condition_1==3)&(condition_2==2.0), 1, previous)",
            ],
        ),
    ],
)
def test_generate_numexpr_expressions(rules_df, expected_expressions):
    con_cols = ["condition_1", "condition_2"]
    class_col = "final_class"

    generated_expressions = generate_numexpr_expressions(
        rules_df[con_cols + [class_col]], class_col, "previous"
    )

    def normalize_expression(expression):
        match = re.match(r"where\((.*), (.*?), (.*?)\)", expression)
        if match:
            conditions, true_value, false_value = match.groups()
            # Split conditions, sort them, and rejoin
            sorted_conditions = "&".join(sorted(conditions.split("&")))
            return f"where({sorted_conditions}, {true_value}, {false_value})"
        return expression

    normalized_expected = [normalize_expression(expr) for expr in expected_expressions]
    normalized_generated = [
        normalize_expression(expr) for expr in generated_expressions
    ]

    assert (
        normalized_generated == normalized_expected
    ), f"Expected {expected_expressions}, but got {generated_expressions}"
