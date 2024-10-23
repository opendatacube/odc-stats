from odc.stats.plugins.lc_level3 import StatsLccsLevel4
import pytest

expected_l4_classes = [
    [111, 112, 215],
    [124, 112, 215],
    [220, 215, 216],
    [220, 255, 220],
]

NODATA = 255

@pytest.fixture(scope="module")
def image_groups():
    l34 = np.array(
        [
            [
                [110, 110, 210],
                [124, 110, 210],
                [221, 210, 210],
                [223, 255, 223],
            ]
        ],
        dtype="int",
    )

    urban = np.array(
        [
            [
                [215, 215, 215],
                [216, 216, 215],
                [116, 215, 216],
                [216, 216, 216],
            ]
        ],
        dtype="int",
    )

    cultivated = np.array(
        [
            [
                [111, 112, 255],
                [255, 112, 255],
                [255, 255, 255],
                [255, 255, 255],
            ]
        ],
        dtype="int",
    )

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
    }
    xx = xr.Dataset(data_vars=data_vars, coords=coords)
    return xx


def test_l4_classes(image_groups):

    stats_l4 = StatsLccsLevel4()
    intertidal_mask, level3 = lc_level3.lc_level3(xx, NODATA)
    # intertidal_mask, level3_classes = lc_level3.reduce(image_groups)

    assert (level3_classes == expected_l3_classes).all()