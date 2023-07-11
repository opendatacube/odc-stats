import pytest

from odc.stats.tasks import SaveTasks
from datacube import Datacube


@pytest.fixture
def query():
    return {
        "time": ("2015-01-01", "2016-01-01"),
        "lat": (-28.377045384680866, -26.469315120533185),
        "lon": (152.04838917952736, 154.50423588612105),
    }


@pytest.fixture
def dc():
    return Datacube(app="test")


@pytest.fixture
def product_str():
    return {
        0: "-+-ga_ls8c_ard_3-+++-ga_ls7e_ard_3+-+-",
        1: "+-+ga_ls8c_ard_3+---++ga_ls7e_ard_3-+-+",
        2: "-+-ga_ls8c_ard_3-+++-ga_ls7e_ard_3+-+-ga_ls_fc_3-+-+",
        3: "-+-ga_ls8c_ard_3-+++-ga_ls7e_ard_3+-+-ga_ls_wo_3-+-+",
    }


def test_find_dss(dc, query, product_str):
    for k, v in product_str.items():
        dss = SaveTasks._find_dss(dc, v, query, fuse_dss=False)
        dss = list(dss)
        if k == 0:
            # union
            assert len(dss) == 14
        elif k == 1:
            # intersect/groupby time
            assert len(dss) == 0
        elif k == 2:
            # fc doesn't have datasets for ls7
            assert len(dss) == 7
            assert len(dss[0]) >= 2
        elif k == 3:
            # wo has dataset for all sensors
            assert len(dss) == 14
            assert len(dss[0]) >= 2
