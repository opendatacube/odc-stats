import pytest

from odc.stats.tasks import sanitize_products_str, SaveTasks
from odc.stats.model import DateTimeRange
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


@pytest.fixture(
    params=[
        # union
        (
            "-+-ga_ls8c_ard_3-+++-ga_ls7e_ard_3+-+-",
            0,
            14,
            [("ga_ls8c_ard_3", True), ("ga_ls7e_ard_3", True)],
        ),
        # intersect/groupby time
        (
            "+-+ga_ls8c_ard_3+---++ga_ls7e_ard_3-+-+",
            1,
            0,
            [("ga_ls8c_ard_3", True), ("ga_ls7e_ard_3", True)],
        ),
        # fc doesn't have datasets for ls7
        (
            "-+-ga_ls8c_ard_3-+++-ga_ls7e_ard_3--+-ga_ls_fc_3-+-+",
            0,
            7,
            [("ga_ls8c_ard_3", True), ("ga_ls7e_ard_3", True), ("ga_ls_fc_3", True)],
        ),
        # wo has dataset for all sensors
        (
            "-+-ga_ls8c_ard_3++++-ga_ls7e_ard_3+-+-ga_ls_wo_3-+-+",
            2,
            14,
            [("ga_ls8c_ard_3", True), ("ga_ls7e_ard_3", True), ("ga_ls_wo_3", True)],
        ),
        # non indexed datasets
        (
            (
                "--+++s3://dea-public-data/derivative/ga_ls_fc_pc_cyear_3/3-0-0/+--+"
                "s3://dea-public-data/derivative/ga_ls_tc_pc_cyear_3/1-0-0/+++"
                "ga_ls_wo_fq_cyear_3+---ga_ls_fc_pc_cyear+-++"
            ),
            3,
            0,
            [
                ("s3://dea-public-data/derivative/ga_ls_fc_pc_cyear_3/3-0-0/", False),
                ("s3://dea-public-data/derivative/ga_ls_tc_pc_cyear_3/1-0-0/", False),
                ("ga_ls_wo_fq_cyear_3", True),
                ("ga_ls_fc_pc_cyear", True),
            ],
        ),
        (
            (
                "+++ga_ls_wo_fq_cyear_3+-+++s3://dea-public-data/derivative/ga_ls_fc_pc_cyear_3/3-0-0/"
                "+--+s3://dea-public-data/derivative/ga_ls_tc_pc_cyear_3/1-0-0/"
                "+---ga_ls_fc_pc_cyear+-++"
            ),
            3,
            0,
            [
                ("ga_ls_wo_fq_cyear_3", True),
                ("s3://dea-public-data/derivative/ga_ls_fc_pc_cyear_3/3-0-0/", False),
                ("s3://dea-public-data/derivative/ga_ls_tc_pc_cyear_3/1-0-0/", False),
                ("ga_ls_fc_pc_cyear", True),
            ],
        ),
        (
            (
                "+++ga_ls_wo_fq_cyear_3+--+s3://dea-public-data/derivative/ga_ls_tc_pc_cyear_3/1-0-0/"
                "+---ga_ls_fc_pc_cyear+-+++s3://dea-public-data/derivative/ga_ls_fc_pc_cyear_3/3-0-0/-+++"
            ),
            3,
            0,
            [
                ("ga_ls_wo_fq_cyear_3", True),
                ("s3://dea-public-data/derivative/ga_ls_tc_pc_cyear_3/1-0-0/", False),
                ("ga_ls_fc_pc_cyear", True),
                ("s3://dea-public-data/derivative/ga_ls_fc_pc_cyear_3/3-0-0/", False),
            ],
        ),
        (
            (
                "+++ga_ls_wo_fq_cyear_3----ga_ls_fc_pc_cyear---+"
                "s3://dea-public-data/derivative/ga_ls_tc_pc_cyear_3/1-0-0/"
                "--+++s3://dea-public-data/derivative/ga_ls_fc_pc_cyear_3/3-0-0/-+++"
            ),
            0,
            0,
            [
                ("ga_ls_wo_fq_cyear_3", True),
                ("ga_ls_fc_pc_cyear", True),
                ("s3://dea-public-data/derivative/ga_ls_tc_pc_cyear_3/1-0-0/", False),
                ("s3://dea-public-data/derivative/ga_ls_fc_pc_cyear_3/3-0-0/", False),
            ],
        ),
    ]
)
def product_str(request):
    return request.param


@pytest.fixture
def indexed_product_str(product_str):
    if "s3" not in product_str[0]:
        return product_str


@pytest.fixture
def s3_path():
    return [
        "s3://dea-public-data/derivative/ga_ls_tc_pc_cyear_3/1-0-0/",
        "s3://dea-public-data/derivative/ga_ls_fc_pc_cyear_3/3-0-0/",
    ]


def test_sanitize_products_str(product_str):
    product_list, group_size = sanitize_products_str(product_str[0])
    assert product_list == product_str[-1]
    assert group_size == product_str[1]


def test_create_dss_by_stac(s3_path):
    temporal_range = DateTimeRange("2010--P2Y")
    tiles = ((35, 37), (32, 34))
    dss, product = SaveTasks.create_dss_by_stac(
        s3_path, tiles=tiles, temporal_range=temporal_range
    )
    dss = list(dss)
    assert len(product) == len(s3_path)
    assert len(dss) == 4 * 2 * len(s3_path)
    for p, key in zip(product, s3_path):
        assert p.name == key.split("/")[-3]
    for d in dss:
        with_uris = False
        for key in s3_path:
            with_uris |= key in d.uris[0]
        assert with_uris


def test_find_dss(dc, query, indexed_product_str):
    dss = SaveTasks._find_dss(dc, indexed_product_str[0], query, {}, fuse_dss=False)
    assert list(dss) == indexed_product_str[2]
