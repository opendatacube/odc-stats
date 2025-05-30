# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2025 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
from datacube.testutils import mk_sample_dataset

from odc.stats._grouper import group_by_nothing, key2num, mid_longitude, solar_offset
from odc.geo.geobox import GeoBox
from odc.geo.geom import box as geom_box


@pytest.mark.parametrize("lon,lat", [(0, 10), (100, -10), (-120, 30)])
def test_mid_lon(lon, lat):
    r = 0.1
    rect = geom_box(lon - r, lat - r, lon + r, lat + r, "epsg:4326")
    assert rect.centroid.coords[0] == pytest.approx((lon, lat))

    assert mid_longitude(rect) == pytest.approx(lon)
    assert mid_longitude(rect.to_crs("epsg:3857")) == pytest.approx(lon)

    offset = solar_offset(rect, "h")
    assert offset.seconds % (60 * 60) == 0

    offset_sec = solar_offset(rect, "s")
    assert abs((offset - offset_sec).seconds) <= 60 * 60


@pytest.mark.parametrize(
    "input_,expect",
    [
        ("ABAAC", [0, 1, 0, 0, 2]),
        ("B", [0]),
        ([1, 1, 1], [0, 0, 0]),
        ("ABCC", [0, 1, 2, 2]),
    ],
)
def test_key2num(input_, expect):
    rr = list(key2num(input_))
    assert rr == expect

    reverse = {}
    rr = list(key2num(input_, reverse))
    assert rr == expect
    assert set(reverse.keys()) == set(range(len(set(input_))))
    assert set(reverse.values()) == set(input_)
    # first entry always gets an index of 0
    assert reverse[0] == input_[0]


@pytest.fixture
def sample_geobox():
    yield GeoBox.from_geopolygon(geom_box(-10, -20, 11, 22, "epsg:4326"), resolution=1)


@pytest.fixture
def sample_ds(sample_geobox):
    yield mk_sample_dataset([{"name": "red"}], geobox=sample_geobox)


def test_grouper(sample_ds):
    xx = group_by_nothing([sample_ds])
    assert xx.values[0] == (sample_ds,)
    assert xx.uuid.values[0] == sample_ds.id

    xx = group_by_nothing([sample_ds, sample_ds], solar_offset(sample_ds.extent))
    assert xx.values[0] == (sample_ds,)
    assert xx.values[0] == (sample_ds,)
    assert xx.uuid.values[1] == sample_ds.id
    assert xx.uuid.values[1] == sample_ds.id
