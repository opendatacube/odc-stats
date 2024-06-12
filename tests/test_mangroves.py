import numpy as np
import xarray as xr
import dask.array as da
import os
from odc.stats.plugins.mangroves import Mangroves
import tempfile
import json
import fiona
from fiona.crs import CRS
from datacube.utils.geometry import GeoBox
from affine import Affine
import pytest


@pytest.fixture
def mangrove_shape():
    data = """
    {
   "type":"FeatureCollection",
   "features":[
      {
         "geometry":{
            "type":"Polygon",
            "coordinates":[
               [
                  [
                     0,
                     0
                  ],
                  [
                     0,
                     100
                  ],
                  [
                     100,
                     100
                  ],
                  [
                     100,
                     0
                  ],
                  [
                     0,
                     0
                  ]
               ]
            ]
         },
         "type":"Feature"
      }
   ]
}
    """
    data = json.loads(data)["features"][0]
    tmpdir = tempfile.mkdtemp()
    filename = os.path.join(tmpdir, "test.json")
    with fiona.open(
        filename,
        "w",
        driver="GeoJSON",
        crs=CRS.from_epsg(3577),
        schema={
            "geometry": "Polygon",
        },
    ) as dst:
        dst.write(data)
    return filename


@pytest.fixture
def dataset():
    band_1 = np.array(
        [
            [
                [255, 57],
                [20, 50],
                [10, 15],
                [30, 40],
                [65, 80],
                [20, 39],
                [90, 52],
                [73, 98],
                [30, 40],
            ],
        ]
    ).astype(np.uint8)

    band_2 = np.array(
        [
            [[0, 1], [2, 2], [1, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [1, 1]],
        ]
    ).astype(np.uint8)

    band_3 = np.array(
        [
            [
                [-1849, 0],
                [-1851, 0],
                [0, 0],
                [0, 5],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 45],
                [0, 0],
            ],
        ]
    ).astype(np.int16)

    band_1 = da.from_array(band_1, chunks=(1, -1, -1))
    band_2 = da.from_array(band_2, chunks=(1, -1, -1))
    band_3 = da.from_array(band_3, chunks=(1, -1, -1))

    index = [np.datetime64("2000-01-01T00")]

    affine = Affine.translation(10, 0) * Affine.scale(
        (20 - 10) / band_1.shape[2], (5 - 0) / band_1.shape[1]
    )
    geobox = GeoBox(
        crs="epsg:3577", affine=affine, width=band_1.shape[2], height=band_1.shape[1]
    )
    coords = geobox.xr_coords()
    coords.update({"time": index})

    data_vars = {
        "pv_pc_10": xr.DataArray(
            band_1, dims=("time", "y", "x"), attrs={"nodata": 255}
        ),
        "qa": xr.DataArray(band_2, dims=("time", "y", "x")),
        "wet_pc_10": xr.DataArray(
            band_3, dims=("time", "y", "x"), attrs={"nodata": -9999}
        ),
    }
    xx = xr.Dataset(data_vars=data_vars, coords=coords)

    return xx


def test_native_transform(dataset, mangrove_shape):
    mangroves = Mangroves(mangroves_extent=mangrove_shape)
    out_xx = mangroves.native_transform(dataset)
    assert (out_xx == dataset).all()


def test_reduce(dataset, mangrove_shape):
    mangroves = Mangroves(mangroves_extent=mangrove_shape)
    yy = mangroves.reduce(dataset)
    expected_results = dataset.pv_pc_10.copy(True)
    expected_results.data = np.array(
        [[[255, 0], [255, 2], [0, 1], [1, 2], [3, 3], [1, 2], [3, 2], [3, 3], [0, 0]]],
        dtype=np.uint8,
    )
    expected_results.attrs["nodata"] = 255
    expected_results = expected_results.to_dataset(name="canopy_cover_class")
    assert yy.canopy_cover_class.dtype == np.uint8
    assert yy.canopy_cover_class.attrs == expected_results.canopy_cover_class.attrs
    assert (yy == expected_results).all()
