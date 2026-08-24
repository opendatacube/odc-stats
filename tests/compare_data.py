#!/usr/bin/env python3
"""Compare a generated GeoTIFF against a golden file.

Replaces gdalcompare.py, which ships with the osgeo Python bindings and so is
not available when GDAL comes from the rasterio wheel. Compares the geospatial
metadata and the pixel data of every band, which is what
``gdalcompare.py -sds -skip_binary`` did for these single-subdataset COGs.
"""

import os
import sys

import numpy as np
import rasterio


def compare(golden: str, candidate: str) -> list[str]:
    differences = []
    with rasterio.open(golden) as g, rasterio.open(candidate) as c:
        for attr in ("count", "width", "height", "dtypes", "nodatavals", "crs"):
            expected, actual = getattr(g, attr), getattr(c, attr)
            if expected != actual:
                differences.append(f"{attr}: golden {expected!r} != {actual!r}")

        if not g.transform.almost_equals(c.transform):
            differences.append(f"transform: golden {g.transform} != {c.transform}")

        if differences:
            # Shapes/band counts may not line up, so do not read the pixels.
            return differences

        for band in g.indexes:
            expected, actual = g.read(band), c.read(band)
            # equal_nan needs a float dtype; integer bands use nodata instead.
            equal_nan = np.issubdtype(expected.dtype, np.floating)
            if not np.array_equal(expected, actual, equal_nan=equal_nan):
                differing = int(np.count_nonzero(expected != actual))
                differences.append(f"band {band}: {differing} pixels differ")

    return differences


def main() -> int:
    golden, candidate = sys.argv[1], sys.argv[2]
    with rasterio.Env(
        AWS_NO_SIGN_REQUEST=os.environ.get("AWS_NO_SIGN_REQUEST", "NO"),
        GDAL_HTTP_MAX_RETRY=os.environ.get("GDAL_HTTP_MAX_RETRY", "0"),
    ):
        differences = compare(golden, candidate)

    for difference in differences:
        print(f"  {difference}", file=sys.stderr)
    return 1 if differences else 0


if __name__ == "__main__":
    sys.exit(main())
