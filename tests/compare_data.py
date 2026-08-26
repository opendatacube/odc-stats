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


def nodata_equal(expected, actual) -> bool:
    """Compare nodata tuples, treating nan as equal to nan."""
    if len(expected) != len(actual):
        return False
    return all(
        e == a
        or (e is not None and a is not None and np.isnan(e) and np.isnan(a))
        or e is None  # skip if golden nodata not set
        for e, a in zip(expected, actual)
    )


def compare(golden: str, candidate: str) -> list[str]:
    differences = []
    with rasterio.open(golden) as g, rasterio.open(candidate) as c:
        for attr in ("count", "width", "height", "dtypes", "nodatavals", "crs"):
            expected, actual = getattr(g, attr), getattr(c, attr)
            if attr == "nodatavals":
                same = nodata_equal(expected, actual)
            else:
                same = expected == actual
            if not same:
                differences.append(f"{attr}: golden {expected!r} != {actual!r}")

        if not g.transform.almost_equals(c.transform):
            differences.append(f"transform: golden {g.transform} != {c.transform}")

        if differences:
            # Shapes/band counts may not line up, so do not read the pixels.
            return differences

        for band in g.indexes:
            expected, actual = g.read(band), c.read(band)

            # Handle NaNs appropriately
            if np.issubdtype(expected.dtype, np.floating):
                diff_mask = ~(np.isnan(expected) & np.isnan(actual))
                differing = int(np.count_nonzero(diff_mask & (expected != actual)))
                total = int(np.count_nonzero(~np.isnan(expected)))

                abs_diff = np.abs(actual - expected)
                mean_change = float(np.nanmean(abs_diff))
                max_change = float(np.nanmax(abs_diff))
            else:
                differing = int(np.count_nonzero(expected != actual))
                total = expected.size

                abs_diff = np.abs(
                    actual.astype(np.float64) - expected.astype(np.float64)
                )
                mean_change = float(abs_diff.mean())
                max_change = float(abs_diff.max())

            if differing:
                differences.append(
                    f"band {band}: {differing} pixels differ "
                    f"({differing/total:.0%}), "
                    f"mean change={mean_change:.3f}, "
                    f"max change={max_change:.3f}"
                )
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
