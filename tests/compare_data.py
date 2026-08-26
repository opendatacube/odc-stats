#!/usr/bin/env python3
"""Compare a generated GeoTIFF against a golden file.

Replaces gdalcompare.py, which ships with the osgeo Python bindings and so is
not available when GDAL comes from the rasterio wheel. Compares the geospatial
metadata and the pixel data of every band, which is what
``gdalcompare.py -sds -skip_binary`` did for these single-subdataset COGs.
"""

import argparse
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


def compare(
    golden: str, candidate: str, max_diff_pct: float = 0.0
) -> tuple[list[str], list[str]]:
    """Compare two rasters.

    Bands where fewer than ``max_diff_pct`` percent of the pixels differ are
    reported as tolerated rather than as differences.

    Returns ``(differences, tolerated)``.
    """
    differences = []
    tolerated = []
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
            return differences, tolerated

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
                diff_pct = 100 * differing / total if total else 100.0
                message = (
                    f"band {band}: {differing} pixels differ "
                    f"({diff_pct:.2f}%), "
                    f"mean change={mean_change:.3f}, "
                    f"max change={max_change:.3f}"
                )
                if diff_pct < max_diff_pct:
                    tolerated.append(f"{message} [under {max_diff_pct}% threshold]")
                else:
                    differences.append(message)
    return differences, tolerated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("golden", help="path to the golden file")
    parser.add_argument("candidate", help="path to the generated file")
    parser.add_argument(
        "--max-diff-pct",
        type=float,
        default=0.0,
        metavar="PCT",
        help="ignore a band whose differing pixels are under this percentage "
        "of the band (default: %(default)s, i.e. any difference fails)",
    )
    args = parser.parse_args()

    if not 0 <= args.max_diff_pct <= 100:
        parser.error("--max-diff-pct must be between 0 and 100")

    with rasterio.Env(
        AWS_NO_SIGN_REQUEST=os.environ.get("AWS_NO_SIGN_REQUEST", "NO"),
        GDAL_HTTP_MAX_RETRY=os.environ.get("GDAL_HTTP_MAX_RETRY", "0"),
    ):
        differences, tolerated = compare(args.golden, args.candidate, args.max_diff_pct)

    for message in tolerated:
        print(f"  IGNORED {message}", file=sys.stderr)
    for difference in differences:
        print(f"  {difference}", file=sys.stderr)
    return 1 if differences else 0


if __name__ == "__main__":
    sys.exit(main())
