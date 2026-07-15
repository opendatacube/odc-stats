import click
import re
import tempfile
import pandas as pd
from os import path
from io import StringIO
import geopandas as gpd
from ._cli_common import main


def locate_grids(grid_shape, extent_shape, attr_filter=None):
    grids = gpd.read_file(grid_shape)

    # Read extent layer, optionally applying an OGR-style attribute filter
    # e.g. attr_filter = "type = 'foo'"
    if attr_filter is not None:
        extents = gpd.read_file(extent_shape, where=attr_filter)
    else:
        extents = gpd.read_file(extent_shape)

    if extents.empty:
        return []

    # Match CRS.
    if grids.crs is not None and extents.crs is not None and grids.crs != extents.crs:
        # We transform the grid as it can be in geographic projection with antemeridian issues
        grids = grids.to_crs(extents.crs)

    extent_geom = extents.geometry.union_all()
    matched_grids = grids[grids.geometry.intersects(extent_geom)]

    return [
        re.findall(r"\d+", str(region_code))
        for region_code in matched_grids["region_code"]
    ]


@main.command("locate-grids")
@click.option(
    "--attr-filter",
    type=str,
    default=None,
    help="Filter the input shape by attributes, e.g., FEAT_CODE != 'sea'",
)
@click.argument("grid-shape", type=str)
@click.argument("extent-shape", type=str)
@click.argument("csv-path", type=str, required=False)
@click.option("--verbose", "-v", is_flag=True, help="Be verbose")
def cli(attr_filter, grid_shape, extent_shape, csv_path, verbose):
    """
    Generate a list of grids overlapping with the input shape extent

    GRID_SHAPE is the geojson or ESRI shape file of the grids.

    EXTENT_SHAPE is the ESRI shape file where the extent covers the grids.

    CSV_PATH is the path where the csv of the grids list will be saved, default is None.
    By default, the file will be saved in the system temporary folder.

    """

    print("It takes time, not frozen...")
    print(f"Input shape {extent_shape} filtered by {attr_filter}")
    extent_grids = locate_grids(grid_shape, extent_shape, attr_filter)
    csv_buffer = StringIO()
    pd.DataFrame(extent_grids).to_csv(csv_buffer, index=None, header=None)
    csv_buffer.seek(0)

    if csv_path is None:
        tmp_path = tempfile.gettempdir()
        csv_path = path.join(tmp_path, "extent_grids.csv")
    with open(csv_path, "w", encoding="utf8") as f:
        f.write(csv_buffer.read())
    print("Results saved to", csv_path)


if __name__ == "__main__":
    cli()
