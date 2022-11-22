import click
import re
import tempfile
import pandas as pd
from os import path
from io import StringIO
from osgeo import ogr, osr
from ._cli_common import main


def locate_grids(grid_shape, extent_shape, attr_filter):
    ds_grid = ogr.Open(grid_shape)
    lyr_grid = ds_grid.GetLayer(0)
    lyr_grid.ResetReading()
    grid_spatial_ref = lyr_grid.GetSpatialRef()

    ds_extent = ogr.Open(extent_shape)
    lyr_extent = ds_extent.GetLayer(0)
    if attr_filter is not None:
        lyr_extent.SetAttributeFilter(attr_filter)
    lyr_extent.ResetReading()

    extent_spatial_ref = lyr_extent.GetSpatialRef()
    transform = osr.CoordinateTransformation(grid_spatial_ref, extent_spatial_ref)
    extent_grids = []
    for grid in lyr_grid:
        grid_geom = grid.geometry()
        grid_geom.Transform(transform)
        lyr_extent.SetSpatialFilter(grid_geom)
        if lyr_extent.GetFeatureCount() > 0:
            extent_grids += [re.findall(r"\d+", grid["region_code"])]
        lyr_extent.SetSpatialFilter(None)
        lyr_extent.ResetReading()
    return extent_grids


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
