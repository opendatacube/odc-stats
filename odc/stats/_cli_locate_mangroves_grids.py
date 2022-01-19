import click
import itertools
import re
import tempfile
import pandas as pd
from os import path
from io import StringIO
from osgeo import gdal, ogr, osr
from ._cli_common import main

def locate_mangroves_grids(grid_shape, mangroves_shape):
    ds_grid = ogr.Open(grid_shape)
    lyr_grid = ds_grid.GetLayer(0)
    lyr_grid.ResetReading()
    grid_spatial_ref = lyr_grid.GetSpatialRef()
    ds_mangroves = ogr.Open(mangroves_shape)
    lyr_mangroves = ds_mangroves.GetLayer(0)
    lyr_mangroves.ResetReading()
    mangroves_spatial_ref = lyr_mangroves.GetSpatialRef()
    transform = osr.CoordinateTransformation(grid_spatial_ref, mangroves_spatial_ref)
    mangroves_grids = []
    for grid in lyr_grid:
        grid_geom = grid.geometry()
        grid_geom.Transform(transform)
        lyr_mangroves.SetSpatialFilter(grid_geom)
        if lyr_mangroves.GetFeatureCount() > 0:
            mangroves_grids += [re.findall(r'\d+', grid['region_code'])]
        lyr_mangroves.SetSpatialFilter(None)
        lyr_mangroves.ResetReading()
    return mangroves_grids

@main.command("mangroves-grids")
@click.argument("grid-shape", type=str)
@click.argument("mangroves-shape", type=str)
@click.argument("csv-path", type=str, required=False)
@click.option("--verbose", "-v", is_flag=True, help="Be verbose")
def cli(grid_shape, mangroves_shape, csv_path, verbose):
    """
    Generate a list of grids overlapping with mangroves
    
    GRID_SHAPE is the geojson or ESRI shape file of the grids.
    
    MANGROVES_SHAPE is the ESRI shape file where the maximum mangroves are is defined.
    
    CSV_PATH is the path where the csv of the grids list will be saved, default is None.
    By default, the file will be saved in the system temporary folder.
    
    """

    print("It takes time, not frozen...")
    mangroves_grids = locate_mangroves_grids(grid_shape, mangroves_shape)
    csv_buffer = StringIO()
    pd.DataFrame(mangroves_grids).to_csv(csv_buffer, index=None, header=None)
    csv_buffer.seek(0)
    
    if csv_path is None:
        tmp_path = tempfile.gettempdir()
        csv_path = path.join(tmp_path, "mangroves_grids.csv")
    with open(csv_path, 'w') as f:
        f.write(csv_buffer.read())
    print("Results saved to", csv_path)

if __name__ == "__main__":
    cli()
