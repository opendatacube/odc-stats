import click
import itertools
from ._cli_common import main

from odc.dscache import create_cache
from ._stac_fetch import s3_fetch_dss


@main.command("generate-cache")
@click.argument("input_glob", type=str)
@click.argument("location", type=str)
def cli(input_glob, location):
    """
    Search AWS S3 for STAC Datasets matching INPUT_GLOB and cache them in a local file in LOCATION directory.

    The output file will be named LOCATION/{product_name}.db, where {product_name} is determined by the first
    dataset found.

    Note: The input bucket must be public otherwise the data can not be listed.

    \b
    Example:
        odc-stats generate-cache s3://my-public-bucket/stats/output-01/*.json local_cache.db

    """

    # Look ahead to get product
    dss = s3_fetch_dss(input_glob)
    ds0 = next(dss)
    product = ds0.type
    dss = itertools.chain(iter([ds0]), dss)

    print(f"Writing {location}/{product.name}.db")

    cache = create_cache(f"{location}/{product.name}.db")
    cache.bulk_save(dss)
    print(f"Found {cache.count:,d} datasets")


if __name__ == "__main__":
    cli()
