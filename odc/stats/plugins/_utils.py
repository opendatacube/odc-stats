import dask
import fiona
from rasterio import features


def rasterize_vector_mask(shape_file, transform, dst_shape, threshold=None):
    with fiona.open(shape_file) as source_ds:
        geoms = [s["geometry"] for s in source_ds]

    mask = features.rasterize(
        geoms,
        transform=transform,
        out_shape=dst_shape[1:],
        all_touched=False,
        fill=0,
        default_value=1,
        dtype="uint8",
    )

    # if valid area >= threshold
    # then the whole tile is valid
    if threshold is not None:
        if mask.sum() > mask.size * threshold:
            return dask.array.ones(dst_shape, name=False)

    return dask.array.from_array(mask.reshape(dst_shape), name=False)
