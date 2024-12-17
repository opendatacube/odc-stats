import re
import operator
import numpy as np
import dask
from osgeo import gdal, ogr, osr
from functools import partial


def rasterize_vector_mask(
    shape_file, transform, dst_shape, filter_expression=None, threshold=None
):
    source_ds = ogr.Open(shape_file)
    source_layer = source_ds.GetLayer()

    if filter_expression is not None:
        source_layer.SetAttributeFilter(filter_expression)

    yt, xt = dst_shape[1:]
    no_data = 0
    albers = osr.SpatialReference()
    albers.ImportFromEPSG(3577)

    geotransform = (
        transform.c,
        transform.a,
        transform.b,
        transform.f,
        transform.d,
        transform.e,
    )
    target_ds = gdal.GetDriverByName("MEM").Create("", xt, yt, gdal.GDT_Byte)
    target_ds.SetGeoTransform(geotransform)
    target_ds.SetProjection(albers.ExportToWkt())
    mask = target_ds.GetRasterBand(1)
    mask.SetNoDataValue(no_data)
    gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[1])

    mask = mask.ReadAsArray()

    # used by landcover level3 urban
    # if valid area >= threshold
    # then the whole tile is valid

    if threshold is not None:
        if mask.sum() > mask.size * threshold:
            return dask.array.ones(dst_shape, name=False)

    return dask.array.from_array(mask.reshape(dst_shape), name=False)


OPERATORS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}

BRACKETS = {
    "[": operator.ge,  # Inclusive lower bound
    "(": operator.gt,  # Exclusive lower bound
    "]": operator.le,  # Inclusive upper bound
    ")": operator.lt,  # Exclusive upper bound
}


def parse_rule(rule):
    """
    Parse a single condition or range condition.
    Supports range notations like '[]', '[)', '(]', and '()',
    and treats standalone numbers as '=='.
    """
    # Special case for 255 (rule doesn't apply)
    if (rule == "255") | (rule == "nan"):
        return None

    # Check for range conditions like '[a, b)' or '(a, b]'
    range_pattern = r"([\[\(])(-?\d+\.?\d*),\s*(-?\d+\.?\d*)([\]\)])"
    match = re.match(range_pattern, rule)
    if match:
        # Extract the bounds and the bracket types
        lower_bracket, lower_value, upper_value, upper_bracket = match.groups()
        return [
            (BRACKETS[lower_bracket], float(lower_value)),
            (BRACKETS[upper_bracket], float(upper_value)),
        ]

    ordered_operators = sorted(OPERATORS.items(), key=lambda x: -len(x[0]))

    # Single condition (no range notation, no explicit operator)
    for op_str, op_func in ordered_operators:
        if op_str in rule:
            value = float(rule.replace(op_str, "").strip())
            return [(op_func, value)]

    # Default to equality (==) if no operator is found
    return [(operator.eq, int(rule.strip()))]


def generate_numexpr_expressions(rules_df, final_class_column, previous):
    """
    Generate a list of numexpr-compatible expressions for classification rules.
    :param rules_df: DataFrame containing the classification rules
    :param final_class_column: Name of the column containing the final class values
    :return: List of expressions (one for each rule)
    """
    expressions = []

    for _, rules in rules_df.iterrows():
        conditions = []

        for col in rules.index:
            if col == final_class_column:
                continue
            subconditions = parse_rule(rules[col])
            if subconditions is None:  # Skip rule if it's None
                continue
            for op_func, value in subconditions:
                if op_func is operator.eq:
                    conditions.append(f"({col}=={value})")
                elif op_func is operator.gt:
                    conditions.append(f"({col}>{value})")
                elif op_func is operator.ge:
                    conditions.append(f"({col}>={value})")
                elif op_func is operator.lt:
                    conditions.append(f"({col}<{value})")
                elif op_func is operator.le:
                    conditions.append(f"({col}<={value})")
                elif op_func is operator.ne:
                    conditions.append(f"({col}!={value})")

        if not conditions:
            continue

        condition = "&".join(conditions)

        final_class = rules[final_class_column]
        expressions.append(f"where({condition}, {final_class}, {previous})")

    expressions = list(set(expressions))
    expressions = sorted(expressions, key=len)

    return expressions


def numpy_mode_exclude_nodata(values, target_value, exclude_values):
    """
    Compute the mode of an array using NumPy, excluding nodata.
    :param values: A flattened 1D array representing the neighborhood.
    :param target_value: The value to be replaced
    :param exclude_values: A list or set of values to exclude from the mode calculation.
    :return: The mode of the array (smallest value in case of ties), excluding nodata.
    """

    valid_mask = ~(
        np.isin(values, list(set(exclude_values) | {target_value})) | np.isnan(values)
    )
    valid_values = values[valid_mask]
    if len(valid_values) == 0:
        return target_value
    unique_vals, counts = np.unique(valid_values, return_counts=True)
    max_count = counts.max()
    # select the smallest value among ties
    mode_value = unique_vals[counts == max_count].min()
    return mode_value


def process_nodata_pixels(block, target_value, exclude_values, max_radius):
    """
    Replace nodata pixels in a block with the mode of their 3x3 neighborhood.
    :param block : numpy.ndarray The 2D array chunk.
    :param target_value: The value to be replaced
    :param exclude_values: A list or set of values to exclude from the mode calculation.
    :param max_radius: maximum size of neighbourhood
    :return: numpy.ndarray The modified block where nodata pixels are replaced.
    """
    result = block.copy()
    nodata_indices = np.argwhere(block == target_value)

    for i, j in nodata_indices:
        # start from the smallest/nearest neighbourhood
        # stop once finding the valid value otherwise expand till the max_radius
        for radius in range(1, max_radius + 1):
            i_min, i_max = max(0, i - radius), min(block.shape[0], i + radius + 1)
            j_min, j_max = max(0, j - radius), min(block.shape[1], j + radius + 1)

            neighborhood = block[i_min:i_max, j_min:j_max].flatten()
            tmp = numpy_mode_exclude_nodata(neighborhood, target_value, exclude_values)
            if np.isnan(tmp) | (tmp == target_value):
                continue
            result[i, j] = tmp
            break

    return result


def replace_nodata_with_mode(
    arr, target_value, exclude_values=None, neighbourhood_size=3
):
    """
    Replace nodata-valued pixels in a Dask array with the mode of their neighborhood,
    processing only the nodata pixels.
    :param arr: A 2D Dask array.
    :param target_value: The value to be replaced
    :param exclude_values: A list or set of values to exclude from the mode calculation.
    :param neighbourhood_size: the size of neighbourhood, e.g., 3:= 3*3 block, 5:=5*5 block
    :return: A Dask array where nodata-valued pixels have been replaced.
    """
    if exclude_values is None:
        exclude_values = set()

    radius = neighbourhood_size // 2
    process_func = partial(
        process_nodata_pixels,
        target_value=target_value,
        exclude_values=exclude_values,
        max_radius=radius,
    )
    # Use map_overlap to handle edges and target only the nodata pixels
    result = arr.map_overlap(
        process_func,
        depth=(radius, radius),
        boundary="nearest",
        dtype=arr.dtype,
        trim=True,
    )
    return result
