import xarray as xr


def apply_mapping(data, class_mapping):
    """
    Utility function to apply mapping on dictionaries
    """
    for o_val, n_val in class_mapping.items():
        data = xr.where(data == o_val, n_val, data)
    return data
