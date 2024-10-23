import xarray as xr


def lc_level3(xx: xr.Dataset, NODATA):

    l34_dss = xx.classes_l3_l4
    urban_dss = xx.urban_classes
    cultivated_dss = xx.cultivated_class

    # Map intertidal areas to water
    intertidal_mask = l34_dss == 223
    l34_dss = xr.where(intertidal_mask, 220, l34_dss)

    # Cultivated pipeline applies a mask which feeds only terrestrial veg (110) to the model
    # Just exclude no data (255) and apply the cultivated results
    cultivated_mask = cultivated_dss != int(NODATA)
    l34_cultivated_masked = xr.where(cultivated_mask, cultivated_dss, l34_dss)

    # Urban is classified on l3/4 surface output (210)
    urban_mask = l34_dss == 210
    l34_urban_cultivated_masked = xr.where(urban_mask, urban_dss, l34_cultivated_masked)

    # Replace nan with NODATA
    l34_urban_cultivated_masked = xr.where(
        l34_urban_cultivated_masked == l34_urban_cultivated_masked,
        l34_urban_cultivated_masked,
        NODATA,
    )

    return intertidal_mask, l34_urban_cultivated_masked
