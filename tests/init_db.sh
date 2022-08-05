#!/usr/bin/env bash      
datacube system init
datacube system check

# index data for integration test

# GeoMAD
datacube metadata add https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/master/product_metadata/eo3_landsat_ard.odc-type.yaml
datacube product add https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/master/products/baseline_satellite_data/c3/ga_ls8c_ard_3.odc-product.yaml
# only index several data to speed up yearly summary run
s3-to-dc "s3://dea-public-data/baseline/ga_ls8c_ard_3/088/079/2015/02/*/*.json" --no-sign-request --skip-lineage --stac ga_ls8c_ard_3
s3-to-dc "s3://dea-public-data/baseline/ga_ls8c_ard_3/088/079/2015/03/*/*.json" --no-sign-request --skip-lineage --stac ga_ls8c_ard_3
s3-to-dc "s3://dea-public-data/baseline/ga_ls8c_ard_3/088/079/2015/04/*/*.json" --no-sign-request --skip-lineage --stac ga_ls8c_ard_3
s3-to-dc "s3://dea-public-data/baseline/ga_ls8c_ard_3/088/079/2015/05/*/*.json" --no-sign-request --skip-lineage --stac ga_ls8c_ard_3

# WO
datacube product add https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/master/products/inland_water/c3_wo/ga_ls_wo_3.odc-product.yaml

# only index several data to speed up yearly summary run
s3-to-dc "s3://dea-public-data/derivative/ga_ls_wo_3/1-6-0/088/079/2015/02/*/*.json" --no-sign-request --skip-lineage --stac ga_ls_wo_3
s3-to-dc "s3://dea-public-data/derivative/ga_ls_wo_3/1-6-0/088/079/2015/03/*/*.json" --no-sign-request --skip-lineage --stac ga_ls_wo_3
s3-to-dc "s3://dea-public-data/derivative/ga_ls_wo_3/1-6-0/088/079/2015/04/*/*.json" --no-sign-request --skip-lineage --stac ga_ls_wo_3
s3-to-dc "s3://dea-public-data/derivative/ga_ls_wo_3/1-6-0/088/079/2015/05/*/*.json" --no-sign-request --skip-lineage --stac ga_ls_wo_3

# FC 
datacube product add https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/master/products/land_and_vegetation/c3_fc/ga_ls_fc_3.odc-product.yaml

# only index several data to speed up yearly summary run
s3-to-dc "s3://dea-public-data/derivative/ga_ls_fc_3/2-5-1/088/079/2015/02/*/*.json" --no-sign-request --skip-lineage --stac ga_ls_fc_3
s3-to-dc "s3://dea-public-data/derivative/ga_ls_fc_3/2-5-1/088/079/2015/03/*/*.json" --no-sign-request --skip-lineage --stac ga_ls_fc_3
s3-to-dc "s3://dea-public-data/derivative/ga_ls_fc_3/2-5-1/088/079/2015/04/*/*.json" --no-sign-request --skip-lineage --stac ga_ls_fc_3
s3-to-dc "s3://dea-public-data/derivative/ga_ls_fc_3/2-5-1/088/079/2015/05/*/*.json" --no-sign-request --skip-lineage --stac ga_ls_fc_3
