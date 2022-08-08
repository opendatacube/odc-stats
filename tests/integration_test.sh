#!/usr/bin/env bash

echo "Test GeoMAD"

odc-stats save-tasks --config https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/master/dev/services/odc-stats/geomedian/ga_ls8c_nbart_gm_cyear_3.yaml --year=2015 --tiles 49:50,24:25 --overwrite geomad-cyear.db
odc-stats run  --threads=1 --config https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/master/dev/services/odc-stats/geomedian/ga_ls8c_nbart_gm_cyear_3.yaml --location file:///tmp --overwrite geomad-cyear.db

sha1sum -c --status ./tests/data/ga_ls8c_nbart_gm_cyear_3_x49y24_2015--P1Y_final.sha1

echo "Test WO summary"

odc-stats save-tasks --config https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/master/dev/services/odc-stats/wofs_summary/ga_ls_wo_fq_cyear_3.yaml --year=2015 --tiles 49:50,24:25 --overwrite wofs-cyear.db
odc-stats run  --threads=1 --config https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/master/dev/services/odc-stats/wofs_summary/ga_ls_wo_fq_cyear_3.yaml --location file:///tmp --overwrite wofs-cyear.db

sha1sum -c --status ./tests/data/ga_ls_wo_fq_cyear_3_x49y24_2015--P1Y_final.sha1

echo "Test FC percentile"

odc-stats save-tasks --config https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/master/dev/services/odc-stats/fc_percentile/ga_ls_fc_pc_cyear_3.yaml --year=2015 --tiles 49:50,24:25 --overwrite fcp-cyear.db
odc-stats run  --threads=1 --config https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/master/dev/services/odc-stats/fc_percentile/ga_ls_fc_pc_cyear_3.yaml --location file:///tmp --overwrite fcp-cyear.db

sha1sum -c --status ./tests/data/ga_ls_fc_pc_cyear_3_x49y24_2015--P1Y_final.sha1

echo "Test TC percentile"
odc-stats save-tasks --config https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/master/dev/services/odc-stats/tc_percentile/ga_ls_tc_pc_cyear_3.yaml --year=2015 --tiles 49:50,24:25 --overwrite tcp-cyear.db
odc-stats run  --threads=1 --config https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/master/dev/services/odc-stats/tc_percentile/ga_ls_tc_pc_cyear_3.yaml --location file:///tmp --overwrite tcp-cyear.db

sha1sum -c --status ./tests/data/ga_ls_tc_pc_cyear_3_x49y24_2015--P1Y_final.sha1

# save time without financial year
# test on calendar year seems enough
# not sure if they're needed save for future

# odc-stats save-tasks --config https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/master/dev/services/odc-stats/geomedian/ga_ls8c_nbart_gm_fyear_3.yaml --temporal-range 2014-07--P1Y --tiles 49:50,24:25 --overwrite geomad-fyear.db
# odc-stats run  --threads=1 --config https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/master/dev/services/odc-stats/geomedian/ga_ls8c_nbart_gm_fyear_3.yaml --location file:///tmp --overwrite geomad-fyear.db

# odc-stats save-tasks --config https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/master/dev/services/odc-stats/wofs_summary/ga_ls_wo_fq_fyear_3.yaml --temporal-range 2014-07--P1Y --tiles 49:50,24:25 --overwrite wofs-fyear.db
# odc-stats run  --threads=1 --config https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/master/dev/services/odc-stats/wofs_summary/ga_ls_wo_fq_fyear_3.yaml --location file:///tmp --overwrite wofs-fyear.db

# odc-stats save-tasks --config https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/master/dev/services/odc-stats/fc_percentile/ga_ls_fc_pc_fyear_3.yaml --temporal-range 2014-07--P1Y --tiles 49:50,24:25 --overwrite fcp-fyear.db
# odc-stats run  --threads=1 --config https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/master/dev/services/odc-stats/fc_percentile/ga_ls_fc_pc_fyear_3.yaml --location file:///tmp --overwrite fcp-fyear.db

# odc-stats save-tasks --config https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/master/dev/services/odc-stats/tc_percentile/ga_ls_tc_pc_fyear_3.yaml --temporal-range 2014-07--P1Y --tiles 49:50,24:25 --overwrite tcp-fyear.db
# odc-stats run  --threads=1 --config https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/master/dev/services/odc-stats/tc_percentile/ga_ls_tc_pc_fyear_3.yaml --location file:///tmp --overwrite tcp-fyear.db
