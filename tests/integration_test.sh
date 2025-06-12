#!/usr/bin/env bash

set -e
set -o pipefail

odc-stats --version
# echo "Checking save tasks"
# odc-stats save-tasks --grid africa-20 --year 2019 --overwrite --input-products s2_l2a test-run.db
# echo "Checking a job run"
# odc-stats run  --threads=1 --plugin pq --location file:///tmp ./test-run.db 0

echo "Test LS GeoMAD"

odc-stats save-tasks --config https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/709daaee176c04e33de4cc9600462717cca5b34d/dev/services/odc-stats/geomedian/ga_ls8c_nbart_gm_cyear_3.yaml --year=2015 --tiles 49:50,24:25 --overwrite ls-geomad-cyear.db
odc-stats run  --threads=1 --config https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/709daaee176c04e33de4cc9600462717cca5b34d/dev/services/odc-stats/geomedian/ga_ls8c_nbart_gm_cyear_3.yaml --location file:///tmp --overwrite ls-geomad-cyear.db

./tests/compare_data.sh /tmp/x49/y24/ ga_ls8c_nbart_gm_cyear_3_x49y24_2015--P1Y_final*.tif

echo "Test LS WO summary"

odc-stats save-tasks --config https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/709daaee176c04e33de4cc9600462717cca5b34d/dev/services/odc-stats/wofs_summary/ga_ls_wo_fq_cyear_3.yaml --year=2015 --tiles 49:50,24:25 --overwrite ls-wofs-cyear.db
odc-stats run  --threads=1 --config https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/709daaee176c04e33de4cc9600462717cca5b34d/dev/services/odc-stats/wofs_summary/ga_ls_wo_fq_cyear_3.yaml --location file:///tmp --overwrite ls-wofs-cyear.db

./tests/compare_data.sh /tmp/x49/y24/ ga_ls_wo_fq_cyear_3_x49y24_2015--P1Y_fina*.tif

echo "Test LS FC percentile"

odc-stats save-tasks --config https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/709daaee176c04e33de4cc9600462717cca5b34d/dev/services/odc-stats/fc_percentile/ga_ls_fc_pc_cyear_3.yaml --year=2015 --tiles 49:50,24:25 --overwrite ls-fcp-cyear.db
odc-stats run  --threads=1 --config https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/709daaee176c04e33de4cc9600462717cca5b34d/dev/services/odc-stats/fc_percentile/ga_ls_fc_pc_cyear_3.yaml --location file:///tmp --overwrite ls-fcp-cyear.db

./tests/compare_data.sh /tmp/x49/y24/ ga_ls_fc_pc_cyear_3_x49y24_2015--P1Y_final*.tif

echo "Test LS TC percentile"
odc-stats save-tasks --config https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/709daaee176c04e33de4cc9600462717cca5b34d/dev/services/odc-stats/tc_percentile/ga_ls_tc_pc_cyear_3.yaml --year=2015 --tiles 49:50,24:25 --overwrite ls-tcp-cyear.db
odc-stats run  --threads=1 --config https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/709daaee176c04e33de4cc9600462717cca5b34d/dev/services/odc-stats/tc_percentile/ga_ls_tc_pc_cyear_3.yaml --location file:///tmp --overwrite ls-tcp-cyear.db

./tests/compare_data.sh /tmp/x49/y24/ ga_ls_tc_pc_cyear_3_x49y24_2015--P1Y_final*.tif

# echo "Test S2 GeoMAD"
# # use au-30 to save cost
# odc-stats save-tasks --config https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/feature/add-S2ab-GM-processing-cfg/dev/services/odc-stats/geomedian/ga_s2ab_gm_4fyear_3.yaml --input-products ga_s2am_ard_3 --grid au-30 --year=2020 --tiles 43:44,15:16 --overwrite s2-geomad-cyear.db
# odc-stats run  --threads=1 --config https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/feature/add-S2ab-GM-processing-cfg/dev/services/odc-stats/geomedian/ga_s2ab_gm_4fyear_3.yaml --location file:///tmp --overwrite s2-geomad-cyear.db
# 
# ./tests/compare_data.sh /tmp/x43/y15/ ga_s2ab_gm_4fyear_3_x43y15_2020--P1Y_final*.tif

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
