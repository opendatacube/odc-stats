#!/usr/bin/env bash

echo "Test GeoMAD"

odc-stats save-tasks --config https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/update_stats_configs/dev/services/odc-stats/geomedian/ga_ls8c_nbart_gm_cyear_3.yaml --year=2015 --tiles 49:50,23:24 --overwrite geomad-eo3-test-run.db
odc-stats run  --threads=1 --config https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/update_stats_configs/dev/services/odc-stats/geomedian/ga_ls8c_nbart_gm_cyear_3.yaml --location file:///tmp --overwrite geomad-eo3-test-run.db
ls -alt /tmp
