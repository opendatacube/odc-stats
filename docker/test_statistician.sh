#!/usr/bin/env bash
set -ex

cd ..
docker compose up -d --quiet-pull
sleep 5

docker compose exec -T stats odc-stats --version
echo "Indexing some data"
docker compose exec -e AWS_DEFAULT_REGION=ap-southeast-2 -T stats ./tests/init_db.sh
echo "Data regression test"
docker compose exec -T stats  ./tests/integration_test.sh
