#!/bin/bash
# iteration 2: transliteration dictionary + candidate-support features + rarest-token house key
R=/c/Users/LawLight/Desktop/tri-source-entity-resolution/student_resource
cd $R/code/business_entity_resolution
until grep -qE "CHAIN COMPLETE|FAILED|failed" $R/runs/iter1_chain.log; do sleep 30; done
run() { python -u -m src.run_pipeline --config config_v2.yaml --stage $1 --split $2 > $R/runs/v2_$1_$2.log 2>&1 || { echo "FAILED $1 $2"; exit 1; }; echo "done $1 $2 $(date +%H:%M)"; }
run translit dev
run block dev
run features dev
run stack dev
run decide dev
run block test
run features test
run stack test
run loco dev
run decide test
run submit test
echo "CHAIN COMPLETE $(date +%H:%M)"
