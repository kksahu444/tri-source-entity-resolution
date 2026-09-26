#!/bin/bash
# iteration 1: finish dev decisions and run the test pipeline end to end
set -o pipefail
R=/c/Users/LawLight/Desktop/tri-source-entity-resolution/student_resource
cd $R/code/business_entity_resolution
until grep -qE "stage stack done|Traceback" $R/runs/stack_dev.log; do sleep 20; done
grep -q "Traceback" $R/runs/stack_dev.log && { echo "stack dev failed"; exit 1; }
run() { python -u -m src.run_pipeline --stage $1 --split $2 > $R/runs/$1_$2.log 2>&1 || { echo "FAILED $1 $2"; exit 1; }; echo "done $1 $2 $(date +%H:%M)"; }
run decide dev &
DEC=$!
run block test
wait $DEC || { echo "FAILED decide dev"; exit 1; }
run features test
run stack test
run loco dev
run decide test
run submit test
echo "CHAIN COMPLETE $(date +%H:%M)"
