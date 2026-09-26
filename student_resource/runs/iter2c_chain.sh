#!/bin/bash
# iteration 2: transliteration dictionary + candidate-support features + rarest-token house key + stage B
R=/c/Users/LawLight/Desktop/tri-source-entity-resolution/student_resource
cd $R/code/business_entity_resolution

run() { python -u -m src.run_pipeline --config config_v2.yaml --stage $1 --split $2 --tag "$3" > $R/runs/v2_$1_$2$3.log 2>&1 || { echo "FAILED $1 $2 $3"; exit 1; }; echo "done $1 $2 $3 $(date +%H:%M)"; }
run features dev ""
run stack dev ""
run decide dev "v2-stageA"
run stackb dev ""
run decide dev "v2-stageB"
run block test ""
run features test ""
run stack test ""
run stackb test ""
run decide test "v2-stageB"
run submit test "v2"
echo "CHAIN COMPLETE $(date +%H:%M)"
