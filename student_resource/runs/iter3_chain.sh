#!/bin/bash
# iteration 3 (config_v3.yaml): + generator-aware features, anchor agreement, K-class count model,
# all-dev model for test inference; REBLOCK=1 also re-runs blocking (second-hop channel C6 enabled)
R=/c/Users/LawLight/Desktop/tri-source-entity-resolution/student_resource
cd $R/code/business_entity_resolution
until grep -qE "CHAIN COMPLETE|FAILED" $R/runs/iter2d_chain.log; do sleep 30; done
grep -q "FAILED" $R/runs/iter2d_chain.log && { echo "iteration 2 failed; not starting iteration 3"; exit 1; }
run() { python -u -m src.run_pipeline --config config_v3.yaml --stage $1 --split $2 --tag "$3" > $R/runs/v3_$1_$2$3.log 2>&1 || { echo "FAILED $1 $2 $3"; exit 1; }; echo "done $1 $2 $3 $(date +%H:%M)"; }
[ "$REBLOCK" = "1" ] && run block dev ""
run features dev ""
run stack dev ""
run decide dev "v3-stageA"
run stackb dev ""
run decide dev "v3-stageB"
[ "$REBLOCK" = "1" ] && run block test ""
run features test ""
run stack test ""
run stackb test ""
run decide test "v3-stageB"
run submit test "v3"
echo "CHAIN COMPLETE $(date +%H:%M)"
