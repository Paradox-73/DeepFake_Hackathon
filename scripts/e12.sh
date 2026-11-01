#!/bin/bash
# Leak-free scores on every outer fold, then the test of retraining on all the data.
for r in cx_f1 cx_f2 cx_f3 cx_f4 d_f1 d_f2 d_f3 d_f4; do mv "runs/nest_$r" "runs/nest0_$r"; done
mv runs/nflipmix_f0 runs/nd_f0
base="data.fold=0 data.relabel_csv=configs/relabel_nest0.csv train.mixup=0.4"
{ echo "ncx_f0 $base"; echo "nr_f0 $base model.arch=resnet50d.ra2_in1k model.drop_path_rate=0.1"; } > configs/jobs_nestfinal0.txt
scripts/run_jobs.sh configs/jobs_nestfinal0.txt 2 configs/exp.yaml
scripts/run_jobs.sh configs/jobs_bag.txt 2 configs/exp.yaml
echo BAG_RUNS_DONE
for k in 1 2 3 4; do scripts/nested_check.sh "$k"; done
.venv/bin/python scripts/compare.py --base deit nd dflipmix
.venv/bin/python scripts/compare.py --base cxd ncx cfm
.venv/bin/python scripts/oof.py --families ncx nd nr
echo E12_DONE
