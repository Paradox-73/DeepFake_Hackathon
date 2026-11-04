#!/bin/bash
# Retrain everything from the dataset: base models, label list, fold models, all-data models, predictions.
set -e
bash scripts/run_jobs.sh configs/jobs_base.txt 2 configs/exp.yaml
.venv/bin/python scripts/clean_labels.py --families cxd deit --thr 0.85 --mode flip --out configs/relabel_flip85.csv
bash scripts/run_jobs.sh configs/jobs_final_lr25.txt 2 configs/exp.yaml
bash scripts/run_jobs.sh configs/jobs_e22.txt 2 configs/exp.yaml
bash scripts/final_build.sh
