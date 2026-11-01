#!/bin/bash
# Leak-free score for one outer fold. configs/relabel_flip85.csv was built by models that saw every fold, so a fold's
# relabelled training set carries a trace of its own validation labels. Here the outer fold is removed from everything:
# 4-fold models on the rest build a new flip list, then all three final families train on that list.
# Usage: scripts/nested_check.sh <outer fold>
k=$1
deit="model.arch=deit3_small_patch16_224.fb_in22k_ft_in1k model.drop_path_rate=0.1 model.drop_rate=0.1"
rn="model.arch=resnet50d.ra2_in1k model.drop_path_rate=0.1"
n=$(awk -F, -v k="$k" 'NR>1 && $4+0==k' configs/folds.csv | wc -l)  # +0 because folds.csv has CRLF line ends
awk -F, -v k="$k" 'NR==1 || $4+0!=k' configs/folds.csv > "configs/folds_nest$k.csv"
[ "$n" -gt 1000 ] && [ "$(wc -l < "configs/folds_nest$k.csv")" -eq $((7001 - n)) ] || { echo "fold $k was not removed"; exit 1; }
: > "configs/jobs_nest$k.txt"
for j in 0 1 2 3 4; do
  [ "$j" = "$k" ] && continue
  echo "nest${k}_cx_f$j data.fold=$j data.folds_csv=configs/folds_nest$k.csv" >> "configs/jobs_nest$k.txt"
  echo "nest${k}_d_f$j data.fold=$j data.folds_csv=configs/folds_nest$k.csv $deit" >> "configs/jobs_nest$k.txt"
done
scripts/run_jobs.sh "configs/jobs_nest$k.txt" 2 configs/exp.yaml
.venv/bin/python scripts/clean_labels.py --families "nest${k}_cx" "nest${k}_d" --n_folds 4 --thr 0.85 --mode flip --out "configs/relabel_nest$k.csv"
base="data.fold=$k data.relabel_csv=configs/relabel_nest$k.csv train.mixup=0.4"
{ echo "ncx_f$k $base"; echo "nd_f$k $base $deit"; echo "nr_f$k $base $rn"; } > "configs/jobs_nestfinal$k.txt"
scripts/run_jobs.sh "configs/jobs_nestfinal$k.txt" 2 configs/exp.yaml
echo "NESTED_FOLD_${k}_DONE"
