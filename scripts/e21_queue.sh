#!/bin/bash
# Fair check of heavy-shrink training: score against fresh second-seed runs of the shipped recipe (s1c, s1d, s1r),
# first on folds 0 and 1, then train folds 2-4 and score all 5 folds.
cd ~/synergy_deepfake
export OMP_NUM_THREADS=2 HF_HUB_DISABLE_PROGRESS_BARS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs/e21
F="cfm dflipmix a_rn50 s1c s1d s1r lrc10 lrd10 lrr10 lrc25 lrd25 lrr25"
.venv/bin/python scripts/ens_eval.py --families $F --folds 0 1 --out runs/e21/eval01_default < /dev/null && echo "eval01 default done"
.venv/bin/python scripts/ens_eval.py --families $F --folds 0 1 --extra --out runs/e21/eval01_extra < /dev/null && echo "eval01 extra done"
while read -r name args; do
  [ -z "$name" ] && continue
  [ -f "runs/$name/metrics.json" ] && { echo "skip $name"; continue; }
  mkdir -p "runs/$name"
  .venv/bin/python train.py --config configs/exp.yaml --set output_dir="runs/$name" $args > "runs/$name/stdout.log" 2>&1 < /dev/null
  echo "finished $name exit=$? $(date +%H:%M)"
  rm -f "runs/$name/last.pt"
done < configs/jobs_e21.txt
.venv/bin/python scripts/ens_eval.py --families $F --out runs/e21/eval5_default < /dev/null && echo "eval5 default done"
.venv/bin/python scripts/ens_eval.py --families $F --extra --out runs/e21/eval5_extra < /dev/null && echo "eval5 extra done"
echo E21_DONE
