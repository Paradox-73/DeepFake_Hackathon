#!/bin/bash
# Screen heavy-shrink training on folds 0 and 1, then score old, new, and mixed ensembles under all 41 kinds of damage.
cd ~/synergy_deepfake
export OMP_NUM_THREADS=2 HF_HUB_DISABLE_PROGRESS_BARS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while read -r name args; do
  [ -z "$name" ] && continue
  [ -f "runs/$name/metrics.json" ] && { echo "skip $name"; continue; }
  mkdir -p "runs/$name"
  .venv/bin/python train.py --config configs/exp.yaml --set output_dir="runs/$name" $args > "runs/$name/stdout.log" 2>&1 < /dev/null
  echo "finished $name exit=$? $(date +%H:%M)"
  rm -f "runs/$name/last.pt"
done < configs/jobs_e20.txt
mkdir -p runs/e20
F="cfm dflipmix a_rn50 lrc10 lrd10 lrr10 lrc25 lrd25 lrr25"
.venv/bin/python scripts/ens_eval.py --families $F --folds 0 1 --out runs/e20/eval_default < /dev/null && echo "eval default done"
.venv/bin/python scripts/ens_eval.py --families $F --folds 0 1 --extra --out runs/e20/eval_extra < /dev/null && echo "eval extra done"
echo E20_DONE
