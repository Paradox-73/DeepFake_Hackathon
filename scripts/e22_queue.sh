#!/bin/bash
# Train the 15 all-data models with the heavy-shrink setting (25%), then build the bundle and predictions twice.
cd ~/synergy_deepfake
export OMP_NUM_THREADS=2 HF_HUB_DISABLE_PROGRESS_BARS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while read -r name args; do
  [ -z "$name" ] && continue
  [ -f "runs/$name/metrics.json" ] && { echo "skip $name"; continue; }
  mkdir -p "runs/$name"
  .venv/bin/python train.py --config configs/exp.yaml --set output_dir="runs/$name" $args > "runs/$name/stdout.log" 2>&1 < /dev/null
  echo "finished $name exit=$? $(date +%H:%M)"
done < configs/jobs_e22.txt
[ -d runs/final ] && [ ! -d runs/final_e15 ] && mv runs/final runs/final_e15
bash scripts/final_build.sh < /dev/null && echo "build done"
.venv/bin/python inference_script.py --test_dir data/dataset/hackathon_dataset/test/images \
  --weights runs/final/model_weights.pt --output runs/final/predictions_rerun.csv < /dev/null && echo "rerun done"
echo E22_DONE
