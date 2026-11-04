#!/bin/bash
# Pack the 15 all-data runs into runs/final/model_weights.pt and write predictions.csv on the test images.
# All-data runs have no unseen images, so their score statistics and the threshold come from the fold runs.
set -e
mkdir -p runs/final
runs=()
for family in lowc lowd lowr; do
  for seed in 42 1 2 3 4; do runs+=("runs/${family}_s$seed"); done
done
.venv/bin/python scripts/build_bundle.py --runs "${runs[@]}" --out runs/final/model_weights.pt \
  --stats_from lowc=lrc25 lowd=lrd25 lowr=lrr25
.venv/bin/python inference_script.py --test_dir data/dataset/hackathon_dataset/test/images \
  --weights runs/final/model_weights.pt --output runs/final/predictions.csv
