#!/bin/bash
# Report material for the heavy-shrink families while the all-data models train: damage tables and Grad-CAM figures.
cd ~/synergy_deepfake
export OMP_NUM_THREADS=2
mkdir -p runs/e22
F="lrc25 lrd25 lrr25"
.venv/bin/python scripts/ens_eval.py --families $F --workers 4 --out runs/e22/ens_eval < /dev/null > runs/e22/ens_eval.log 2>&1 && echo "eval default done"
.venv/bin/python scripts/ens_eval.py --families $F --workers 4 --extra --out runs/e22/ens_eval_extra < /dev/null > runs/e22/ens_eval_extra.log 2>&1 && echo "eval extra done"
.venv/bin/python scripts/gradcam.py --run runs/lrc25_f0 --n 12 --out runs/e22/gradcam.jpg < /dev/null > runs/e22/gradcam.log 2>&1 && echo "gradcam done"
.venv/bin/python scripts/gradcam.py --run runs/lrc25_f0 runs/lrd25_f0 runs/lrr25_f0 --n 4 --out runs/e22/gradcam_families.jpg < /dev/null >> runs/e22/gradcam.log 2>&1 && echo "gradcam families done"
echo SIDE_DONE
