#!/bin/bash
# After the all-data build: check the rerun, score the fold version on the test images, export to TensorFlow, fresh-env check.
cd ~/synergy_deepfake
export OMP_NUM_THREADS=4
until grep -q E22_DONE runs/jobs_e22.log; do sleep 60; done
cmp runs/final/predictions.csv runs/final/predictions_rerun.csv && echo "rerun byte-identical" || echo "RERUN DIFFERS"
fold=(); for f in lrc25 lrd25 lrr25; do for k in 0 1 2 3 4; do fold+=("runs/${f}_f$k"); done; done
.venv/bin/python scripts/build_bundle.py --runs "${fold[@]}" --out runs/final/fold_bundle.pt > runs/final/fold_bundle.log 2>&1 \
  && .venv/bin/python inference_script.py --test_dir data/dataset/hackathon_dataset/test/images --weights runs/final/fold_bundle.pt \
     --output runs/final/fold_predictions.csv >> runs/final/fold_bundle.log 2>&1 && echo "fold predictions done"
rm -f runs/final/fold_bundle.pt
.venv/bin/python scripts/export_tf_ensemble.py dump --weights runs/final/model_weights.pt --images data/dataset/hackathon_dataset/test/images \
  --out runs/final/tf_dump.npz > runs/final/export_tf.log 2>&1 \
  && .venv_tf/bin/python scripts/export_tf_ensemble.py build --dump runs/final/tf_dump.npz --out runs/final/tf_savedmodel >> runs/final/export_tf.log 2>&1 \
  && echo "tf export done"
.venv/bin/python - <<'E' && echo "tf zip done"
import zipfile
from pathlib import Path
src = Path("runs/final/tf_savedmodel")
with zipfile.ZipFile("runs/final/tf_savedmodel.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(src.rglob("*")):
        info = zipfile.ZipInfo(p.relative_to(src).as_posix() + ("/" if p.is_dir() else ""), (1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = (0o755 if p.is_dir() else 0o644) << 16
        z.writestr(info, b"" if p.is_dir() else p.read_bytes())
E
rm -f runs/final/tf_dump.npz
bash scripts/clean_env_check.sh > runs/final/clean_env.log 2>&1 && echo "clean env done"
echo AFTER_DONE
