#!/bin/bash
# Run the deliverable in a fresh Python 3.10 environment built only from requirements.txt, then compare with predictions.csv.
set -e
rm -rf /tmp/clean_env_check && mkdir -p /tmp/clean_env_check
python3.10 -m venv /tmp/clean_env_check/venv
/tmp/clean_env_check/venv/bin/pip install -q -r requirements.txt
/tmp/clean_env_check/venv/bin/pip freeze | grep -iE '^(torch|timm|opencv-python-headless|numpy)=='
cp inference_script.py /tmp/clean_env_check/
cd /tmp/clean_env_check
venv/bin/python inference_script.py --test_dir "$OLDPWD/data/dataset/hackathon_dataset/test/images" \
  --weights "$OLDPWD/runs/final/model_weights.pt" --output clean.csv
venv/bin/python - "$OLDPWD/runs/final/predictions.csv" <<'E'
import csv, sys
a = list(csv.DictReader(open(sys.argv[1])))
b = list(csv.DictReader(open("clean.csv")))
assert [r["image_id"] for r in a] == [r["image_id"] for r in b]
print("rows", len(b), "max confidence difference", max(abs(float(x["confidence"]) - float(y["confidence"])) for x, y in zip(a, b)),
      "label changes", sum(x["label"] != y["label"] for x, y in zip(a, b)))
E
cd "$OLDPWD"
rm -rf /tmp/clean_env_check
echo CLEAN_ENV_DONE
