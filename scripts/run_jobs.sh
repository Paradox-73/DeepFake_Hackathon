#!/bin/bash
# Run a job file in parallel on the server. Each line: <run name> <config overrides...>
# Usage: nohup scripts/run_jobs.sh configs/jobs_e1.txt 3 configs/exp.yaml > runs/jobs_e1.log 2>&1 &
jobs_file=$1; parallel=${2:-3}; config=${3:-configs/exp.yaml}
export OMP_NUM_THREADS=2 HF_HUB_DISABLE_PROGRESS_BARS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run_one() {
  name=$1; shift
  out=runs/$name
  mkdir -p "$out"
  [ -f "$out/metrics.json" ] && { echo "skip $name (done)"; return; }
  nice -n 10 .venv/bin/python train.py --config "$config" --set output_dir="$out" "$@" > "$out/stdout.log" 2>&1 \
    && nice -n 10 .venv/bin/python robust_eval.py --config "$config" --ckpt "$out/best.pt" --set output_dir="$out" "$@" >> "$out/stdout.log" 2>&1
  status=$?
  rm -f "$out/last.pt"  # last.pt is 300 MB per run and the shared server disk is 93% full
  echo "finished $name exit=$status"
}
export -f run_one
export config
grep -v '^\s*$' "$jobs_file" | xargs -P "$parallel" -I{} bash -c 'run_one {}'
echo ALL_DONE
