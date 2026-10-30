#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${DATA_ROOT:?Set DATA_ROOT to the dataset root}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ARCH="${ARCH:-ViT-B/32}"
DATASET="${DATASET:-DTD}"
EPS="${EPS:-4}"
STEPS="${STEPS:-100}"
MODE="${MODE:-probability}"
CLEAN_STRATEGY="${CLEAN_STRATEGY:-direct}"
NUM_SHARDS="${NUM_SHARDS:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output_results/ttp}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS:-0}"
[[ "$NUM_SHARDS" =~ ^[1-9][0-9]*$ && "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] || exit 2
pids=()
wait_batch() {
  local failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  pids=()
  return "$failed"
}
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  gpu="${GPU_LIST[$((shard % ${#GPU_LIST[@]}))]}"
  cmd=("$PYTHON_BIN" "$ROOT/main.py" "$DATA_ROOT" --test_sets "$DATASET"
       -a "$ARCH" --gpu 0 --eps "$EPS" --steps "$STEPS"
       --ensemble-mode "$MODE" --clean-strategy "$CLEAN_STRATEGY"
       --shard-id "$shard" --num-shards "$NUM_SHARDS" --output_dir "$OUTPUT_DIR")
  if [[ -n "${MAX_SAMPLES:-}" ]]; then cmd+=(--max-samples "$MAX_SAMPLES"); fi
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" &
  pids+=("$!")
  if (( ${#pids[@]} >= MAX_PARALLEL )); then wait_batch; fi
done
wait_batch
EPS_SLUG="$("$PYTHON_BIN" -c 'import sys; print(f"{float(sys.argv[1]):g}")' "$EPS")"
"$PYTHON_BIN" "$ROOT/scripts/aggregate_results.py" \
  --run-dir "$OUTPUT_DIR/$MODE/$CLEAN_STRATEGY/${ARCH//\//-}/$DATASET/eps_${EPS_SLUG}_steps_$STEPS" \
  --num-shards "$NUM_SHARDS"
