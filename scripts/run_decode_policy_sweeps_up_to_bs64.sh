#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VENV_PATH="${VENV_PATH:-$HOME/projects/sglang/.venv/bin/activate}"
OUTPUT_DIR="${OUTPUT_DIR:-./sparkinfer_decode_policy_sweeps}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-8}"
REPLAYS="${REPLAYS:-100}"
PROBE_BATCH_REPLAYS="${PROBE_BATCH_REPLAYS:-10}"
CI_LEVEL="${CI_LEVEL:-0.99}"
PAGE_START="${PAGE_START:-1}"
PAGE_STOP="${PAGE_STOP:-4096}"
CAPTURE_PAGE_COUNT="${CAPTURE_PAGE_COUNT:-4096}"
CANDIDATE_CTAS_PER_SM="${CANDIDATE_CTAS_PER_SM:-1,2,3}"
CANDIDATE_SPLITS="${CANDIDATE_SPLITS:-1,512}"
BATCH_SIZES_CSV="${BATCH_SIZES_CSV:-2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,36,40,42,44,48,50,52,54,56,60,64,66,70,72,78,80,84,88,90,96,100,104,108,110,112,120,128,130,132,140,144,150,156,160,168,176,180,192,200,208,220,224,240,256,260,264,280,288,300,312,320,336,360,384}"

DTYPES=(
  "bf16"
  "fp8_e4m3fn"
)

# Comma-separated capture batch sizes, expanded into a proper bash array below.
IFS=',' read -r -a BATCH_SIZES <<< "$BATCH_SIZES_CSV"

if [[ "${#BATCH_SIZES[@]}" -eq 0 ]]; then
  echo "BATCH_SIZES_CSV must contain at least one batch size" >&2
  exit 1
fi

if [[ ! -f "$VENV_PATH" ]]; then
  echo "missing venv activate script: $VENV_PATH" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV_PATH"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"

mkdir -p "$OUTPUT_DIR"

cd "$ROOT_DIR"


for batch_size in "${BATCH_SIZES[@]}"; do
  for kv_dtype in "${DTYPES[@]}"; do

    output_json="$OUTPUT_DIR/${kv_dtype}_decode_graph_policy_bs${batch_size}.json"

    echo "==> sweep kv_dtype=${kv_dtype} batch_size=${batch_size}"
    python scripts/sweep_decode_graph_policy.py \
      --kv-dtype "$kv_dtype" \
      --batch-list "$batch_size" \
      --page-start "$PAGE_START" \
      --page-stop "$PAGE_STOP" \
      --capture-page-count "$CAPTURE_PAGE_COUNT" \
      --candidate-ctas-per-sm "$CANDIDATE_CTAS_PER_SM" \
      --candidate-splits "$CANDIDATE_SPLITS" \
      --parallel-workers "$PARALLEL_WORKERS" \
      --replays "$REPLAYS" \
      --probe-batch-replays "$PROBE_BATCH_REPLAYS" \
      --ci-level "$CI_LEVEL" \
      --output "$output_json" \
      --chunk-fill-windowed \
      --chunk-fill-window-sample-divisor 18 \
      --chunk-fill-window-relative-pad 0.10 \
      --chunk-fill-window-absolute-pad 10 \
      --summary

    echo "==> generate tuning kv_dtype=${kv_dtype} batch_size=${batch_size}"
    python scripts/generate_decode_policy_tuning.py \
      --input "$output_json"

    sudo killall -9 python || true
  done
done

echo "all sweeps complete"
