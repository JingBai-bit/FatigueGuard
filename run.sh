#!/usr/bin/env bash
set -euo pipefail

# ======================== User configuration ========================

DATASET_NAME="DROZY"
INPUT_FILE="/path/to/data.npz"
OUTPUT_DIR="/path/to/experiment_results"

NUM_FOLDS=5
FOLDS="all"                        # all, 1, or e.g. "1,3,5"
VALIDATION_OFFSET=2
PURGE_BOUNDARY_WINDOWS=0

MODALITIES="multi,eeg,eog"
MIXTURE_EXPERT_LAYERS=3
SELECTION_METRIC="acc"             # auroc, f1, acc, or composite
PRETRAIN_EPOCHS=200
FINETUNE_EPOCHS=100
PATIENCE=20
BATCH_SIZE=128
PRETRAIN_LR="1e-4"
FINETUNE_LR="3e-4"
WEIGHT_DECAY="0.05"
FINETUNE_FRACTION="1.0"
SEED=2026

# 1: retrain on all non-test folds after epoch selection; 0: test the selected model directly.
REFIT_NON_TEST=1
SAVE_CHECKPOINTS=1
# 1: validate the NPZ interface and temporal folds without training.
CHECK_DATA_ONLY=0

GPU_ID="1"
DEVICE="cuda"                      # cuda, cuda:0, or cpu

# ====================================================================

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
check_config=0
if (($#)); then
  if [[ "$1" == "--check-config" && $# -eq 1 ]]; then
    check_config=1
  else
    echo "Set parameters at the top of this script; the only supported command-line option is --check-config" >&2
    exit 2
  fi
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "fuse" ]]; then
  echo "Run 'conda activate fuse' first; current environment: ${CONDA_DEFAULT_ENV:-not activated}" >&2
  exit 2
fi
command -v python >/dev/null 2>&1 || { echo "Python was not found in the current environment" >&2; exit 2; }
if (( check_config == 0 )); then
  [[ -f "$INPUT_FILE" ]] || { echo "Preprocessed NPZ file not found: $INPUT_FILE" >&2; exit 2; }
fi
case "$DEVICE" in
  cuda|cuda:0)
    [[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo "GPU_ID must be a non-negative integer" >&2; exit 2; }
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    ;;
  cpu) ;;
  *) echo "DEVICE must be cuda, cuda:0, or cpu" >&2; exit 2 ;;
esac

run_stamp="$(date -u +%Y%m%dT%H%M%S%3NZ)"
run_dir="$OUTPUT_DIR/experiment_${run_stamp}"
runner_args=(
  --input-file "$INPUT_FILE"
  --dataset-name "$DATASET_NAME"
  --output-dir "$run_dir"
  --num-folds "$NUM_FOLDS"
  --folds "$FOLDS"
  --validation-offset "$VALIDATION_OFFSET"
  --purge-boundary-windows "$PURGE_BOUNDARY_WINDOWS"
  --modalities "$MODALITIES"
  --mixture-expert-layers "$MIXTURE_EXPERT_LAYERS"
  --selection-metric "$SELECTION_METRIC"
  --pretrain-epochs "$PRETRAIN_EPOCHS"
  --finetune-epochs "$FINETUNE_EPOCHS"
  --patience "$PATIENCE"
  --batch-size "$BATCH_SIZE"
  --pretrain-lr "$PRETRAIN_LR"
  --finetune-lr "$FINETUNE_LR"
  --weight-decay "$WEIGHT_DECAY"
  --finetune-fraction "$FINETUNE_FRACTION"
  --seed "$SEED"
  --device "$DEVICE"
)
(( REFIT_NON_TEST == 0 )) || runner_args+=(--refit-non-test)
(( SAVE_CHECKPOINTS == 0 )) || runner_args+=(--save-checkpoints)
(( CHECK_DATA_ONLY == 0 )) || runner_args+=(--check-data-only)

printf '%s temporal CV: input=%s folds=%s/%s output=%s\n' \
  "$DATASET_NAME" "$INPUT_FILE" "$FOLDS" "$NUM_FOLDS" "$run_dir"
if (( check_config == 1 )); then
  printf 'Command:'
  printf ' %q' python "$script_dir/main.py" "${runner_args[@]}"
  printf '\n'
  exit 0
fi

mkdir -p -- "$OUTPUT_DIR"
mkdir -- "$run_dir"
log_path="$run_dir/train_${run_stamp}.log"
{
  printf 'Training log: %s\n' "$log_path"
  printf 'Dataset: %s; input: %s\n' "$DATASET_NAME" "$INPUT_FILE"
  printf 'Device: %s; GPU_ID: %s; CUDA_VISIBLE_DEVICES: %s\n' \
    "$DEVICE" "$GPU_ID" "${CUDA_VISIBLE_DEVICES:-unset}"
} | tee "$log_path"
PYTHONUNBUFFERED=1 python "$script_dir/main.py" "${runner_args[@]}" \
  2>&1 | tee -a "$log_path"

