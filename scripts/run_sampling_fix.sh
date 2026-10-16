#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -e datasets && -d ../datasets ]]; then
  ln -s ../datasets datasets
fi

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-eidrec}"

OUTPUT_DIR="${OUTPUT_DIR:-output_sampling_fix}"
MODEL_IDX="${MODEL_IDX:-0}"
DATA_NAME="${DATA_NAME:-Beauty}"
RUN_ALL="${RUN_ALL:-0}"

CLUSTER_UPDATE_INTERVAL="${CLUSTER_UPDATE_INTERVAL:-3}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
SAMPLE_ETA="${SAMPLE_ETA:-1.0}"
EVAL_TOPK_ON_GPU="${EVAL_TOPK_ON_GPU:-1}"
SPECTRAL_VIEW_FILTER="${SPECTRAL_VIEW_FILTER:-1}"
SPECTRAL_TAU_INIT="${SPECTRAL_TAU_INIT:-0.1}"
SPECTRAL_LAMBDA_SCALE="${SPECTRAL_LAMBDA_SCALE:-1.0}"
SPECTRAL_FILTER_RESIDUAL="${SPECTRAL_FILTER_RESIDUAL:-0.5}"
GUIDE_ENTROPY_LAMBDA="${GUIDE_ENTROPY_LAMBDA:-0.1}"
GUIDE_ENTROPY_TEMPERATURE="${GUIDE_ENTROPY_TEMPERATURE:-1.0}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-256}"

mkdir -p "${OUTPUT_DIR}"

dataset_args() {
  case "$1" in
    Beauty)
      echo "--diff_weight 1 --cl_weight 0.2 --timesteps 50 --intent_num 32 --w 2"
      ;;
    Toys_and_Games)
      echo "--diff_weight 0.2 --cl_weight 0.2 --timesteps 200 --intent_num 1024 --w 2"
      ;;
    Sports_and_Outdoors)
      echo "--diff_weight 1 --cl_weight 0.4 --timesteps 200 --intent_num 256 --w 2"
      ;;
    Yelp)
      echo "--diff_weight 1 --cl_weight 0.4 --timesteps 200 --intent_num 256 --w 2 --guide_entropy_lambda 0.01 --guide_entropy_temperature 2.0"
      ;;
    *)
      echo "Unsupported DATA_NAME: $1" >&2
      exit 2
      ;;
  esac
}

run_one() {
  local data_name="$1"
  shift || true

  local eval_topk_arg=(--eval_topk_on_gpu)
  if [[ "${EVAL_TOPK_ON_GPU}" != "1" ]]; then
    eval_topk_arg=(--no-eval_topk_on_gpu)
  fi

  local spectral_arg=(--spectral_view_filter)
  if [[ "${SPECTRAL_VIEW_FILTER}" != "1" ]]; then
    spectral_arg=(--no-spectral_view_filter)
  fi

  # shellcheck disable=SC2207
  local data_args=($(dataset_args "${data_name}"))

  "${PYTHON_BIN}" EIDRec_main.py \
    --model_name EIDRec_sampling_fix \
    --model_idx "${MODEL_IDX}" \
    --output_dir "${OUTPUT_DIR}" \
    --data_name "${data_name}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --cluster_update_interval "${CLUSTER_UPDATE_INTERVAL}" \
    --sample_steps "${SAMPLE_STEPS}" \
    --sample_eta "${SAMPLE_ETA}" \
    --spectral_tau_init "${SPECTRAL_TAU_INIT}" \
    --spectral_lambda_scale "${SPECTRAL_LAMBDA_SCALE}" \
    --spectral_filter_residual "${SPECTRAL_FILTER_RESIDUAL}" \
    --guide_entropy_lambda "${GUIDE_ENTROPY_LAMBDA}" \
    --guide_entropy_temperature "${GUIDE_ENTROPY_TEMPERATURE}" \
    "${eval_topk_arg[@]}" \
    "${spectral_arg[@]}" \
    "${data_args[@]}" \
    "$@"
}

if [[ "${RUN_ALL}" == "1" ]]; then
  for dataset in Beauty Toys_and_Games Sports_and_Outdoors Yelp; do
    run_one "${dataset}"
  done
else
  run_one "${DATA_NAME}"
fi
