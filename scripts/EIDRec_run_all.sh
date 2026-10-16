#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"
MODEL_IDX="${MODEL_IDX:-0}"
RESULT_CSV="${RESULT_CSV:-${OUTPUT_DIR}/EIDRec_results.csv}"

CLUSTER_UPDATE_INTERVAL="${CLUSTER_UPDATE_INTERVAL:-3}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
EVAL_TOPK_ON_GPU="${EVAL_TOPK_ON_GPU:-1}"
SPECTRAL_VIEW_FILTER="${SPECTRAL_VIEW_FILTER:-1}"
SPECTRAL_TAU_INIT="${SPECTRAL_TAU_INIT:-0.1}"
SPECTRAL_LAMBDA_SCALE="${SPECTRAL_LAMBDA_SCALE:-1.0}"
SPECTRAL_FILTER_RESIDUAL="${SPECTRAL_FILTER_RESIDUAL:-0.5}"
GUIDE_ENTROPY_LAMBDA="${GUIDE_ENTROPY_LAMBDA:-0.1}"
GUIDE_ENTROPY_TEMPERATURE="${GUIDE_ENTROPY_TEMPERATURE:-1.0}"

mkdir -p "${OUTPUT_DIR}"

common_args=(
  --model_name EIDRec
  --model_idx "${MODEL_IDX}"
  --output_dir "${OUTPUT_DIR}"
  --cluster_update_interval "${CLUSTER_UPDATE_INTERVAL}"
  --sample_steps "${SAMPLE_STEPS}"
  --spectral_tau_init "${SPECTRAL_TAU_INIT}"
  --spectral_lambda_scale "${SPECTRAL_LAMBDA_SCALE}"
  --spectral_filter_residual "${SPECTRAL_FILTER_RESIDUAL}"
  --guide_entropy_lambda "${GUIDE_ENTROPY_LAMBDA}"
  --guide_entropy_temperature "${GUIDE_ENTROPY_TEMPERATURE}"
)

if [[ "${EVAL_TOPK_ON_GPU}" == "1" ]]; then
  common_args+=(--eval_topk_on_gpu)
else
  common_args+=(--no-eval_topk_on_gpu)
fi

if [[ "${SPECTRAL_VIEW_FILTER}" == "1" ]]; then
  common_args+=(--spectral_view_filter)
else
  common_args+=(--no-spectral_view_filter)
fi

"${PYTHON_BIN}" EIDRec_main.py "${common_args[@]}" --data_name Beauty --diff_weight 1 --cl_weight 0.2 --timesteps 50 --intent_num 32 --w 2
"${PYTHON_BIN}" EIDRec_main.py "${common_args[@]}" --data_name Toys_and_Games --diff_weight 0.2 --cl_weight 0.2 --timesteps 200 --intent_num 1024 --w 2
"${PYTHON_BIN}" EIDRec_main.py "${common_args[@]}" --data_name Sports_and_Outdoors --diff_weight 1 --cl_weight 0.4 --timesteps 200 --intent_num 256 --w 2
"${PYTHON_BIN}" EIDRec_main.py "${common_args[@]}" --data_name Yelp --diff_weight 1 --cl_weight 0.4 --timesteps 200 --intent_num 256 --w 2 --guide_entropy_lambda 0.01 --guide_entropy_temperature 2.0

"${PYTHON_BIN}" - "${OUTPUT_DIR}" "${MODEL_IDX}" "${RESULT_CSV}" <<'PY'
import ast
import csv
import os
import sys

output_dir, model_idx, result_csv = sys.argv[1:4]
model_name = "EIDRec"
datasets = ["Beauty", "Toys_and_Games", "Sports_and_Outdoors", "Yelp"]
metric_keys = ["HR@5", "NDCG@5", "HR@10", "NDCG@10", "HR@20", "NDCG@20"]

rows = []
for data_name in datasets:
    log_file = os.path.join(output_dir, f"{model_name}-{data_name}-{model_idx}.txt")
    latest_metrics = None
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not (line.startswith("{") and "HR@5" in line):
                    continue
                try:
                    record = ast.literal_eval(line)
                except (SyntaxError, ValueError):
                    continue
                if all(key in record for key in metric_keys):
                    latest_metrics = record

    row = {"dataset": data_name, "model_name": model_name, "model_idx": model_idx, "log_file": log_file}
    row.update({key: latest_metrics[key] if latest_metrics else "" for key in metric_keys})
    rows.append(row)

fieldnames = ["dataset", "model_name", "model_idx", *metric_keys, "log_file"]
with open(result_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote EIDRec results to {result_csv}")
PY
