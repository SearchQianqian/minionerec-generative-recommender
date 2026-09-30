#!/usr/bin/env bash
set -euo pipefail

INFO_FILE="data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt"
TEST_FILE="data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv"
INDEX_FILE="data/Amazon/index/Industrial_and_Scientific.index.json"
ROOT="artifacts/eval_full"

mkdir -p "${ROOT}"

while IFS='|' read -r NAME CHECKPOINT
do
    OUTPUT_DIR="${ROOT}/${NAME}"
    mkdir -p "${OUTPUT_DIR}"

    echo "Evaluating ${NAME}"
    echo "Checkpoint: ${CHECKPOINT}"

    python evaluate.py \
      --base_model "${CHECKPOINT}" \
      --info_file "${INFO_FILE}" \
      --category Industrial_and_Scientific \
      --test_data_path "${TEST_FILE}" \
      --result_json_data "${OUTPUT_DIR}/predictions.json" \
      --batch_size 8 \
      --num_beams 5 \
      --max_new_tokens 5 \
      --length_penalty 0.0

    python calc.py \
      --path "${OUTPUT_DIR}/predictions.json" \
      --item_path "${INFO_FILE}" \
      > "${OUTPUT_DIR}/official_metrics.txt"

    python analyze_predictions.py \
      --predictions "${OUTPUT_DIR}/predictions.json" \
      --index "${INDEX_FILE}" \
      --topk 5 \
      > "${OUTPUT_DIR}/diagnostic_metrics.txt"

    echo "Finished ${NAME}"
done <<'MODELS'
b1_main_only|/scratch/public/sg2240/outputs/minionerec_sft_b1_main_only/checkpoint-700
b2_equal_mix|/scratch/public/sg2240/outputs/minionerec_sft_pilot/checkpoint-700
b3_main_heavy|/scratch/public/sg2240/outputs/minionerec_sft_b3_main_heavy/checkpoint-700
MODELS
