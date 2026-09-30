#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/scratch/public/sg2240/hf_cache
export TMPDIR=/scratch/public/sg2240/tmp
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

CATEGORY="Industrial_and_Scientific"

TRAIN_FILE="data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv"
EVAL_FILE="data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv"
SID_INDEX="data/Amazon/index/Industrial_and_Scientific.index.json"
ITEM_META="data/Amazon/index/Industrial_and_Scientific.item.json"

OUTPUT_DIR="/scratch/public/sg2240/outputs/minionerec_sft_smoke"

mkdir -p "${OUTPUT_DIR}"

python sft.py \
  --base_model "Qwen/Qwen2.5-0.5B-Instruct" \
  --train_file "${TRAIN_FILE}" \
  --eval_file "${EVAL_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --category "${CATEGORY}" \
  --sid_index_path "${SID_INDEX}" \
  --item_meta_path "${ITEM_META}" \
  --sample 32 \
  --batch_size 8 \
  --micro_batch_size 2 \
  --num_epochs 1 \
  --max_steps 10 \
  --eval_steps 5 \
  --learning_rate 1e-5 \
  --cutoff_len 256 \
  --seed 42 \
  --train_from_scratch False \
  --freeze_LLM False \
  --wandb_project "" \
  --wandb_run_name "single-gpu-smoke"