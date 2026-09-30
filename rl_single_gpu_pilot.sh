#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/scratch/public/sg2240/hf_cache
export TMPDIR=/scratch/public/sg2240/tmp
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

MODEL_PATH="/scratch/public/sg2240/outputs/minionerec_sft_full_lr1e4/checkpoint-4500"

TRAIN_FILE="data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv"
EVAL_FILE="data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv"
INFO_FILE="data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt"
SID_INDEX="data/Amazon/index/Industrial_and_Scientific.index.json"
ITEM_META="data/Amazon/index/Industrial_and_Scientific.item.json"

OUTPUT_DIR="/scratch/public/sg2240/outputs/minionerec_grpo_pilot"

mkdir -p "${OUTPUT_DIR}"

python rl_single_gpu_smoke.py   --model_path "${MODEL_PATH}"   --train_file "${TRAIN_FILE}"   --eval_file "${EVAL_FILE}"   --info_file "${INFO_FILE}"   --category "Industrial_and_Scientific"   --sid_index_path "${SID_INDEX}"   --item_meta_path "${ITEM_META}"   --output_dir "${OUTPUT_DIR}"   --sample 32   --seq_title_sample 32   --eval_sample 8   --train_batch_size 16   --eval_batch_size 16   --gradient_accumulation_steps 1   --num_train_epochs 1   --max_steps 20   --max_completion_length 8   --num_generations 16   --reward_type "ranking"   --beam_search True   --temperature 1.0   --learning_rate 1e-6   --beta 1e-3   --add_gt False   --dynamic_sampling False   --sync_ref_model False   --test_during_training False   --sample_train False   --dapo False   --gspo False   --eval_step 10   --wandb_project ""   --wandb_run_name "single-gpu-grpo-smoke"
