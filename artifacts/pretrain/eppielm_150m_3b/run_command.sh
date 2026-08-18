#!/usr/bin/env bash

PYTHONPATH=. python -u train/pretrain.py \
  --data_path /root/autodl-tmp/data/eppielm_3b.bin \
  --output_dir /root/autodl-tmp/outputs/eppielm_150m_3b \
  --precision bf16 \
  --micro_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --sdpa_backend flash \
  --compile \
  --compile_mode max-autotune \
  --max_steps 0 \
  --warmup_steps 2000 \
  --log_steps 20 \
  --save_steps 5000 \
  --keep_last_checkpoints 3
