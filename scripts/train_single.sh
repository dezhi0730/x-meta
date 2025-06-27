#!/bin/bash

# 设置 CUDA 可见设备（只使用一张卡）
export CUDA_VISIBLE_DEVICES=0

# 设置 OMP 线程数
export OMP_NUM_THREADS=1

# 配置文件路径
CONFIG="gutclip/configs/default.yaml"

echo "[Info] Starting single GPU training..."
echo "[Info] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# 启动单卡训练
python gutclip/cmdline/main.py \
    --cfg "${CONFIG}" \
    "wandb.mode=offline" 