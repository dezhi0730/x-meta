#!/bin/bash

# 设置 CUDA 可见设备（根据实际 GPU 数量调整）
export CUDA_VISIBLE_DEVICES=0,1

# 设置 NCCL 调试信息（可选）
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # 如果使用 InfiniBand，可以注释掉这行

# 设置超时和调试
export NCCL_TIMEOUT=30
export NCCL_BLOCKING_WAIT=1

# 设置 OMP 线程数（可选）
export OMP_NUM_THREADS=1

# 配置文件路径
CONFIG="gutclip/configs/default.yaml"

echo "[Info] Starting distributed training with 2 GPUs..."
echo "[Info] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "[Info] NCCL_DEBUG: $NCCL_DEBUG"

# 启动分布式训练
torchrun \
    --nproc_per_node=2 \
    --master_port=29510 \
    gutclip/cmdline/main.py \
    --cfg "${CONFIG}" \
    "wandb.mode=offline"

