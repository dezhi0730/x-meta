#!/bin/bash

# 设置 CUDA 可见设备（根据实际 GPU 数量调整）
export CUDA_VISIBLE_DEVICES=1,2,3

# 设置 NCCL 调试信息（可选）
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # 如果使用 InfiniBand，可以注释掉这行

# 设置 OMP 线程数（可选）
export OMP_NUM_THREADS=1

# 配置文件路径
CONFIG="gutclip/configs/default.yaml"

# 启动分布式训练
torchrun \
    --nproc_per_node=3 \
    --master_port=29500 \
    gutclip/cmdline/main.py \
    --cfg "${CONFIG}" \
    "wandb.mode=online"

# 训练完成后，将最佳模型复制到指定位置
if [ -f "checkpoints/${NAME}_best.pt" ]; then
    cp "checkpoints/${NAME}_best.pt" "checkpoints/${NAME}_final.pt"
fi
