#!/bin/bash

# 训练基于Prompt Embeddings的条件扩散模型
# 使用预训练的分离扩散模型作为基础

echo "=== 开始训练Prompt条件扩散模型 ==="

# 激活环境
source ~/.bashrc
conda activate flora_plus

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/data/home/wudezhi/project/school/x-meta:$PYTHONPATH"

# 训练参数
CONFIG_FILE="gutclip/configs/train_prompt_conditional_diffusion.yaml"
PRETRAINED_CKPT="/data/home/wudezhi/project/school/x-meta/checkpoints/tree_diffusion/best_sep_20250809_000547_abun0.0907_pres10.9625.pt"
OUTPUT_DIR="/data/home/wudezhi/project/school/x-meta/checkpoints/prompt_conditional_diffusion"

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo "配置文件: $CONFIG_FILE"
echo "预训练模型: $PRETRAINED_CKPT"
echo "输出目录: $OUTPUT_DIR"

# 开始训练
python gutclip/cmdline/train_conditional_diffusion.py \
    --config $CONFIG_FILE \
    --pretrained_ckpt $PRETRAINED_CKPT \
    --device cuda \
    --seed 42

echo "=== 训练完成 ==="
