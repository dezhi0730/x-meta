#!/bin/bash

# 步骤1：拆分大文件（一次性执行）
echo "[STEP 1] Splitting large pkl file..."
python scripts/split_pkl_to_pt.py \
    --src /data/home/wudezhi/project/school/x-meta/datasets/raw/tree/all_samples.pkl \
    --dst /data/home/wudezhi/project/school/x-meta/datasets/raw/tree/tree_split

# 步骤2：使用拆分模式训练
echo "[STEP 2] Training with split dataset mode..."
python gutclip/cmdline/main.py \
    --cfg gutclip/configs/default.yaml \
    "data.type=split" \
    "wandb.mode=offline" 