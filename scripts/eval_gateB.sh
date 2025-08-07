#!/usr/bin/env bash
# ============================================
# Evaluate Gate B: Residual Calibration & Sampling Stability
# Usage:
#   ./scripts/eval_gateB.sh                       # 使用默认配置
#   ./scripts/eval_gateB.sh path/to/checkpoint    # 指定检查点
#
# 参数说明：
#   $1  检查点路径（可选，默认使用最新检查点）
#   $2  输出目录（可选，默认 logs/gateB_${TS}）
# ============================================

set -euo pipefail

# 生成时间戳
TS=$(date "+%Y%m%d-%H%M%S")

# 默认参数
CHECKPOINT_PATH=${1:-"/data/shared/x-meta/checkpoints/diffusion/gutclip_diff_20250729-133757_best_tail_20250729-145326_vmse0.184.pt"}
OUTPUT_DIR=${2:-"logs/gateB_${TS}"}

DEFAULT_OUTPUT_DIR="logs/gateB_${TS}"

# 如果没有指定检查点，尝试找到最新的
if [[ -z "${CHECKPOINT_PATH}" ]]; then
    # 查找最新的检查点（支持 .ckpt 和 .pt 格式）
    LATEST_CKPT=$(find . -name "*.ckpt" -o -name "*.pt" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" ")
    if [[ -n "${LATEST_CKPT}" ]]; then
        CHECKPOINT_PATH="${LATEST_CKPT}"
        echo "[Info] Using latest checkpoint: ${CHECKPOINT_PATH}"
    else
        echo "[Error] No checkpoint found. Please specify a checkpoint path."
        exit 1
    fi
fi

# 设置输出目录
if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
fi

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "[Gate B] Residual Calibration & Sampling Stability"
echo "[Checkpoint] ${CHECKPOINT_PATH}"
echo "[Output] ${OUTPUT_DIR}"
echo "============================================"


# 运行评估
python -m gutclip.evaluate.eval_gateB \
    --checkpoint "${CHECKPOINT_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --config gutclip/configs/train_gen_model.yaml \
    --max_batches 20 \
    --num_ddim_steps 50 \
    --plot \
    2>&1 | tee "${OUTPUT_DIR}/gateB_eval.log"

echo "============================================"
echo "[Gate B] Evaluation completed!"
echo "[Results] Check ${OUTPUT_DIR}/gateB_results.json"
echo "[Plots] Check ${OUTPUT_DIR}/gateB_plots.png"
echo "============================================" 