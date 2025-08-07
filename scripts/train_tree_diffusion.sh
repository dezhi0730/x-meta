#!/usr/bin/env bash
# ============================================
# Train Tree Diffusion Model
# Usage:
#   ./scripts/train_tree_diffusion.sh                    # 单卡
#   ./scripts/train_tree_diffusion.sh ddp 2              # 多卡，2 张 GPU
#
# 可选覆盖示例：
#   ./scripts/train_tree_diffusion.sh \
#     single \
#     gutclip/configs/train_tree_diffusion.yaml \
#     10 \
#     offline
#
# 参数说明（有默认值，不写就用默认）：
#   $1  模式：single | ddp                (默认 single)
#   $2  GPU 数（仅 ddp 模式需要），例如 2
#   $3  CFG 路径（顶层 yaml），默认 gutclip/configs/train_tree_diffusion.yaml
#   $4  EPOCHS 训练轮数（默认 10）
#   $5  WANDB_MODE online | offline | disabled（默认 offline）
# ============================================

set -euo pipefail
export PYTHONWARNINGS="ignore::FutureWarning"
MODE=${1:-single}
NGPUS=${2:-1}

CFG=${3:-gutclip/configs/train_tree_diffusion.yaml}
EPOCHS=${4:-10}
WANDB_MODE=${5:-offline}

# 线程数（根据机器调整）
export OMP_NUM_THREADS=16
# 如需限制可见 GPU，自行修改
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

# 生成一个时间戳 run_name 方便区分多次实验
TS=$(date "+%Y%m%d-%H%M%S")
RUN_NAME="tree_diffusion_${TS}"

echo "============================================"
echo "[Run]   ${RUN_NAME}"
echo "[Mode]  ${MODE}  (NGPUS=${NGPUS})"
echo "[Config] ${CFG}"
echo "[Epochs] ${EPOCHS}"
echo "[wandb] ${WANDB_MODE}"
echo "============================================"

# 创建日志目录
mkdir -p logs

if [[ "${MODE}" == "ddp" ]]; then
  echo "[Info] launching torchrun with ${NGPUS} GPUs..."
  torchrun --nproc_per_node="${NGPUS}" -m gutclip.cmdline.train_tree_diffusion \
    --cfg "${CFG}" \
    --epochs "${EPOCHS}" \
    2>&1 | tee "logs/${RUN_NAME}.log"
else
  echo "[Info] launching single GPU..."
  python -m gutclip.cmdline.train_tree_diffusion \
    --cfg "${CFG}" \
    --epochs "${EPOCHS}" \
    2>&1 | tee "logs/${RUN_NAME}.log"
fi 