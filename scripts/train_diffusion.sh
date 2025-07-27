#!/usr/bin/env bash
# ============================================
# Train conditional diffusion for GutCLIP prior → y
# Usage:
#   ./scripts/train_diffusion.sh                       # 单卡
#   ./scripts/train_diffusion.sh ddp 2                 # 多卡，2 张 GPU
#
# 可选覆盖示例：
#   ./scripts/train_diffusion.sh \
#     single \
#     gutclip/configs/train_gen_model.yaml \
#     datasets/diffusion/V3/faiss_index.faiss \
#     datasets/diffusion/V3/faiss_index.y.npy \
#     datasets/diffusion/V3/faiss_index.ids.npy \
#     5 \
#     offline
#
# 参数说明（有默认值，不写就用默认）：
#   $1  模式：single | ddp                (默认 single)
#   $2  GPU 数（仅 ddp 模式需要），例如 2
#   $3  CFG 路径（顶层 yaml），默认 gutclip/configs/train_gen_model.yaml
#   $4  RET_INDEX  faiss 索引
#   $5  RET_Y      y.npy
#   $6  RET_IDS    ids.npy（可选，留空或不存在则不覆盖）
#   $7  K          top-k（默认 5）
#   $8  WANDB_MODE online | offline | disabled（默认 offline）
# ============================================

set -euo pipefail

MODE=${1:-single}
NGPUS=${2:-1}

CFG=${3:-gutclip/configs/train_gen_model.yaml}
RET_INDEX=${4:-datasets/diffusion/V3/faiss_index.faiss}
RET_Y=${5:-datasets/diffusion/V3/faiss_index.y.npy}
RET_IDS=${6:-datasets/diffusion/V3/faiss_index.ids.npy}
K=${7:-5}
WANDB_MODE=${8:-offline}


# 线程数（根据机器调整）
export OMP_NUM_THREADS=16
# 如需限制可见 GPU，自行修改
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

# 生成一个时间戳 run_name 方便区分多次实验
TS=$(date "+%Y%m%d-%H%M%S")
RUN_NAME="gutclip_diff_${TS}"

echo "============================================"
echo "[Run]   ${RUN_NAME}"
echo "[Mode]  ${MODE}  (NGPUS=${NGPUS})"
echo "[Index] ${RET_INDEX}"
echo "[Y]     ${RET_Y}"
echo "[IDs]   ${RET_IDS}"
echo "[k]     ${K}"
echo "[wandb] ${WANDB_MODE}"
echo "============================================"

# 组装通用的 Hydra overrides
OVERRIDES=(
  run_name="${RUN_NAME}"
  wandb.mode="${WANDB_MODE}"
  retrieval.index="${RET_INDEX}"
  retrieval.y="${RET_Y}"
  retrieval.k="${K}"
  # 避免 Hydra 改变工作目录
  hydra.job.chdir=false
  hydra.run.dir=.
)

# 只有当 ids 文件真实存在时再覆盖（启用 LOO）
if [[ -f "${RET_IDS}" ]]; then
  OVERRIDES+=( retrieval.ids="${RET_IDS}" )
else
  echo "[Info] ids file not found → skip leave-one-out."
fi

# 创建日志目录
mkdir -p logs

if [[ "${MODE}" == "ddp" ]]; then
  echo "[Info] launching torchrun with ${NGPUS} GPUs..."
  torchrun --nproc_per_node="${NGPUS}" -m gutclip.cmdline.train_gen_model \
    --ddp \
    "${OVERRIDES[@]}" \
    2>&1 | tee "logs/${RUN_NAME}.log"
else
  echo "[Info] launching single GPU..."
  python -m gutclip.cmdline.train_gen_model \
    "${OVERRIDES[@]}" \
    2>&1 | tee "logs/${RUN_NAME}.log"
fi