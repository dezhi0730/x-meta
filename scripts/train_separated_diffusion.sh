#!/usr/bin/env bash
# ============================================
# Train Separated Diffusion Model (Bernoulli + Gaussian)
# Usage:
#   ./scripts/train_separated_diffusion.sh                    # 单卡
#   ./scripts/train_separated_diffusion.sh ddp 2              # 多卡，2 张 GPU
#
# 可选覆盖示例：
#   ./scripts/train_separated_diffusion.sh ddp 2              # 多卡，2 张 GPU
#   ./scripts/train_separated_diffusion.sh single              # 单卡
#
# 参数说明：
#   $1  模式：single | ddp                (默认 single)
#   $2  GPU 数（仅 ddp 模式需要），例如 2
#   注意：所有训练参数（epochs、lr、batch_size等）都从配置文件中读取
# ============================================

set -euo pipefail
export PYTHONWARNINGS="ignore::FutureWarning"
MODE=${1:-single}
NGPUS=${2:-1}

# 固定配置文件路径
CFG="gutclip/configs/train_separated_diffusion.yaml"

# 线程数（根据机器调整）
export OMP_NUM_THREADS=16
# 如需限制可见 GPU，自行修改
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

# 生成一个时间戳 run_name 方便区分多次实验
TS=$(date "+%Y%m%d-%H%M%S")
RUN_NAME="separated_diffusion_${TS}"

echo "============================================"
echo "[Run]   ${RUN_NAME}"
echo "[Mode]  ${MODE}  (NGPUS=${NGPUS})"
echo "[Config] ${CFG}"
echo "============================================"

# 检查配置文件是否存在
if [[ ! -f "${CFG}" ]]; then
    echo "[ERROR] Config file not found: ${CFG}"
    echo "Available configs:"
    ls -la gutclip/configs/train_separated_diffusion*.yaml 2>/dev/null || echo "No separated diffusion configs found"
    exit 1
fi

# 检查数据路径是否存在
echo "[INFO] Checking data paths..."
DATA_CFG=$(python -c "import yaml; cfg=yaml.safe_load(open('${CFG}')); print(yaml.dump(cfg['data'], default_flow_style=False))")
echo "Data config:"
echo "${DATA_CFG}"

# 创建日志目录
mkdir -p logs

if [[ "${MODE}" == "ddp" ]]; then
  echo "[Info] launching torchrun with ${NGPUS} GPUs..."
  torchrun --nproc_per_node="${NGPUS}" -m gutclip.cmdline.train_separated_diffusion \
    --config "${CFG}" \
    2>&1 | tee "logs/${RUN_NAME}.log"
else
  echo "[Info] launching single GPU..."
  python -m gutclip.cmdline.train_separated_diffusion \
    --config "${CFG}" \
    2>&1 | tee "logs/${RUN_NAME}.log"
fi

echo "[INFO] Training completed! Check logs/${RUN_NAME}.log for details." 