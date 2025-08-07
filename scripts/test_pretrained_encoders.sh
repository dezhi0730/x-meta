#!/usr/bin/env bash
# ============================================
# 测试预训练编码器加载
# ============================================

set -euo pipefail
export PYTHONWARNINGS="ignore::FutureWarning"

# 预训练检查点路径
PRETRAINED_CKPT="/data/shared/x-meta/checkpoints/gutclip_exp_best_20250724-090522_top10.5034_recall@50.8152_mrr0.6400.pt"

echo "============================================"
echo "[INFO] 测试预训练编码器加载"
echo "[INFO] 检查点路径: ${PRETRAINED_CKPT}"
echo "============================================"

# 检查检查点文件是否存在
if [[ ! -f "${PRETRAINED_CKPT}" ]]; then
    echo "[ERROR] 预训练检查点文件不存在: ${PRETRAINED_CKPT}"
    exit 1
fi

# 测试DNA encoder加载
echo "[INFO] 测试DNA encoder加载..."
python -c "
import torch
from gutclip.models.diffusion.separated_unet import SeparatedDiffusionModelWithDNA

# 创建模型
model = SeparatedDiffusionModelWithDNA(
    input_dim=4,
    dna_dim=768,
    hidden_dim=128,
    out_dim=256,
    num_layers=4,
    dropout_rate=0.25,
    pretrained_dna_encoder=True
)

# 加载预训练DNA encoder
model.load_pretrained_encoders('${PRETRAINED_CKPT}', load_tree_encoder=False)

print('[SUCCESS] DNA encoder加载测试通过')
"

# 测试DNA encoder + tree encoder加载
echo "[INFO] 测试DNA encoder + tree encoder加载..."
python -c "
import torch
from gutclip.models.diffusion.separated_unet import SeparatedDiffusionModelWithDNA

# 创建模型
model = SeparatedDiffusionModelWithDNA(
    input_dim=4,
    dna_dim=768,
    hidden_dim=128,
    out_dim=256,
    num_layers=4,
    dropout_rate=0.25,
    pretrained_dna_encoder=True
)

# 加载预训练编码器
model.load_pretrained_encoders('${PRETRAINED_CKPT}', load_tree_encoder=True)

print('[SUCCESS] DNA encoder + tree encoder加载测试通过')
"

echo "[SUCCESS] 所有预训练编码器加载测试通过！" 