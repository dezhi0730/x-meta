#!/usr/bin/env bash
# ============================================
# 测试基于配置文件的模型创建
# ============================================

set -euo pipefail
export PYTHONWARNINGS="ignore::FutureWarning"

echo "============================================"
echo "[INFO] 测试基于配置文件的模型创建"
echo "============================================"

# 测试基础模型创建
echo "[INFO] 测试基础模型创建..."
python -c "
import torch
from omegaconf import OmegaConf
from gutclip.models.diffusion.separated_unet import SeparatedDiffusionModel, SeparatedDiffusionModelWithDNA

# 加载模型配置
model_cfg = OmegaConf.load('gutclip/configs/model/separated_diffusion.yaml')
print(f'[INFO] 模型配置: {model_cfg}')

# 创建基础模型
model = SeparatedDiffusionModel(
    input_dim=4,
    hidden_dim=128,
    out_dim=256,
    num_layers=4,
    dropout_rate=0.25,
    model_cfg=model_cfg
)
print('[SUCCESS] 基础模型创建成功')

# 创建带DNA的模型
model_with_dna = SeparatedDiffusionModelWithDNA(
    input_dim=4,
    dna_dim=768,
    hidden_dim=128,
    out_dim=256,
    num_layers=4,
    dropout_rate=0.25,
    pretrained_dna_encoder=True,
    dna_output_dim=None,
    model_cfg=model_cfg
)
print('[SUCCESS] 带DNA的模型创建成功')
print(f'[INFO] DNA输出维度: {model_with_dna.dna_output_dim}')
"

echo "[SUCCESS] 基于配置文件的模型创建测试通过！" 