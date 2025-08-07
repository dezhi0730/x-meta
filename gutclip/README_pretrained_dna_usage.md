# 使用预训练DNA Encoder的Separated Diffusion Model

## 概述

本实现支持使用预训练的GutCLIP模型中的DNA encoder来增强Separated Diffusion Model的性能。预训练的DNA encoder已经在DNA-丰度对齐任务上取得了良好的效果，可以更好地理解DNA序列与微生物丰度之间的关系。

## 主要改进

### 1. 预训练DNA Encoder集成
- 使用GutCLIP中训练好的DNAEncoder
- 支持注意力池化和序列变换
- 自动处理padding和随机mask

### 2. 权重加载机制
- 从GutCLIP检查点中提取DNA encoder权重
- 自动处理Lightning/DDP前缀
- 支持冻结预训练参数

### 3. 维度对齐
- DNA encoder输出维度与模型hidden_dim对齐
- 保持与原有架构的兼容性

## 使用方法

### 1. 基础训练（不使用预训练DNA encoder）
```bash
# 使用随机初始化的DNA encoder
./scripts/train_separated_diffusion.sh
```

### 2. 使用预训练DNA encoder
```bash
# 指定预训练GutCLIP检查点路径
./scripts/train_separated_diffusion.sh \
  single \
  1 \
  gutclip/configs/train_separated_diffusion.yaml \
  50 \
  offline \
  /path/to/your/gutclip_checkpoint.pt
```

### 3. 直接使用Python脚本
```bash
python -m gutclip.cmdline.train_separated_diffusion \
  --config gutclip/configs/train_separated_diffusion.yaml \
  --pretrained_ckpt /path/to/your/gutclip_checkpoint.pt
```

## 模型架构

### SeparatedDiffusionModelWithDNA
```python
class SeparatedDiffusionModelWithDNA(SeparatedDiffusionModel):
    def __init__(self, input_dim=4, dna_dim=768, hidden_dim=128, ...):
        # 使用预训练的DNAEncoder
        self.dna_encoder = DNAEncoder(
            input_dim=dna_dim,      # 768 (DNA-BERT嵌入维度)
            output_dim=hidden_dim,  # 128 (与模型hidden_dim对齐)
            dropout_rate=dropout_rate
        )
    
    def load_pretrained_dna_encoder(self, gutclip_checkpoint_path):
        # 从GutCLIP检查点加载DNA encoder权重
        # 自动处理前缀和冻结参数
```

## 数据流程

### 1. DNA数据处理
```python
# 输入: batch.dna (B, L_max, 768)
# 输入: batch.dna_pad_mask (B, L_max) - padding mask
# 输入: batch.dna_rand_mask (B, L_max) - 随机mask

# 使用预训练DNA encoder
dna_encoded = self.dna_encoder(batch.dna, dna_pad_mask, dna_rand_mask)  # (B, 128)

# 扩展到节点级别
batch_indices = batch.batch.long()
dna_nodes = dna_encoded[batch_indices]  # (ΣN, 128)
```

### 2. 特征拼接
```python
# 基础特征: (ΣN, 4)
x_comb_base = torch.cat([x_t, x_static, mask_feat], dim=1)

# DNA特征: (ΣN, 128)
dna_nodes = dna_encoded[batch_indices]

# 最终输入: (ΣN, 132)
x_comb = torch.cat([x_comb_base, dna_nodes], dim=1)
```

### 3. 树编码器处理
```python
# 通过树编码器
h = self.tree_encoder(x_comb, edge_index, pos, batch)  # (ΣN, 256)

# 时间条件化
t_cond_nodes = t_cond[batch.batch]  # (ΣN, 256)
h = h + t_cond_nodes  # (ΣN, 256)

# 分离头预测
eps_hat = self.abun_head(h).squeeze(-1)      # (ΣN,)
pres_logit = self.pres_head(h).squeeze(-1)   # (ΣN,)
```

## 预训练检查点格式

GutCLIP检查点应包含以下DNA encoder权重：
```
dna_encoder.pool.query.weight
dna_encoder.pool.proj.weight
dna_encoder.transform.0.weight
dna_encoder.transform.0.bias
dna_encoder.transform.2.weight
dna_encoder.transform.2.bias
dna_encoder.transform.3.weight
dna_encoder.transform.3.bias
dna_encoder.transform.5.weight
dna_encoder.transform.5.bias
dna_encoder.transform.6.weight
dna_encoder.transform.6.bias
```

## 配置示例

### 配置文件 (train_separated_diffusion.yaml)
```yaml
# 启用DNA条件
use_dna: true
dna_dim: 768

# 模型配置
model:
  hidden_dim: 128
  out_dim: 256
  num_layers: 4
  dropout_rate: 0.25

# 训练配置
train:
  epochs: 50
  batch_size: 128
  lr: 1.0e-4
  separated_modeling: true
  lambda_abun: 1.0
  lambda_pres: 1.0
```

## 优势

### 1. 更好的DNA理解
- 预训练DNA encoder已经学习了DNA序列与丰度的对应关系
- 注意力机制能够关注重要的DNA片段
- 支持序列级别的mask和padding处理

### 2. 训练稳定性
- 预训练权重提供良好的初始化
- 减少训练时间
- 提高收敛稳定性

### 3. 性能提升
- 利用已有的DNA-丰度对齐知识
- 更好的特征表示
- 更准确的丰度预测

## 注意事项

### 1. 检查点兼容性
- 确保GutCLIP检查点包含完整的DNA encoder权重
- 检查DNA encoder的架构参数是否匹配

### 2. 内存使用
- 预训练DNA encoder会增加模型参数量
- 建议使用较小的batch_size

### 3. 训练策略
- 可以选择冻结或微调DNA encoder
- 建议先冻结训练几个epoch，再解冻微调

## 故障排除

### 1. 权重加载失败
```bash
# 检查检查点中的键
python -c "
import torch
ckpt = torch.load('your_checkpoint.pt', map_location='cpu')
state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))
dna_keys = [k for k in state_dict.keys() if k.startswith('dna_encoder')]
print('DNA encoder keys:', dna_keys)
"
```

### 2. 维度不匹配
- 确保DNA encoder的output_dim与模型的hidden_dim一致
- 检查DNA数据的输入维度是否为768

### 3. 内存不足
- 减少batch_size
- 使用梯度累积
- 启用混合精度训练 