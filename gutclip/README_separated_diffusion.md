# 分离建模扩散模型 (Separated Diffusion Model)

本项目实现了支持 Bernoulli + Gaussian 分离建模的扩散模型，用于微生物丰度预测。

## 核心特性

### 1. 分离建模架构
- **Presence (Bernoulli)**: 使用离散扩散或二分类交叉熵处理物种存在性
- **Abundance (Gaussian)**: 仅对 presence==1 的节点加高斯噪声
- **条件注入**: 使用 class-condition 条件注入

### 2. 数据流程

#### TreeDiffusionDataset.__getitem__
```python
# 只选特定列
x_abun   = tree.x[:, [2]].float()     # log_abun
x_pres   = tree.x[:, [1]].long()      # presence 0/1
x_static = tree.x[:, [3,8]].float()   # depth_sc, degree_sc

data = Data(
    x0_abun = x_abun,
    x0_pres = x_pres,
    x_static= x_static,
    ...
)
```

#### GaussianDiffusionCollate.__call__
```python
# 连续流：log_abun
x_t, noise = self._add_noise(x0_abun, t_idx, batch)

# 离散流：presence Bernoulli 扩散
x_t_pres = self._bernoulli_noisy(x0_pres, t_idx, batch)
mask_feat = x_t_pres.float()  # 显式 mask 特征
```

### 3. 模型架构

#### 输入特征组合
```python
x_comb = torch.cat([batch.x_t, batch.x_static, batch.mask_feat], 1)
h = self.tree_encoder(x_comb, edge_index, pos, batch)
```

#### 分离头
```python
eps_hat = self.abun_head(h).squeeze(-1)      # 连续 ε̂
pres_logit = self.pres_head(h).squeeze(-1)   # presence logits
```

### 4. 损失函数

#### Presence BCE
```python
loss_pres = F.binary_cross_entropy_with_logits(
    pres_logit, batch.x0_pres.squeeze(-1).float())
```

#### Abundance MSE (掩码)
```python
mask = batch.x0_pres.squeeze(-1).float()
loss_abun = ((eps_hat - batch.noise.squeeze(-1))**2 * mask).sum() \
            / mask.sum().clamp_min(1)
```

#### 总损失
```python
loss = loss_pres + λ * loss_abun  # λ≈1
```

### 5. 两阶段采样

#### Stage-1: 离散 reverse step
```python
logp_1 = model.pres_head(h).squeeze(-1)
p_1 = torch.sigmoid(logp_1)
prob_keep = 1. / (1 + torch.exp(-logp_1))
x_{t-1}_pres = torch.bernoulli(prob_keep).long()
```

#### Stage-2: 连续流
```python
mask_feat = x_{t-1}_pres.float()
x_comb = torch.cat([x_t, x_static, mask_feat], 1)
eps_hat = model.abun_head(tree_encoder(...))
# 反推 log_abun_{t-1}; 缺失节点强制 0
```

## 使用方法

### 1. 训练模型

```bash
# 基础训练
python -m gutclip.cmdline.train_separated_diffusion \
    --config configs/train_separated_diffusion.yaml

# 带DNA条件的训练
python -m gutclip.cmdline.train_separated_diffusion \
    --config configs/train_separated_diffusion_dna.yaml
```

### 2. 采样生成

```bash
# 基础采样
python -m gutclip.cmdline.sample_separated_diffusion \
    --ckpt checkpoints/tree_diffusion/latest.pt \
    --config configs/train_separated_diffusion.yaml

# 带引导的采样
python -m gutclip.cmdline.sample_separated_diffusion \
    --ckpt checkpoints/tree_diffusion/latest.pt \
    --config configs/train_separated_diffusion.yaml \
    --guidance_scale 7.5 \
    --temperature 0.8
```

### 3. 配置文件

#### 基础配置 (configs/train_separated_diffusion.yaml)
```yaml
# 数据配置
data:
  type: "split"
  tree_dir: "/data/home/wudezhi/project/school/x-meta/datasets/raw/tree"
  train_meta: "/data/home/wudezhi/project/school/x-meta/datasets/V3_34806/train_meta.csv"
  val_meta: "/data/home/wudezhi/project/school/x-meta/datasets/V3_34806/val_meta.csv"
  dna_dir: "/data/home/wudezhi/project/school/x-meta/datasets/raw/dna_embeddings_v3"
  dna_ext: ".pt"
  max_genes: 1024

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
  lr: 1.0e-3
  weight_decay: 1.0e-4
  separated_modeling: true  # 重要：启用分离建模
  lambda_abun: 1.0
  lambda_pres: 1.0
  use_ema: false  # 暂时关闭EMA

# 扩散配置
T: 1000
beta_schedule: "linear"
beta_start: 1.0e-4
beta_end: 2.0e-2
```

#### 带DNA配置 (configs/train_separated_diffusion_dna.yaml)
```yaml
# 在基础配置基础上添加
use_dna: true
dna_dim: 768
```

## 模型文件结构

```
gutclip/
├── data/
│   └── tree_diffusion_dataset.py      # 分离建模数据集
├── models/
│   └── diffusion/
│       └── separated_unet.py          # 分离建模模型
├── engine/
│   └── trainer_tree_diffusion.py      # 分离建模训练器
├── diffusion/
│   └── separated_sampler.py           # 分离建模采样器
├── cmdline/
│   ├── train_separated_diffusion.py   # 训练脚本
│   └── sample_separated_diffusion.py  # 采样脚本
└── configs/
    └── train_separated_diffusion.yaml # 配置文件
```

## 关键优势

1. **分离建模**: 分别处理存在性和丰度，更符合生物学特性
2. **条件注入**: 使用 noisy mask 作为条件，提高生成质量
3. **掩码损失**: 只在存在节点上计算丰度损失，避免无效梯度
4. **两阶段采样**: 先确定存在性，再生成丰度，逻辑更清晰
5. **灵活配置**: 支持带/不带DNA条件，适应不同场景

## 重要修复

### 1. 设备兼容性
- betas 保持在 CPU 上，避免 DataLoader 进程中的设备混用
- 修复了 CUDA/CPU 张量混用问题

### 2. 模型维度
- 修复了 SeparatedDiffusionModelWithDNA 的 input_dim 计算
- DNA 编码后维度为 hidden_dim (128)，而不是 dna_dim (768)
- 正确传入 `input_dim=4 + cfg.model.hidden_dim` (132)

### 3. 配置访问
- 修复了 separated_modeling 开关的配置访问路径
- 从 `cfg.get("separated_modeling")` 改为 `cfg["train"].get("separated_modeling")`
- 修复了 warmup_steps、amp、use_ema 等参数的访问路径

### 4. EMA 兼容性
- 暂时关闭 EMA 功能，避免模型实例化问题
- 设置 `use_ema: false`

### 5. 数据访问
- 修复了 Sampler 中的 batch 访问方式
- 从 dict 访问改为属性访问：`batch.x0_abun` 而不是 `batch["x0_abun"]`

### 6. 学习率调度
- 修复了学习率调度器的分母问题，防止除零错误
- 使用 `max(1, steps_total - steps_wu)` 避免分母为0

### 7. 训练稳定性
- 降低学习率到 1e-4，提高训练稳定性
- 调整 prefetch_factor 到 2，避免内存问题
- 优化数据加载参数

## 注意事项

1. **数据预处理**: 确保节点特征按正确顺序排列
2. **损失权重**: 根据数据特性调整 `lambda_abun` 和 `lambda_pres`
3. **采样温度**: 调整 `temperature` 参数控制生成的多样性
4. **引导强度**: 使用 `guidance_scale` 提高生成质量，但可能降低多样性
5. **配置路径**: 确保 separated_modeling 参数在 train 节点下

## 扩展功能

1. **多模态条件**: 可扩展支持更多条件信息
2. **自适应权重**: 根据训练进度动态调整损失权重
3. **质量评估**: 添加更多评估指标
4. **可视化**: 添加生成结果的可视化工具 