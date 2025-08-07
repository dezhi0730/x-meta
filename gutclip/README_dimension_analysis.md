# Separated Diffusion Model 维度分析

## 1. 数据加载阶段

### 1.1 TreeDiffusionDataset.__getitem__
```python
# 输入：单个样本
x_abun   = tree.x[:, [2]].float()     # (N, 1) log_abun
x_pres   = tree.x[:, [1]].long()      # (N, 1) presence 0/1
x_static = tree.x[:, [3,8]].float()   # (N, 2) depth_sc, degree_sc
dna      = dna_rec["embedding"].float() # (L_i, 768) DNA嵌入

# 输出：Data对象
data = Data(
    x0_abun   = x_abun,          # (N, 1) log_abun
    x0_pres   = x_pres,          # (N, 1) presence 0/1
    x_static  = x_static,        # (N, 2) 静态特征
    pos       = tree.pos,         # (N, 3) 位置特征
    edge_index = tree.edge_index, # (2, E) 边索引
    batch     = torch.zeros(N),   # (N,) batch索引
    dna       = dna,              # (L_i, 768) DNA嵌入
    sample_id = sid,              # str
)
```

### 1.2 GaussianDiffusionCollate.__call__
```python
# 输入：List[Data] (batch_size个样本)
# 输出：Batch对象

# 1) Graph merge
pyg = Batch.from_data_list(batch)  # 合并所有图

# 2) DNA处理
dna_tensor = torch.stack(pad_dnas, 0)  # (B, L_max, 768)
pad_mask   = torch.stack(pad_masks, 0) # (B, L_max)

# 3) 加噪处理
B = len(batch)
t_idx = torch.randint(0, T, (B,))  # (B,) 时间步索引

# 连续流：log_abun
pyg.x_t, pyg.noise = self._add_noise(pyg.x0_abun, t_idx, pyg.batch)
# pyg.x_t: (ΣN, 1) 加噪后的log_abun
# pyg.noise: (ΣN, 1) 真实噪声ε

# 离散流：presence
pyg.x_t_pres = self._bernoulli_noisy(pyg.x0_pres, t_idx, pyg.batch)
# pyg.x_t_pres: (ΣN, 1) 加噪后的presence

# mask特征
pyg.mask_feat = pyg.x_t_pres.float()  # (ΣN, 1) 显式mask特征

# 最终batch包含：
# - x_t: (ΣN, 1) 加噪后的log_abun
# - x0_abun: (ΣN, 1) 原始log_abun
# - x0_pres: (ΣN, 1) 原始presence
# - x_static: (ΣN, 2) 静态特征
# - mask_feat: (ΣN, 1) mask特征
# - noise: (ΣN, 1) 真实噪声
# - dna: (B, L_max, 768) DNA嵌入
# - edge_index: (2, E) 边索引
# - pos: (ΣN, 3) 位置特征
# - batch: (ΣN,) batch索引
# - t_idx: (B,) 时间步索引
```

## 2. 模型前向传播

### 2.1 SeparatedDiffusionModel (基础版)
```python
# 输入维度分析
x_t      = batch.x_t       # (ΣN, 1) 加噪后的log_abun
x_static = batch.x_static  # (ΣN, 2) 静态特征
mask_feat = batch.mask_feat # (ΣN, 1) mask特征

# 特征拼接
x_comb = torch.cat([x_t, x_static, mask_feat], dim=1)  # (ΣN, 4)

# 树编码器
h = self.tree_encoder(x_comb, edge_index, pos, batch)  # (ΣN, out_dim)

# 时间条件化
t_emb = self.time_embed(t_idx.float().unsqueeze(-1))  # (B, hidden_dim)
t_cond = self.time_condition(t_emb)  # (B, out_dim)
t_cond_nodes = t_cond[batch.batch]  # (ΣN, out_dim)
h = h + t_cond_nodes  # (ΣN, out_dim)

# 分离头
eps_hat = self.abun_head(h).squeeze(-1)      # (ΣN,) 连续ε̂
pres_logit = self.pres_head(h).squeeze(-1)   # (ΣN,) presence logits

# 输出
return {
    "eps_hat": eps_hat,      # (ΣN,) 预测的噪声
    "pres_logit": pres_logit # (ΣN,) presence的logits
}
```

### 2.2 SeparatedDiffusionModelWithDNA (带DNA条件)
```python
# 输入维度分析
x_t      = batch.x_t       # (ΣN, 1) 加噪后的log_abun
x_static = batch.x_static  # (ΣN, 2) 静态特征
mask_feat = batch.mask_feat # (ΣN, 1) mask特征
dna      = batch.dna       # (B, L_max, 768) DNA嵌入

# DNA处理
dna_pooled = batch.dna.mean(dim=1)  # (B, 768) 池化
batch_indices = batch.batch.long()   # (ΣN,) batch索引
dna_nodes = dna_pooled[batch_indices]  # (ΣN, 768) 扩展到节点级别
dna_encoded = self.dna_encoder(dna_nodes)  # (ΣN, hidden_dim)

# 特征拼接
x_comb = torch.cat([x_t, x_static, mask_feat, dna_encoded], dim=1)  # (ΣN, 4+hidden_dim)

# 树编码器 (input_dim = 4 + hidden_dim)
h = self.tree_encoder(x_comb, edge_index, pos, batch)  # (ΣN, out_dim)

# 时间条件化 (同基础版)
t_emb = self.time_embed(t_idx.float().unsqueeze(-1))  # (B, hidden_dim)
t_cond = self.time_condition(t_emb)  # (B, out_dim)
t_cond_nodes = t_cond[batch.batch]  # (ΣN, out_dim)
h = h + t_cond_nodes  # (ΣN, out_dim)

# 分离头 (同基础版)
eps_hat = self.abun_head(h).squeeze(-1)      # (ΣN,) 连续ε̂
pres_logit = self.pres_head(h).squeeze(-1)   # (ΣN,) presence logits

# 输出
return {
    "eps_hat": eps_hat,      # (ΣN,) 预测的噪声
    "pres_logit": pres_logit # (ΣN,) presence的logits
}
```

## 3. 损失函数

### 3.1 _separated_loss_fn
```python
# 输入
x0_abun = batch.x0_abun           # (ΣN, 1) 原始log_abun
x0_pres = batch.x0_pres           # (ΣN, 1) 原始presence
noise = batch.noise               # (ΣN, 1) 真实噪声
eps_hat = model_output["eps_hat"]      # (ΣN,) 预测的噪声
pres_logit = model_output["pres_logit"]   # (ΣN,) presence的logits

# Presence BCE
loss_pres = F.binary_cross_entropy_with_logits(
    pres_logit, x0_pres.squeeze(-1).float())  # scalar

# Abundance MSE (只在presence=1的节点)
mask = x0_pres.squeeze(-1).float()  # (ΣN,) presence mask
loss_abun = ((eps_hat - noise.squeeze(-1))**2 * mask).sum() / mask.sum().clamp_min(1)  # scalar

# 总损失
loss = lambda_pres * loss_pres + lambda_abun * loss_abun  # scalar
```

## 4. 维度总结

### 4.1 关键维度
- **基础输入维度**: 4 (x_t + x_static + mask_feat)
- **DNA编码后维度**: 256 (与预训练模型对齐)
- **带DNA的输入维度**: 4 + 256 = 260
- **树编码器输出维度**: out_dim (256)
- **时间嵌入维度**: hidden_dim (128)
- **时间条件维度**: out_dim (256)

### 4.2 模型配置
```python
# 基础模型
SeparatedDiffusionModel(
    input_dim=4,        # x_t(1) + x_static(2) + mask_feat(1)
    hidden_dim=128,
    out_dim=256,
    num_layers=4,
    dropout_rate=0.25
)

# 带DNA模型
SeparatedDiffusionModelWithDNA(
    input_dim=4,        # 基础输入维度
    dna_dim=768,        # DNA嵌入维度
    hidden_dim=128,     # DNA编码后维度
    out_dim=256,        # 树编码器输出维度
    num_layers=4,
    dropout_rate=0.25
)
# 实际输入维度: 4 + 128 = 132 (在__init__中计算)
```

### 4.3 数据流维度
```
原始数据: (N, 1) log_abun + (N, 1) presence + (N, 2) static
加噪后: (ΣN, 1) x_t + (ΣN, 1) mask_feat + (ΣN, 2) x_static
拼接: (ΣN, 4) 基础特征
DNA编码: (ΣN, 128) DNA特征 (如果启用)
最终输入: (ΣN, 4) 或 (ΣN, 132) 到树编码器
树编码器输出: (ΣN, 256)
时间条件: (ΣN, 256)
分离头输出: (ΣN,) eps_hat + (ΣN,) pres_logit
```

## 5. 潜在问题

### 5.1 维度不匹配问题
- **问题**: 训练脚本中 `input_dim=4 + cfg.model.hidden_dim` (132)
- **原因**: 模型内部又加了 `hidden_dim`，导致最终维度为 260
- **解决**: 训练脚本中应该使用 `input_dim=4`，让模型内部处理DNA编码

### 5.2 DNA处理问题
- **问题**: DNA池化可能不适用于图结构
- **建议**: 考虑使用注意力机制或图池化

### 5.3 时间条件化问题
- **问题**: 时间条件直接加到节点特征上
- **建议**: 考虑使用FiLM或交叉注意力 