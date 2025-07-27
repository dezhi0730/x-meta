"""
check_prior.py  —— 评估 y_prior 与 y_true 的 Spearman / Bray-Curtis
"""

import torch, numpy as np
from gutclip.engine.trainer_diffusion import RetrievalIndex
from scipy.stats import spearmanr
from scipy.spatial.distance import braycurtis

# 1) 读你离线导出的 valid  embedding
obj = torch.load("datasets/diffusion/V3/valid_embeddings.pt", map_location="cpu")
z_all = obj["z"]          # (B,D)  torch.FloatTensor
y_all = obj["y"]          # (B,N)  torch.FloatTensor

# 2) 加载索引
ret = RetrievalIndex(index_file="datasets/diffusion/V3/faiss_index.faiss",
                     y_file    ="datasets/diffusion/V3/faiss_index.y.npy",
                     gpu=0, k=5)

# 3) 分 batch 计算先验
batch_sz = 512
rhos, bcs = [], []
for i in range(0, len(z_all), batch_sz):
    z = z_all[i:i+batch_sz].cuda()
    y_true  = y_all[i:i+batch_sz]
    y_prior = ret.query(z).cpu()

    for yp, yt in zip(y_prior, y_true):
        rhos.append( spearmanr(yp, yt).correlation )
        bcs .append( braycurtis(yp, yt) )

print(f"Spearman ρ̄ = {np.nanmean(rhos):.3f} | Bray-Curtis ḃ = {np.mean(bcs):.3f}")