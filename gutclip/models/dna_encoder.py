import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.query = nn.Linear(input_dim, num_heads, bias=False)
        self.proj  = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = x.shape
        attn_logits = self.query(x)                          # (B, N, H)

        # ------- 处理 padding -------
        if mask is not None:
            mask = mask.bool().unsqueeze(-1).expand_as(attn_logits)  # (B, N, H)

            # 若某些样本整行全 pad → 至少保留第 0 列，避免 NaN
            all_pad = (~mask).all(dim=1)                     # (B, H)
            if all_pad.any():
                mask[all_pad, 0, :] = True

            # 使用适合半精度浮点数的值，避免溢出
            neg_inf = torch.finfo(attn_logits.dtype).min
            attn_logits = attn_logits.masked_fill(~mask, neg_inf)

        # ------- 归一化权重 -------
        attn = F.softmax(attn_logits, dim=1)                 # (B, N, H)
        attn = attn / (D ** 0.5)                             # 缩放抑制方差

        # ------- 加权求和 -------
        pooled = torch.einsum('bnh,bnd->bhd', attn, x)       # (B, H, D)
        return self.proj(pooled.mean(dim=1))                 # (B, output_dim)

class DNAEncoder(nn.Module):
    """
    Args:
        input_dim   : DNA-BERT 每条基因的 embedding 维度
        output_dim  : 输出到 CLIP 的维度（与 TreeEncoder 对齐）
        dropout_rate: Dropout 概率
    """
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.1):
        super().__init__()

        reduced_dim = 256                              # 统一降到 256
        self.pool = AttentionPooling(input_dim, reduced_dim, num_heads=8)

        self.norm = nn.LayerNorm(reduced_dim)          # 替换 BatchNorm → 更稳

        self.proj = nn.Sequential(                     # 内部自带 Dropout
            nn.Dropout(dropout_rate),
            nn.Linear(reduced_dim, reduced_dim),
            nn.GELU(),
            nn.Linear(reduced_dim, output_dim),
        )

    def forward(self,
                x: torch.Tensor,                   # (B, L, 768)
                pad_mask: Optional[torch.Tensor] = None,   # (B, L)
                rand_mask: Optional[torch.Tensor] = None   # (B, L)
               ) -> torch.Tensor:
        # 1) 随机遮掉的 token 置零
        if rand_mask is not None:
            x = x.clone()
            x[rand_mask] = 0.0

        # 2) AttentionPooling 跳过 <PAD> 位置
        if x.dim() == 3:
            x = self.pool(x.float(), pad_mask)     # (B, 256)
        else:                                      # 万一提前已 pool
            x = x.float()

        # 3) 归一化 & 投影
        x = self.norm(x)
        x = self.proj(x)
        return F.normalize(x, dim=-1)               # 保证特征范数 = 1