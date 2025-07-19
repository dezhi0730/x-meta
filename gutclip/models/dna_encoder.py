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

            # 根据数据类型设置合适的负无穷值
            neg_inf = torch.finfo(attn_logits.dtype).min
            attn_logits = attn_logits.masked_fill(~mask, neg_inf)

        # ------- 归一化权重 -------
        attn = F.softmax(attn_logits, dim=1)                 # (B, N, H)
        attn = attn / (D ** 0.5)                             # 缩放抑制方差

        # ------- 加权求和 -------
        pooled = torch.einsum('bnh,bnd->bhd', attn, x)       # (B, H, D)
        
        # 使用多头注意力的加权池化，而不只是简单平均
        head_weights = F.softmax(torch.mean(attn, dim=1), dim=-1).unsqueeze(1)  # (B, 1, H)
        pooled = torch.bmm(head_weights, pooled).squeeze(1)  # (B, D)
        
        return self.proj(pooled)                              # (B, output_dim)

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

        # 使用序列式结构，便于灵活调整
        self.transform = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(reduced_dim, reduced_dim),
            nn.GELU(),
            nn.LayerNorm(reduced_dim),  # 将LayerNorm移至非线性激活后
            nn.Dropout(dropout_rate),
            nn.Linear(reduced_dim, output_dim),
            nn.LayerNorm(output_dim)    # 添加额外的LayerNorm稳定输出
        )

    def forward(self,
                x: torch.Tensor,                   # (B, L, 768)
                pad_mask: Optional[torch.Tensor] = None,   # (B, L)
                rand_mask: Optional[torch.Tensor] = None   # (B, L)
               ) -> torch.Tensor:
        
        # 检查输入维度
        assert x.dim() in [2, 3], f"Input tensor must be 2D or 3D, got {x.dim()}D"
        
        # 1) 改进的随机掩码策略 - 使用dropout而不是直接置零
        if rand_mask is not None:
            # 创建一个可学习的掩码权重，防止完全丢弃信息
            mask_weight = nn.Parameter(torch.ones(1, device=x.device), requires_grad=True)
            x = x.clone()
            x[rand_mask] *= torch.sigmoid(mask_weight)  # 使用sigmoid确保权重在(0,1)之间

        # 2) AttentionPooling 跳过 <PAD> 位置
        if x.dim() == 3:
            x = self.pool(x.float(), pad_mask)     # (B, 256)
        else:                                      # 万一提前已 pool
            x = x.float()

        # 3) 转换 & 归一化
        x = self.transform(x)
        return F.normalize(x, dim=-1)               # 保证特征范数 = 1