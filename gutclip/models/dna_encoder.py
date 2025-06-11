import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=8):
        super().__init__()
        self.query = nn.Linear(input_dim, num_heads)
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, x, mask=None):  # x: (B, N, D), mask: (B, N)
        attn_logit = self.query(x)  # (B, N, H)
        
        # 处理padding mask
        if mask is not None:
            # 扩展mask到与attention heads匹配
            mask = mask.unsqueeze(-1).expand_as(attn_logit)  # (B, N, H)
            attn_logit[~mask] = -1e4  # 把pad位置的attention设为-inf
            
        attn = F.softmax(attn_logit, dim=1)  # (B, N, H)
        pooled = torch.einsum('bnh,bnd->bhd', attn, x)  # (B, H, D)
        return self.proj(pooled.mean(dim=1))  # (B, output_dim)

class DNAEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.2):
        """
        Args:
            input_dim: DNA-BERT输出的单基因Embedding维度
            output_dim: 样本级Embedding输出维度
            dropout_rate: Dropout概率
        """
        super().__init__()
        
        # 正则化层
        self.dropout = nn.Dropout(dropout_rate)

        # 固定降维到256
        reduced_dim = 256
        self.pool = AttentionPooling(input_dim, reduced_dim, num_heads=8)
        
        # 添加BatchNorm
        self.bn = nn.BatchNorm1d(reduced_dim)
        
        # 投影层 - 添加非线性
        self.proj = nn.Sequential(
            nn.Linear(reduced_dim, reduced_dim),
            nn.GELU(),
            nn.Linear(reduced_dim, output_dim)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        输入: 
            x: (B, N, D) 或 (B, D)
                B = batch size, 
                N = 基因数量, 
                D = 基因Embedding维度
            mask: (B, N) 可选，表示哪些位置是有效基因（非padding）
        输出: 
            (B, output_dim)
        """
        # Ensure input is float32 to match model parameters
        x = x.float()
        
        # 如果是 3D 输入，先做attention pooling
        if x.dim() == 3:
            x = self.pool(x, mask)  # (B, reduced_dim)
            
        # 应用BatchNorm
        x = self.bn(x)
            
        # 随机丢弃部分特征
        x = self.dropout(x)  # (B, reduced_dim)
        
        # 投影
        x = self.proj(x)  # (B, output_dim)
        
        return x