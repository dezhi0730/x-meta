import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=8):
        super().__init__()
        self.query = nn.Linear(input_dim, num_heads)
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):  # x: (B, N, D)
        attn = F.softmax(self.query(x), dim=1)  # (B, N, H)
        pooled = torch.einsum('bnh,bnd->bhd', attn, x)  # (B, H, D)
        return self.proj(pooled.mean(dim=1))  # (B, output_dim)

class DNAEncoder(nn.Module):
    def __init__(self, input_dim=768, num_heads=8, output_dim=1280, 
                 dropout_rate=0.1, use_norm=True):
        """
        Args:
            input_dim: DNA-BERT输出的单基因Embedding维度
            num_heads: 注意力头数
            output_dim: 样本级Embedding输出维度
            dropout_rate: Dropout概率
            use_norm: 是否使用LayerNorm
        """
        super().__init__()
        self.use_norm = use_norm
        
        # 归一化层 (基因级别)
        if use_norm:
            self.norm = nn.LayerNorm(input_dim)  # 对每个基因的D维做归一化
        
        # 正则化层
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # 注意力池化层
        self.pooler = AttentionPooling(
            input_dim=input_dim, 
            num_heads=num_heads,
            output_dim=output_dim
        )

    def forward(self, embedding_tensor):
        """
        输入: 
            embedding_tensor: (B, N, D) 
                B = batch size, 
                N = 基因数量, 
                D = 基因Embedding维度
        输出: 
            (B, output_dim)
        """
        # 1. 基因级别归一化
        if self.use_norm:
            x = self.norm(embedding_tensor)  # (B, N, D)
        else:
            x = embedding_tensor
            
        # 2. 随机丢弃部分基因特征
        x = self.dropout(x)  # (B, N, D)
        
        # 3. 注意力池化
        return self.pooler(x)  # (B, output_dim)