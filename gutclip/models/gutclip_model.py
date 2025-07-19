# -*- coding: utf-8 -*-
import torch, torch.nn as nn, torch.nn.functional as F
from typing import Union, Dict, Any
from .dna_encoder  import DNAEncoder
from .tree_encoder import TreeEncoder


class GutCLIPModel(nn.Module):
    """
    双模态 CLIP 主干
      - TreeEncoder → tree_emb  (B, D)  已 L2-norm
      - DNAEncoder  →  dna_emb  (B, D)  已 L2-norm
      - logit_scale 固定常数（不训练）
    """
    def __init__(self,
                 tree_dim : int = 256,
                 dna_dim  : int = 768,
                 output_dict: bool = False,
                 sparse_aware: bool = False):  # 新增：稀疏感知编码
        super().__init__()
        self.output_dict = output_dict
        self.sparse_aware = sparse_aware  # 控制是否使用稀疏感知编码

        # ---------- encoders ----------
        self.tree_encoder = TreeEncoder(
            input_dim=9, hidden_dim=128, out_dim=tree_dim, num_layers=4  # 输入维度从8→9
        )
        self.dna_encoder = DNAEncoder(
            input_dim=dna_dim, output_dim=tree_dim
        )

        # ---------- 温度 τ 固定 ----------
        self.register_buffer("logit_scale", torch.tensor(1/0.07))

    # --------------------------------------------------
    def encode_tree(self, tree: dict, normalize: bool = True):
        """编码树结构，支持稀疏感知"""
        # 提取TreeEncoder需要的参数
        tree_encoder_input = {
            "x": tree["x"],
            "edge_index": tree["edge_index"], 
            "pos": tree["pos"],
            "batch": tree["batch"]
        }
        
        z = self.tree_encoder(**tree_encoder_input)  # (B, D)
        
        # 可选：应用稀疏感知掩码
        if self.sparse_aware and 'node_zero' in tree:
            # 获取样本级0丰度比例
            node_zero = tree['node_zero'].float()  # (总节点数,)
            batch_idx = tree['batch']  # (总节点数,) 每个节点属于哪个样本
            
            # 为每个样本计算0丰度比例
            batch_size = batch_idx.max().item() + 1
            zero_ratios = []
            
            for i in range(batch_size):
                mask = (batch_idx == i)
                if mask.any():
                    sample_zero_ratio = node_zero[mask].mean()
                    zero_ratios.append(sample_zero_ratio)
                else:
                    zero_ratios.append(torch.tensor(0.0, device=node_zero.device))
            
            zero_ratio = torch.stack(zero_ratios).unsqueeze(1)  # (B, 1)
            
            # 根据0丰度比例调整嵌入
            z = z * (1.0 - zero_ratio * 0.5)  # 降低高0丰度样本的权重
            
        return F.normalize(z, dim=-1) if normalize else z

    def encode_dna(self,
                   dna: torch.Tensor,               # (B,L,768)
                   pad_mask: Union[torch.Tensor, None],   # (B,L) True=valid
                   rand_mask: Union[torch.Tensor, None],  # (B,L) True=random_mask
                   normalize: bool = True):
        z = self.dna_encoder(dna, pad_mask, rand_mask)   # (B,D)
        return F.normalize(z, dim=-1) if normalize else z

    # --------------------------------------------------
    def forward(self, batch) -> Dict[str, Any]:
        # ---- 1) tree encoder ----
        tree_data = {
            "x"        : batch.x,
            "edge_index": batch.edge_index,
            "pos"      : batch.pos,
            "batch"    : batch.batch,
        }
        
        # 传递节点级0丰度信息（如果有）
        if hasattr(batch, 'node_zero'):
            tree_data['node_zero'] = batch.node_zero
            
        tree_emb = self.encode_tree(tree_data)

        # ---- 2) dna encoder ----
        dna_emb = self.encode_dna(
            batch.dna,                 # (B,L,768)
            getattr(batch, "dna_pad_mask", None),
            getattr(batch, "dna_rand_mask", None)
        )

        # ---- 3) 输出 ----
        output = {
            "tree_emb"   : tree_emb,
            "dna_emb"    : dna_emb,
            "logit_scale": self.logit_scale.clamp(0, 4.6),
        }
        
        # 可选：传递节点级0丰度信息
        if hasattr(batch, 'node_zero'):
            output['node_zero'] = batch.node_zero
            output['zero_ratio'] = batch.zero_ratio  # 样本级0丰度比例
            
        if self.output_dict:  # 训练阶段
            return output

        # 推理 / 评估阶段：返回相似度 logits
        logits = torch.exp(self.logit_scale) * tree_emb @ dna_emb.T
        assert torch.isfinite(logits).all(), "NaN in logits"
        assert logits.abs().max() < 1e4, "Out of range in logits"
        return {
            "logits": logits,
            "logits_t": logits.T,
            "temperature": torch.exp(self.logit_scale)
        }