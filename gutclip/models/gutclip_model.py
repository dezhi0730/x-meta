# gutclip/models/gutclip_model.py
# -*- coding: utf-8 -*-
import torch, torch.nn as nn, torch.nn.functional as F
from typing import Union
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
                 output_dict: bool = False):
        super().__init__()
        self.output_dict = output_dict

        # ---------- encoders ----------
        self.tree_encoder = TreeEncoder(
            input_dim=8, hidden_dim=128, out_dim=tree_dim, num_layers=4
        )
        self.dna_encoder = DNAEncoder(
            input_dim=dna_dim, output_dim=tree_dim
        )

        # ---------- 温度 τ 固定 ----------
        self.register_buffer("logit_scale", torch.tensor(1/0.07))   # τ = 2

    # --------------------------------------------------
    def encode_tree(self, tree: dict, normalize: bool = True):
        z = self.tree_encoder(**tree)            # (B, D)
        return F.normalize(z, dim=-1) if normalize else z

    def encode_dna(self,
                   dna: torch.Tensor,               # (B,L,768)
                   pad_mask: Union[torch.Tensor, None],   # (B,L) True=valid
                   rand_mask: Union[torch.Tensor, None],  # (B,L) True=random_mask
                   normalize: bool = True):
        z = self.dna_encoder(dna, pad_mask, rand_mask)   # (B,D)
        return F.normalize(z, dim=-1) if normalize else z

    # --------------------------------------------------
    def forward(self, batch):
        # ---- 1) tree encoder ----
        tree_data = {
            "x"        : batch.x,
            "edge_index": batch.edge_index,
            "pos"      : batch.pos,
            "batch"    : batch.batch,
        }
        tree_emb = self.encode_tree(tree_data)

        # ---- 2) dna encoder ----
        dna_emb = self.encode_dna(
            batch.dna,                 # (B,L,768)
            getattr(batch, "dna_pad_mask", None),
            getattr(batch, "dna_rand_mask", None)
        )

        # ---- 3) 输出 ----
        if self.output_dict:           # 训练阶段
            return {
                "tree_emb"   : tree_emb,
                "dna_emb"    : dna_emb,
                "logit_scale": self.logit_scale.clamp(0, 4.6),   # exp(4.6)≈100,   # τ=2
            }

        # 推理 / 评估阶段：返回相似度 logits
        logits = torch.exp(self.logit_scale) * tree_emb @ dna_emb.T
        return logits, logits.T, torch.exp(self.logit_scale)