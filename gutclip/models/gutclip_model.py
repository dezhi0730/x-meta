import math, torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Dict, Any

from .dna_encoder  import DNAEncoder
from .tree_encoder import TreeEncoder


class GutCLIPModel(nn.Module):
    """
    双模态 CLIP 主干
      - TreeEncoder → tree_emb  (B, D)  已 L2-norm
      - DNAEncoder  →  dna_emb  (B, D)  已 L2-norm
      - logit_scale 可训练、并用 soft-clamp 到 [e⁰, e²⁰]
    """
    LOGIT_MAX = math.log(20.0)        # 最大放大 20 倍

    def __init__(self,
                 tree_dim : int = 256,
                 dna_dim  : int = 768,
                 output_dict : bool = False,
                 sparse_aware: bool = False):
        super().__init__()
        self.output_dict  = output_dict
        self.sparse_aware = sparse_aware

        # ---------- encoders ----------
        self.tree_encoder = TreeEncoder(
            input_dim=9, hidden_dim=128, out_dim=tree_dim, num_layers=4
        )
        self.dna_encoder = DNAEncoder(
            input_dim=dna_dim, output_dim=tree_dim
        )

        # ---------- 温度 τ：可训练 ----------
        # 初始放大 ≈ 10，便于收敛；在 forward 中再做 soft-clamp
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    # --------------------------------------------------
    def encode_tree(self, tree: dict, normalize: bool = True):
        z = self.tree_encoder(
            x=tree["x"],
            edge_index=tree["edge_index"],
            pos=tree["pos"],
            batch=tree["batch"]
        )                                   # (B, D)

        # ---------- 稀疏感知权重 ----------
        if self.sparse_aware and 'node_zero' in tree:
            node_zero  = tree['node_zero'].float()
            batch_idx  = tree['batch']
            B = batch_idx.max().item() + 1
            zero_ratio = torch.stack([
                node_zero[batch_idx == i].mean()
                if (batch_idx == i).any() else
                node_zero.new_tensor(0.0)
                for i in range(B)
            ]).unsqueeze(1)
            z = z * (1.0 - zero_ratio * 0.5)

        return F.normalize(z, dim=-1) if normalize else z

    def encode_dna(self,
                   dna: torch.Tensor,
                   pad_mask : Union[torch.Tensor, None],
                   rand_mask: Union[torch.Tensor, None],
                   normalize: bool = True):
        z = self.dna_encoder(dna, pad_mask, rand_mask)
        return F.normalize(z, dim=-1) if normalize else z

    # --------------------------------------------------
    def forward(self, batch) -> Dict[str, Any]:
        # ---- 1) tree ----
        tree_data = {
            "x": batch.x, "edge_index": batch.edge_index,
            "pos": batch.pos, "batch": batch.batch
        }
        if hasattr(batch, 'node_zero'):
            tree_data['node_zero'] = batch.node_zero
        tree_emb = self.encode_tree(tree_data)

        # ---- 2) dna ----
        dna_emb = self.encode_dna(
            batch.dna,
            getattr(batch, "dna_pad_mask",  None),
            getattr(batch, "dna_rand_mask", None)
        )

        # ---- 3) 温度缩放（soft-clamp）----
        logit_scale = self.logit_scale.clamp(0.0, self.LOGIT_MAX)
        scale = logit_scale.exp()           # 标量

        output = {
            "tree_emb":   tree_emb,
            "dna_emb":    dna_emb,
            "logit_scale": logit_scale      # 保留 clamp 后的值
        }
        if hasattr(batch, 'node_zero'):
            output["node_zero"]  = batch.node_zero
            output["zero_ratio"] = batch.zero_ratio

        # 训练阶段
        if self.output_dict:
            return output

        # 推理 / 评估阶段
        logits = scale * (tree_emb @ dna_emb.T)
        assert torch.isfinite(logits).all(), "NaN in logits"
        assert logits.abs().max() < 1e4,    "Out of range in logits"

        return {
            "logits":      logits,
            "logits_t":    logits.T,
            "temperature": scale
        }