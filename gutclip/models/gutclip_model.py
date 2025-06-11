import torch
import torch.nn as nn
import torch.nn.functional as F

from .dna_encoder import DNAEncoder
from .tree_encoder import TreeEncoder

class GutCLIPModel(nn.Module):
    def __init__(self, embed_dim: int = 1280,
                 tree_dim: int = 256, dna_dim: int = 768,
                 output_dict: bool = False):
        super().__init__()
        self.output_dict = output_dict

        self.tree_encoder = TreeEncoder(input_dim=6, hidden_dim=128,
                                out_dim=tree_dim, num_layers=4)
        self.dna_encoder = DNAEncoder(input_dim=dna_dim,
                                      output_dim=tree_dim)

        self.logit_scale = nn.Parameter(torch.tensor([2.0]))

    # --------------------------------------------------
    def encode_tree(self, tree, normalize=True):
        x = self.tree_encoder(**tree)          # (B, tree_dim)
        return x if not normalize else F.normalize(x, dim=-1)

    def encode_dna(self, dna, mask=None, normalize=True):
        x = self.dna_encoder(dna, mask)              # (B, tree_dim)
        return x if not normalize else F.normalize(x, dim=-1)

    # --------------------------------------------------
    def forward(self, batch):
        """
        Args:
            batch: PyG Batch object containing:
                - x: node features
                - edge_index: edge indices
                - pos: node positions
                - batch: batch indices
                - dna: DNA embeddings
                - dna_mask: (B, N) mask for DNA padding
        """
        # 提取树数据
        tree_data = {
            "x": batch.x,
            "edge_index": batch.edge_index,
            "pos": batch.pos,
            "batch": batch.batch
        }
        
        # 提取 DNA 数据
        dna_data = batch.dna
        dna_mask = getattr(batch, 'dna_mask', None)  # 如果没有mask就返回None

        # 编码 - 移除早期归一化
        tree_feat = self.encode_tree(tree_data, normalize=False)
        dna_feat  = self.encode_dna(dna_data, dna_mask, normalize=False)

        # 在forward时做一次归一化
        tree_feat = F.normalize(tree_feat, dim=-1)
        dna_feat  = F.normalize(dna_feat, dim=-1)

        if self.output_dict:
            return {
                "tree_emb": tree_feat,
                "dna_emb":  dna_feat,
                "logit_scale": self.logit_scale,  # 返回原始参数，不 exp
            }

        # 计算相似度
        logits_per_tree = self.logit_scale.exp() * tree_feat @ dna_feat.T
        logits_per_dna  = logits_per_tree.T
        return logits_per_tree, logits_per_dna, self.logit_scale.exp()