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

        self.tree_encoder = TreeEncoder(input_dim=1, hidden_dim=128,
                                        out_dim=tree_dim, num_layers=4)
        self.dna_encoder = DNAEncoder(input_dim=dna_dim,
                                      output_dim=dna_dim)

        self.proj_tree = nn.Linear(tree_dim, embed_dim)
        self.proj_dna  = nn.Linear(dna_dim,  embed_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    # --------------------------------------------------
    def encode_tree(self, tree, normalize=True):
        x = self.tree_encoder(**tree)          # (B, tree_dim)
        x = self.proj_tree(x)
        return F.normalize(x, dim=-1) if normalize else x

    def encode_dna(self, dna, normalize=True):
        x = self.dna_encoder(dna)              # (B, dna_dim)
        x = self.proj_dna(x)
        return F.normalize(x, dim=-1) if normalize else x

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

        # 编码
        tree_feat = self.encode_tree(tree_data, normalize=True)
        dna_feat  = self.encode_dna(dna_data, normalize=True)

        # (B, B) 相似度
        logits_per_tree = self.logit_scale.exp() * tree_feat @ dna_feat.T
        logits_per_dna  = logits_per_tree.T

        if self.output_dict:
            return {
                "tree_emb": tree_feat,
                "dna_emb":  dna_feat,
                "logit_scale": self.logit_scale.exp(),
                "logits_per_tree": logits_per_tree,   # 可选，调试用
                "logits_per_dna": logits_per_dna,
            }

        return logits_per_tree, logits_per_dna, self.logit_scale.exp()