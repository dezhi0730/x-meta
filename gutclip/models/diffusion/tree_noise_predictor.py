
import torch
import torch.nn as nn
import math
from torch_geometric.data import Batch
from gutclip.models.tree_encoder import TreeEncoder 
from gutclip.models.dna_encoder  import DNAEncoder
import torch.nn.functional as F

def get_timestep_embedding(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    """Sinusoidal position embedding (same as in DDPM / Transformer)."""
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, dtype=torch.float32, device=t.device)
        * -(math.log(10000.0) / half)
    )  # (half,)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        # Zero‑pad for odd dim sizes
        emb = F.pad(emb, (0, 1))
    return emb


class TreeNoisePredictor(nn.Module):
    """Noise‑prediction network with FiLM conditioning by (time + DNA)."""

    LOGIT_MAX = math.log(20.0)

    def __init__(
        self,
        node_dim: int = 2,
        hid: int = 128,
        t_emb_dim: int = 128,
        dna_dim: int = 768,
    ) -> None:
        super().__init__()
        
        # Tree encoder for node-level processing
        self.tree_enc = TreeEncoder(
            input_dim = 4,           # 2 noisy + 4 static
            hidden_dim = hid,
            out_dim = hid,           # 节点隐维
            num_layers = 4,
            return_node_emb = True   # ★关键★
        )
        
        self.dna_enc = DNAEncoder(input_dim=dna_dim, output_dim=hid * 2)

        self.time_mlp = nn.Sequential(
            nn.Linear(t_emb_dim, hid * 4), nn.SiLU(),
            nn.Linear(hid * 4, hid * 2)
        )  # → γ_t, β_t

        self.out_proj = nn.Linear(hid, node_dim)

    # --------------------------------------------------
    def forward(self, batch: Batch):
        # 1) DNA → γ_d, β_d  (B, hid)
        z_dna = self.dna_enc(batch.dna, batch.dna_pad_mask, batch.dna_rand_mask)
        γ_d, β_d = z_dna.chunk(2, -1)

        # 2) time embedding
        t_emb = get_timestep_embedding(batch.t_idx, dim=128)  # (B,128)
        γ_t, β_t = self.time_mlp(t_emb).chunk(2, -1)

        γ = torch.sigmoid(γ_d + γ_t)  # (B,hid)
        β = β_d + β_t                 # (B,hid)

        # 3) Tree encoder with concatenated noisy and static features
        # 我们需要节点级别的特征，所以需要修改PhyloEGNN来返回节点特征而不是图特征
        x_combined = torch.cat([batch.x_t, batch.x_static], dim=1)  # (ΣN,6)
        
        h = self.tree_enc(               # ← 直接返回 (ΣN, hid)
            x_combined,
            batch.edge_index,
            batch.pos,
            batch.batch
        )
        # 现在h是节点级别的特征 (ΣN, hid)
        
        # FiLM modulation (broadcast via batch.batch index)
        # γ and β have shape (B, hid), batch.batch has shape (total_nodes,)
        # We need to index γ and β with batch.batch to get (total_nodes, hid)
        batch_indices = batch.batch
        γ_expanded = γ[batch_indices]  # (total_nodes, hid)
        β_expanded = β[batch_indices]  # (total_nodes, hid)
        
        h = h * γ_expanded + β_expanded

        return self.out_proj(h)       # 输出维度 = 2，匹配 noisy 列 