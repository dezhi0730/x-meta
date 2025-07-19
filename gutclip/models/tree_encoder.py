import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GlobalAttention, GraphConv
from torch.nn import Linear, SiLU

def center_rms(pos: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """把坐标质心移到 0，并把平均半径缩放到 1"""
    pos = pos - pos.mean(0, keepdim=True)          # 质心置零
    rms = pos.pow(2).sum(-1).mean().sqrt()
    return pos / rms.clamp_min(eps)

# ---------- 1. 改进的坐标归一化器 ----------
class CoorsNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=0.2, max_scale=5.0):
        super().__init__()
        self.eps = eps
        self.max_scale = max_scale
        self.scale = nn.Parameter(torch.tensor([scale_init], dtype=torch.float32))

    def forward(self, c):
        s = self.scale.clamp(0, self.max_scale)
        norm = c.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        return c / norm * s


# ---------- 2. 双向坐标更新 EGCL ----------
class PhyloEGCL(MessagePassing):
    def __init__(self, h, edge_dim=3, update_coords=True, aggr="add"):
        super().__init__()
        self.update_coords, self.aggr = update_coords, aggr

        if update_coords:
            self.coord_mlp = nn.Sequential(
                Linear(2*h + edge_dim, 2*h), SiLU(),
                Linear(2*h, h),           SiLU(),
                Linear(h, 3)
            )
            self.edge_w = nn.Sequential(
                Linear(3, h), SiLU(),
                Linear(h, 1), nn.Sigmoid()
            )
            self.coors_norm = CoorsNorm()

        self.edge_mlp = nn.Sequential(
            Linear(2*h + edge_dim, 2*h), SiLU(),
            nn.LayerNorm(2*h),
            Linear(2*h, h),             SiLU(),
            nn.LayerNorm(h)
        )
        self.node_mlp = nn.Sequential(
            Linear(2*h, 2*h), SiLU(),
            nn.LayerNorm(2*h),
            Linear(2*h, h)
        )

        self.res_drop = nn.Dropout(0.15)
        self.ln = nn.LayerNorm(h)

    # -------------- helpers ----------------
    def _rel(self, p, r, c):
        d = p[r] - p[c]
        d = torch.nan_to_num(d, nan=0.0, posinf=1e4, neginf=-1e4)
        norm = d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return d, norm

    # -------------- forward ----------------
    def forward(self, x, edge_index, pos):
        row, col = edge_index
        assert torch.isfinite(x).all(), "[NaN] in-x"
        # ---- 1. 坐标更新 ----
        if self.update_coords:
            rel, _ = self._rel(pos, row, col)

            raw_fwd = self.coord_mlp(torch.cat([x[row], x[col],  rel], -1))
            raw_bwd = self.coord_mlp(torch.cat([x[col], x[row], -rel], -1))
            assert torch.isfinite(pos).all(), "[NaN] pos after Δ-update"

            # 步长 ≤ 5 % 当前边长
            edge_len = rel.norm(dim=-1, keepdim=True).detach()
            delta_fwd = self.coors_norm(torch.tanh(raw_fwd)) * 0.05 * edge_len
            delta_bwd = self.coors_norm(torch.tanh(raw_bwd)) * 0.05 * edge_len

            delta_fwd = delta_fwd * self.edge_w(rel)
            delta_bwd = delta_bwd * self.edge_w(-rel)

            pos = pos.clone()
            pos.index_add_(0, row, delta_fwd.to(pos.dtype))
            pos.index_add_(0, col, delta_bwd.to(pos.dtype))

        # ---- 2. 消息 ----
        rel, _ = self._rel(pos, row, col)
        msg = self.edge_mlp(torch.cat([x[row], x[col], rel], -1))
        assert torch.isfinite(msg).all(), "[NaN] msg"

        # ---- 3. 聚合 ----
        if self.aggr == "add":
            agg = torch.zeros_like(x)
            agg.index_add_(0, row, msg.to(agg.dtype))
        elif self.aggr == "mean":
            from torch_geometric.utils import scatter_mean
            agg = scatter_mean(msg, row, dim=0, dim_size=x.size(0))
        else:
            raise ValueError("Unsupported aggregation")

        # ---- 4. 更新节点 ----
        mixed = torch.cat([x, agg], -1)

        assert torch.isfinite(x).all(),   "[NaN] x-before-mix"
        assert torch.isfinite(agg).all(), "[NaN] agg-before-mix"

        x = x + self.res_drop(self.node_mlp(mixed))
        assert torch.isfinite(x).all(), "[NaN] x-after-nodeMLP"
        x = self.ln(x)

        # 新增：LayerNorm 后
        assert torch.isfinite(x).all(), "[NaN] x-after-LayerNorm"

        return x, pos

# ---------- 3. 增强的EGNN网络 ----------
class PhyloEGNN(nn.Module):
    def __init__(self, in_dim=9, h=128, out_dim=256, L=4, aggr="add", dropout=0.15):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, h),
                                  nn.LayerNorm(h), nn.GELU())

        self.layers = nn.ModuleList([
            PhyloEGCL(h, aggr=aggr) if i % 2 == 0 else GraphConv(h, h)
            for i in range(L)
        ])

        self.pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(h, h), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(h, 1)
            )
        )
        self.out = nn.Sequential(
            nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(h, out_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _norm_pos(self, p):  # 每层坐标归一化
        return center_rms(p)

    def forward(self, x, pos, edge_index, batch):
        x   = self.proj(x)
        pos = self._norm_pos(pos)          # 入口统一量纲

        assert torch.isfinite(x).all(),   "[NaN] after proj-MLP"
        assert torch.isfinite(pos).all(), "[NaN] pos after norm"

        for idx, layer in enumerate(self.layers):
            assert torch.isfinite(x).all(), f"[NaN] before layer {idx}"
            if isinstance(layer, PhyloEGCL):
                x, pos = layer(x, edge_index, pos)
            else:
                x = layer(x, edge_index)
            assert torch.isfinite(x).all(), f"[NaN] after layer {idx}"
            pos = self._norm_pos(pos)      # 每层归一化
            

        return self.out(self.pool(x, batch))


# ---------- TreeEncoder ----------
class TreeEncoder(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=128, out_dim=256,
                 num_layers=4, dropout_rate=0.25):
        super().__init__()
        self.feat_drop = nn.Dropout(dropout_rate)
        self.egnn = PhyloEGNN(
            in_dim=input_dim, h=hidden_dim, out_dim=out_dim,
            L=num_layers, aggr="add", dropout=dropout_rate
        )

    def forward(self, x, edge_index, pos, batch):
        x = self.feat_drop(x)
        return self.egnn(x, pos, edge_index, batch)