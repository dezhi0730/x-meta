import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, GlobalAttention
from torch.nn import Linear, SiLU

# ---------- 1. Coord Normaliser ----------
class CoorsNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=0.2):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.tensor([scale_init], dtype=torch.float32))
    def forward(self, c):
        return c / c.norm(dim=-1, keepdim=True).clamp(min=self.eps) * self.scale


# ---------- 2. EGCL ----------
class PhyloEGCL(MessagePassing):
    def __init__(self, h, edge_dim=3, update_coords=True):
        super().__init__(aggr="add")
        self.update_coords = update_coords

        if update_coords:
            self.coord_mlp = nn.Sequential(
                Linear(2*h + edge_dim, 2*h), SiLU(), Linear(2*h, 3)
            )
            self.edge_w   = nn.Sequential(Linear(3,1), nn.Sigmoid())
            self.coors_norm = CoorsNorm()

        self.edge_mlp = nn.Sequential(
            Linear(2*h + edge_dim, 2*h), SiLU(), Linear(2*h, h), SiLU()
        )
        self.node_mlp = nn.Sequential(
            Linear(2*h, 2*h), SiLU(), Linear(2*h, h)
        )

        self.res_drop = nn.Dropout(0.15)
        self.ln       = nn.LayerNorm(h)

    def _rel(self, p, r, c):
        d = p[r] - p[c]
        return d, d.norm(dim=-1, keepdim=True)

    def forward(self, x, edge_index, pos):
        row, col = edge_index

        # ① coord update
        if self.update_coords:
            rel,_ = self._rel(pos, row, col)
            delta = self.coors_norm(self.coord_mlp(torch.cat([x[row], x[col], rel], -1)))
            delta = delta * self.edge_w(rel)
            pos   = pos.clone()
            pos.index_add_(0, row, delta.to(pos.dtype))

        # ② message passing
        rel,_  = self._rel(pos, row, col)
        msg    = self.edge_mlp(torch.cat([x[row], x[col], rel], -1))
        agg    = torch.zeros_like(x)
        agg.index_add_(0, row, msg.to(agg.dtype))

        # ③ node update + LN
        x = x + self.res_drop(self.node_mlp(torch.cat([x, agg], -1)))
        return self.ln(x), pos


# ---------- 3. Stacked EGNN ----------
class PhyloEGNN(nn.Module):
    def __init__(self, in_dim=8, h=128, out_dim=256, L=4):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, h), nn.LayerNorm(h))
        self.layers = nn.ModuleList([PhyloEGCL(h) for _ in range(L)])
        self.pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU(),
                nn.Dropout(0.1), nn.Linear(h, h//2), nn.ReLU(),
                nn.Dropout(0.1), nn.Linear(h//2, 1)
            )
        )
        self.out = nn.Linear(h, out_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.zeros_(m.bias)

    def forward(self, x, pos, edge_index, batch):
        x = self.proj(x)
        for layer in self.layers:
            x, pos = layer(x, edge_index, pos)
        return self.out(self.pool(x, batch))


# ---------- 4. TreeEncoder ---------------
class TreeEncoder(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128, out_dim=256, num_layers=4, dropout_rate=0.25):
        super().__init__()
        self.feat_drop = nn.Dropout(dropout_rate)
        self.egnn      = PhyloEGNN(input_dim, hidden_dim, out_dim, num_layers)

    def forward(self, x, edge_index, pos, batch):
        x = self.feat_drop(x)
        return self.egnn(x, pos, edge_index, batch)