import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch.nn import Linear, SiLU


class CoorsNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=1e-2):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.tensor([scale_init]))

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return coors / norm * self.scale


class PhyloEGCL(MessagePassing):
    """
    顺序：
        1. 用旧坐标预测 coord_update           (coord_mlp)
        2. coords ← coords + coord_update      (scatter_add)
        3. 用 **新坐标** 计算边消息 / 节点更新   (edge_mlp, node_mlp)
    这样 coord_mlp 参数梯度会穿过新坐标, 包括最后一层。
    """
    def __init__(self, hidden_dim, edge_dim=3, update_coords=True):
        super().__init__(aggr="add")
        self.update_coords = update_coords

        # --- 1. 坐标增量网络 ---
        if update_coords:
            self.coord_mlp = nn.Sequential(
                Linear(2 * hidden_dim + edge_dim, hidden_dim * 2),
                SiLU(),
                Linear(hidden_dim * 2, 3)
            )
            self.edge_weight_c = nn.Sequential(
                Linear(3, 1),                # 只用几何信息调权
                nn.Sigmoid()
            )
            self.coors_norm = CoorsNorm()

        # --- 2. 消息 / 节点更新网络 ---
        self.edge_mlp = nn.Sequential(
            Linear(2 * hidden_dim + edge_dim, hidden_dim * 2),
            SiLU(),
            Linear(hidden_dim * 2, hidden_dim),
            SiLU()
        )
        self.node_mlp = nn.Sequential(
            Linear(hidden_dim * 2, hidden_dim * 2),
            SiLU(),
            Linear(hidden_dim * 2, hidden_dim)
        )

    # ------------------------------------------------------------------ #
    def _compute_rel(self, coords, row, col):
        rel = coords[row] - coords[col]                     # (E, 3)
        return rel, torch.norm(rel, dim=-1, keepdim=True)   # (E,3) , (E,1)

    def forward(self, x, edge_index, coords):
        row, col = edge_index

        # --------------------- ① 预测并应用坐标更新 --------------------- #
        if self.update_coords:
            rel, _ = self._compute_rel(coords, row, col)
            coord_in = torch.cat([x[row], x[col], rel], dim=-1)      # (E, 2H+3)
            delta   = self.coord_mlp(coord_in)                       # (E, 3)
            delta   = self.coors_norm(delta)
            delta   = delta * self.edge_weight_c(rel)                # (E, 3)

            coords = coords.clone()
            # 确保数据类型匹配
            delta = delta.to(coords.dtype)
            coords.index_add_(0, row, delta)                         # scatter_add

        # --------------------- ② 用 **新坐标** 计算消息 ------------------ #
        rel, _    = self._compute_rel(coords, row, col)
        edge_in   = torch.cat([x[row], x[col], rel], dim=-1)
        e_ij      = self.edge_mlp(edge_in)                           # (E, H)

        agg = torch.zeros_like(x)
        # 确保数据类型匹配
        e_ij = e_ij.to(agg.dtype)
        agg.index_add_(0, row, e_ij)                                 # scatter_add

        x = x + self.node_mlp(torch.cat([x, agg], dim=-1))           # 残差
        return x, coords


class PhyloEGNN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, out_dim=128, num_layers=4):
        super().__init__()
        self.input_proj = Linear(input_dim, hidden_dim)

        # 所有层都允许更新坐标，因为现在梯度链完整
        self.layers = nn.ModuleList(
            [PhyloEGCL(hidden_dim) for _ in range(num_layers)]
        )

        self.output_proj = nn.Sequential(
            nn.ReLU(),
            Linear(hidden_dim, out_dim)
        )

    def forward(self, x, edge_index, coords, batch):
        x = self.input_proj(x)
        for layer in self.layers:
            x, coords = layer(x, edge_index, coords)
            x = F.layer_norm(x, x.shape[-1:])               # 轻量 LN
        # 使用 global_mean_pool 而不是 mean，以保持批次维度
        out = self.output_proj(global_mean_pool(x, batch))
        return F.normalize(out, p=2, dim=-1)


class TreeEncoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, out_dim=256, num_layers=4):
        super().__init__()
        self.egnn = PhyloEGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers
        )

    def forward(self, x, edge_index, pos, batch):
        return self.egnn(x, edge_index, pos, batch)