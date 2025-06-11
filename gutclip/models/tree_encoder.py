import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GlobalAttention
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
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.egnn_layers = nn.ModuleList([
            PhyloEGCL(hidden_dim)
            for _ in range(num_layers)
        ])

        # 更强 attention gate
        self.pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(hidden_dim // 2, 1)
            )
        )

        self.output_proj = nn.Linear(hidden_dim, output_dim)
        # self.layer_norm = nn.LayerNorm(output_dim)  # 注释掉
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.input_proj.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.input_proj.bias)
        for layer in self.pool.gate_nn:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(layer.bias)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.output_proj.bias)
        # nn.init.ones_(self.layer_norm.weight)  # 注释掉
        # nn.init.zeros_(self.layer_norm.bias)   # 注释掉

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for layer in self.egnn_layers:
            x, pos = layer(x, edge_index, pos)

        # batch_size = batch.max().item() + 1
        # for b in range(batch_size):
        #     mask = (batch == b)
        #     if mask.sum() > 1:
        #         batch_x = x_norm[mask]
        #         cos_sim = torch.mm(batch_x, batch_x.t())
        #         print(f"[Debug] Before pooling - Batch {b} node-level cosine similarity matrix:\n{cos_sim.detach().cpu().numpy()}")
        #         print(f"[Debug] Before pooling - Batch {b} node-level std: {x[mask].std(dim=0).mean().item():.4f}")

        # pooling
        pooled_out = self.pool(x, batch)

        # pooling后分布分析
        # x_norm = F.normalize(pooled_out, p=2, dim=-1)
        # cos_sim = torch.mm(x_norm, x_norm.t())
        # print(f"[Debug] After pooling - Batch-level cosine similarity matrix:\n{cos_sim.detach().cpu().numpy()}")
        # print(f"[Debug] After pooling - Batch-level std: {pooled_out.std(dim=0).mean().item():.4f}")

        x = self.output_proj(pooled_out)
        # x = self.layer_norm(x)  # 注释掉
        return x



class TreeEncoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, out_dim=256, num_layers=4, dropout_rate=0.1):
        """
        Args:
            input_dim: Number of input features (default=6 for [quant_abundance, is_present, log_abundance, depth, parent_abundance, sibling_mean])
            hidden_dim: Hidden dimension size
            out_dim: Output dimension size
            num_layers: Number of EGNN layers
            dropout_rate: Dropout rate for feature dropout
        """
        super().__init__()
        self.egnn = PhyloEGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=out_dim,
            num_layers=num_layers
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, pos, batch):
        # 训练时添加微噪声
        
        # Apply dropout to input features during training
            
        x = self.egnn(x, pos, edge_index, batch)
        return x