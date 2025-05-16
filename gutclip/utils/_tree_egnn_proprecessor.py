import torch
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, List


class TreeEGNNPreprocessor:
    def __init__(self, graph):
        self.graph = graph
        self.node_dict = graph.NODE_DICT

    def assign_sibling_indices(self):
        """为每层节点按名字排序，生成 sibling_index。"""
        layer_dict = defaultdict(list)
        for node in self.node_dict.values():
            layer_dict[node.get_layer()].append(node)

        for nodes in layer_dict.values():
            for i, node in enumerate(sorted(nodes, key=lambda x: str(x.get_id()))):
                node.sibling_index = i

    def scale_minmax(self, arr: torch.Tensor, min_val: float = -1, max_val: float = 1) -> torch.Tensor:
        """将输入张量缩放到 [min_val, max_val]"""
        normed = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)  # → [0, 1]
        return normed * (max_val - min_val) + min_val  # → [min_val, max_val]

    def get_node_coords_and_features(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        同时生成 node coordinates 和 features
        coords: (scaled depth, scaled sibling index, raw abundance)
        x: node feature = raw abundance (unsqueeze 1)
        """
        # 收集 abundance
        abundances = [float(node.get_abundance()) for node in self.node_dict.values()]
        raw_abundance = torch.tensor(abundances, dtype=torch.float32)  # 保留原始 abundance

        # 计算相对 depth 和 sibling index
        depths = torch.tensor([float(node.get_layer()) for node in self.node_dict.values()])
        sibling_indices = torch.tensor([float(getattr(node, 'sibling_index', 0)) for node in self.node_dict.values()])

        # 放缩 depth 和 sibling index
        scaled_depths = self.scale_minmax(depths, -1, 1)
        scaled_siblings = self.scale_minmax(sibling_indices, -1, 1)

        # 组合 coords
        coords = torch.stack([scaled_depths, scaled_siblings, raw_abundance], dim=1)  # shape (N, 3)
        x = raw_abundance.unsqueeze(1)  # shape (N, 1)

        # 打印调试信息
        print("[DEBUG] Depth stats:", scaled_depths.min().item(), scaled_depths.max().item(), scaled_depths.mean().item())
        print("[DEBUG] Sibling index stats:", scaled_siblings.min().item(), scaled_siblings.max().item(), scaled_siblings.mean().item())
        print("[DEBUG] Abundance stats:", raw_abundance.min().item(), raw_abundance.max().item(), raw_abundance.mean().item())
        print("[DEBUG] Coords stats:", coords.min(dim=0)[0], coords.max(dim=0)[0], coords.mean(dim=0))

        return coords, x

    def build_edge_index(self) -> torch.Tensor:
        """构建 PyG 格式的 edge_index，方向为 child → parent"""
        edges = []
        for node in self.node_dict.values():
            if node.parent:
                src = self._node_to_idx(node)
                dst = self._node_to_idx(node.parent)
                edges.append([src, dst])
                edges.append([dst, src])  # bidirectional
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def _node_to_idx(self, node) -> int:
        if not hasattr(self, '_node_idx_map'):
            self._node_idx_map = {
                node: idx for idx, node in enumerate(self.node_dict.values())
            }
        return self._node_idx_map[node]

    def process(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.assign_sibling_indices()
        coords, x = self.get_node_coords_and_features()
        edge_index = self.build_edge_index()
        return x, coords, edge_index