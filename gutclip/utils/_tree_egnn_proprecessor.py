import torch
import numpy as np
from collections import defaultdict
from typing import Optional, Tuple

class TreeEGNNPreprocessor:
    def __init__(self, graph):
        self.graph = graph
        self.node_dict = graph.NODE_DICT

    def assign_sibling_indices(self):
        layer_dict = defaultdict(list)
        for node in self.node_dict.values():
            layer_dict[node.get_layer()].append(node)
        for nodes in layer_dict.values():
            for i, node in enumerate(sorted(nodes, key=lambda x: str(x.get_id()))):
                node.sibling_index = i

    def fill_abundance(self, otu2node, abundance_vec, otu_list):
        """填充节点丰度并递归自底向上聚合"""
        # 1. 清零所有节点
        for node in self.node_dict.values():
            node.set_abundance(0.0)

        # 2. 映射叶子丰度
        node_values = []
        missing_nodes = []
        for i, otu in enumerate(otu_list):
            node = otu2node.get(otu, None)
            if node is not None:
                val = float(abundance_vec[i])
                node.set_abundance(val)
                node_values.append(val)
            else:
                missing_nodes.append(otu)
        print(f"[INFO] Assigned abundances: {len(node_values)}/{len(otu_list)}")
        if node_values:
            print(f"[INFO] Abundance stats: min={min(node_values):.3f}, max={max(node_values):.3f}, mean={sum(node_values)/len(node_values):.3f}")
        if missing_nodes:
            print(f"[WARNING] {len(missing_nodes)} nodes not found in tree")
            print(f"[WARNING] First 5 missing nodes: {missing_nodes[:5]}")

        # 3. 递归自底向上聚合
        def aggregate_abundance(node):
            if not node.children:
                return node.get_abundance()
            total = sum(aggregate_abundance(child) for child in node.children)
            node.set_abundance(total)
            return total

        if self.graph.root is not None:
            aggregate_abundance(self.graph.root)
        else:
            print("[ERROR] Tree has no root!")

        # 4. 打印最终丰度分布
        final_abundances = [node.get_abundance() for node in self.node_dict.values()]
        print(f"[INFO] Final node abundance stats: min={min(final_abundances):.3f}, max={max(final_abundances):.3f}, mean={sum(final_abundances)/len(final_abundances):.3f}")
        print(f"[INFO] Non-zero nodes: {sum(1 for x in final_abundances if x > 0)}/{len(final_abundances)}")

    def scale_minmax(self, arr, min_val: float = -1, max_val: float = 1) -> torch.Tensor:
        arr = torch.as_tensor(arr, dtype=torch.float32)
        if arr.numel() < 2: return torch.zeros_like(arr)
        normed = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        return normed * (max_val - min_val) + min_val

    def abundance_quantile_norm(self, raw_abundance):
        try:
            arr = np.asarray(raw_abundance, dtype=np.float32)
            log_abun = np.log1p(arr)
            nonzero = log_abun > 0
            quantile = np.zeros_like(log_abun)
            if nonzero.sum() > 0:
                sorted_idx = np.argsort(log_abun[nonzero])
                ranks = np.empty_like(sorted_idx)
                ranks[sorted_idx] = np.arange(sorted_idx.size)
                quantile_vals = ranks / (sorted_idx.size - 1 + 1e-8)
                quantile[nonzero] = quantile_vals
            return torch.tensor(quantile, dtype=torch.float32)
        except Exception as e:
            print(f"[ERROR] Quantile normalization failed: {str(e)}")
            return torch.zeros_like(torch.as_tensor(raw_abundance, dtype=torch.float32))

    def get_node_coords_and_features(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        nodes = list(self.node_dict.values())
        N = len(nodes)
        abundance_vec = [node.get_abundance() for node in nodes]
        abundance_vec = np.array(abundance_vec)
        if N < 10 or (abundance_vec.shape[0] != N):
            print(f"[WARNING] Skipped: node count {N}, abundance shape {abundance_vec.shape}")
            return None

        raw_abundance = torch.tensor(abundance_vec, dtype=torch.float32)
        log_abundance = torch.log1p(raw_abundance)
        quant_abundance = self.abundance_quantile_norm(raw_abundance)
        is_present = (raw_abundance > 0).float()
        depths = torch.tensor([node.get_layer() for node in nodes], dtype=torch.float32)
        sibling_indices = torch.tensor([getattr(node, 'sibling_index', 0) for node in nodes], dtype=torch.float32)

        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        parent_abundance = torch.zeros(N, dtype=torch.float32)
        sibling_mean_abundance = torch.zeros(N, dtype=torch.float32)
        for node, idx in node_to_idx.items():
            if node.parent and node.parent in node_to_idx:
                parent_idx = node_to_idx[node.parent]
                parent_abundance[idx] = raw_abundance[parent_idx]
                siblings = [n for n in nodes if n.parent == node.parent and n != node]
                if siblings:
                    sib_idx = [node_to_idx[s] for s in siblings]
                    sibling_mean_abundance[idx] = raw_abundance[sib_idx].mean()

        quant_parent_abundance = self.abundance_quantile_norm(parent_abundance)
        quant_sibling_mean = self.abundance_quantile_norm(sibling_mean_abundance)

        features = torch.stack([
            quant_abundance,
            is_present,
            log_abundance,
            depths,
            quant_parent_abundance,
            quant_sibling_mean
        ], dim=1)

        scaled_depths = self.scale_minmax(depths, -1, 1)
        scaled_siblings = self.scale_minmax(sibling_indices, -1, 1)
        coords = torch.stack([
            scaled_depths,
            scaled_siblings,
            torch.zeros_like(scaled_depths)
        ], dim=1)
        return coords, features

    def build_edge_index(self) -> torch.Tensor:
        nodes = list(self.node_dict.values())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        edges = []
        for node in nodes:
            if node.parent and node.parent in node_to_idx:
                src = node_to_idx[node]
                dst = node_to_idx[node.parent]
                edges.append([src, dst])
                edges.append([dst, src])
        if not edges:
            edges = [[i, i] for i in range(len(nodes))]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

    def process(self) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        self.assign_sibling_indices()
        res = self.get_node_coords_and_features()
        if res is None: return None
        coords, x = res
        edge_index = self.build_edge_index()
        if not (x.shape[0] == coords.shape[0] == edge_index.max().item() + 1):
            print(f"[ERROR] Inconsistent node shapes: {x.shape}, {coords.shape}, edge max {edge_index.max().item()}")
            return None
        return x, coords, edge_index