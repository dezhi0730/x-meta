# tree_preprocessor.py
# ============================================================
import torch, numpy as np
from collections import defaultdict
from typing import Optional, Tuple

# ------------------------------------------------------------
class TreeEGNNPreprocessor:
    def __init__(self, graph, detection_threshold: float = 0.0,
                 augment_ratio: float = 0.1):
        self.graph = graph
        self.node_dict = graph.NODE_DICT
        self.detection_threshold = detection_threshold
        self.augment_ratio = augment_ratio

    # ---------- 工具 ----------
    @staticmethod
    def scale_minmax(arr, lo=-1., hi=1.):
        arr = torch.as_tensor(arr, dtype=torch.float32)
        if arr.numel() < 2:
            return torch.zeros_like(arr)
        out = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        return out * (hi - lo) + lo

    @staticmethod
    def quantile_norm(arr):
        arr = np.asarray(arr, np.float32)
        logv = np.log1p(arr)
        idx = np.argsort(logv[logv > 0])
        if idx.size < 2:
            if logv.max() - logv.min() < 1e-6:
                return torch.zeros_like(torch.tensor(logv, dtype=torch.float32))
            q = (logv - logv.min()) / (logv.max() - logv.min())
            return torch.tensor(q, dtype=torch.float32)
        ranks = np.empty_like(idx); ranks[idx] = np.arange(idx.size)
        q = np.zeros_like(logv); q[logv > 0] = ranks / (idx.size - 1 + 1e-8)
        return torch.tensor(q, dtype=torch.float32)

    def assign_sibling_indices(self):
        layers = defaultdict(list)
        for n in self.node_dict.values():
            layers[n.get_layer()].append(n)
        for nodes in layers.values():
            for i, n in enumerate(sorted(nodes, key=lambda x: str(x.get_id()))):
                n.sibling_index = i

    # ---------- 数据增强 ----------
    def augment_graph(self, x, pos, edge_index, node_zero):
        if not self.training or self.augment_ratio <= 0:
            return x, pos, edge_index
        is_nonzero = ~node_zero.bool()
        feat_noise = torch.randn_like(x) * self.augment_ratio
        x = x + feat_noise * is_nonzero.unsqueeze(1).float()
        pos = pos + torch.randn_like(pos) * (self.augment_ratio * 0.1)
        if len(edge_index.t()) > 2:
            mask = torch.rand(len(edge_index.t())) > (self.augment_ratio * 0.5)
            edge_index = edge_index[:, mask]
            if 0 < len(edge_index.t()) < 2:
                edge_index = torch.cat([edge_index, edge_index], dim=1)
        return x, pos, edge_index

    # ---------- 稀疏过滤 ----------
    def filter_sparse_nodes(self, x, pos, edge_index, node_zero):
        N = x.size(0)
        if N < 2:
            return None, None, None, None
        deg = torch.zeros(N, dtype=torch.long, device=x.device)
        one = torch.ones(edge_index.size(1), dtype=torch.long, device=x.device)
        deg.scatter_add_(0, edge_index[0], one)
        deg.scatter_add_(0, edge_index[1], one)
        keep = (x[:, 1] > 0) | (deg > 0)
        idx = torch.nonzero(keep).view(-1)
        if idx.numel() < 2:
            top_by_abun = torch.topk(x[:, 2], k=min(2, N)).indices
            idx = torch.unique(top_by_abun)
            if idx.numel() < 2:
                top_by_deg = torch.topk(deg.float(), k=min(2, N)).indices
                idx = torch.unique(torch.cat([idx, top_by_deg]))
        if idx.numel() < 2:
            return None, None, None, None
        o2n = torch.full((N,), -1, dtype=torch.long, device=x.device)
        o2n[idx] = torch.arange(idx.size(0), device=x.device)
        m = (o2n[edge_index[0]] >= 0) & (o2n[edge_index[1]] >= 0)
        ei_new = edge_index[:, m]
        if ei_new.numel() == 0 and idx.numel() >= 2:
            v0 = torch.zeros(idx.numel() - 1, dtype=torch.long, device=x.device)
            oth = torch.arange(1, idx.numel(), dtype=torch.long, device=x.device)
            ei_new = torch.cat([torch.stack([v0, oth]), torch.stack([oth, v0])], 1)
        else:
            ei_new = torch.stack([o2n[edge_index[0, m]], o2n[edge_index[1, m]]])
        return x[idx], pos[idx], ei_new, node_zero[idx]

    # ---------- 主函数 ----------
    def get_tensors(self, training=False) -> Optional[Tuple[torch.Tensor, ...]]:
        self.training = training
        self.assign_sibling_indices()
        nodes = list(self.node_dict.values())
        N = len(nodes)
        if N == 0:
            return None, None, None, None

        raw = torch.tensor([n.get_abundance() for n in nodes], dtype=torch.float32)
        log_abun = torch.log1p(raw)
        quant_abun = self.quantile_norm(raw)
        is_present = (raw > self.detection_threshold).float()
        node_zero = (raw == 0).float()

        depth = torch.tensor([n.get_layer() for n in nodes], dtype=torch.float32)
        depth_sc = self.scale_minmax(depth)
        sib_idx = torch.tensor([getattr(n, 'sibling_index', 0) for n in nodes],
                               dtype=torch.float32)
        sib_sc = self.scale_minmax(sib_idx)

        rank = (torch.argsort(torch.argsort(log_abun)).float() /
                (N - 1) if N > 1 else torch.zeros_like(log_abun))
        zscr = (log_abun - log_abun.mean()) / (log_abun.std() + 1e-6)

        parent_abun = torch.zeros_like(raw)
        sibling_mean = torch.zeros_like(raw)
        node2idx = {n: i for i, n in enumerate(nodes)}
        for n, i in node2idx.items():
            if n.parent and n.parent in node2idx:
                p = node2idx[n.parent]
                parent_abun[i] = raw[p]
                sibs = [s for s in n.parent.children if s is not n]
                if sibs:
                    si = [node2idx[s] for s in sibs if s in node2idx]
                    if si:
                        sibling_mean[i] = raw[si].mean()

        q_parent = self.quantile_norm(parent_abun)
        q_sib = self.quantile_norm(sibling_mean)

        degree = torch.zeros(N, dtype=torch.float32)
        for n in nodes:
            if n.parent and n.parent in node2idx:
                p = node2idx[n.parent]; c = node2idx[n]
                degree[[p, c]] += 1
        degree_sc = self.scale_minmax(degree)

        x = torch.stack([quant_abun, is_present, log_abun, depth_sc,
                         q_parent, q_sib, rank, zscr, degree_sc], 1)

        pos = torch.stack([
            depth_sc,
            sib_sc,
            self.scale_minmax(log_abun)
        ], 1)

        # ★ 入口统一：center + RMS
        pos = pos - pos.mean(0, keepdim=True)
        rms = pos.pow(2).sum(-1).mean().sqrt()
        pos = pos / rms.clamp_min(1e-6)

        # 边
        ei = []
        for n in nodes:
            if n.parent and n.parent in node2idx and n in node2idx:
                p, c = node2idx[n.parent], node2idx[n]
                ei += [[p, c], [c, p]]
        if not ei and N > 1:
            ei = [[0, i] for i in range(1, N)] + [[i, 0] for i in range(1, N)]
        edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()

        x, pos, edge_index, node_zero = self.filter_sparse_nodes(
            x, pos, edge_index, node_zero)
        if x is None:
            return None, None, None, None

        if training:
            x, pos, edge_index = self.augment_graph(x, pos, edge_index, node_zero)
        return x, pos, edge_index, node_zero

    # ---------- 递归填丰度 ----------
    def fill_abundance(self, otu2node: dict, vec: np.ndarray, otu_list: list[str]):
        for n in self.node_dict.values():
            n.set_abundance(0.0)
        for i, otu in enumerate(otu_list):
            node = otu2node.get(otu)
            if node is not None:
                node.set_abundance(float(vec[i]))

        def agg(node):
            if not node.children:
                return node.get_abundance()
            s = sum(agg(c) for c in node.children)
            node.set_abundance(s)
            return s

        if self.graph.root is not None:
            agg(self.graph.root)

    # ---------- 外部统一调用 ----------
    def process(self, training=False):
        out = self.get_tensors(training)
        if any(t is None for t in out):
            return None, None, None, None
        x, pos, ei, node_zero = out
        assert x.shape[0] == pos.shape[0]
        if ei is not None and ei.numel() > 0:
            assert ei.max().item() < x.shape[0]
        return x, pos, ei, node_zero
