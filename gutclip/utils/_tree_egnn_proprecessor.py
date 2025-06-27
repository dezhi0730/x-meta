
import torch, numpy as np
from collections import defaultdict
from typing import Optional, Tuple

class TreeEGNNPreprocessor:
    def __init__(self, graph):
        self.graph = graph
        self.node_dict = graph.NODE_DICT  # 假设外部已构建

    # ---------- 工具函数 -------------------------------------------------
    @staticmethod
    def scale_minmax(arr, lo=-1., hi=1.):
        arr = torch.as_tensor(arr, dtype=torch.float32)
        if arr.numel() < 2: return torch.zeros_like(arr)
        out = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        return out * (hi - lo) + lo

    @staticmethod
    def quantile_norm(arr):
        arr = np.asarray(arr, np.float32)
        logv = np.log1p(arr)
        idx = np.argsort(logv[logv > 0])
        ranks = np.empty_like(idx); ranks[idx] = np.arange(idx.size)
        q = np.zeros_like(logv); q[logv > 0] = ranks / (idx.size-1+1e-8)
        return torch.tensor(q, dtype=torch.float32)

    # ---------- 1. sibling index -------------
    def assign_sibling_indices(self):
        layers = defaultdict(list)
        for n in self.node_dict.values():
            layers[n.get_layer()].append(n)
        for nodes in layers.values():
            for i, n in enumerate(sorted(nodes, key=lambda x: str(x.get_id()))):
                n.sibling_index = i

    # ---------- 2. 主函数 --------------------
    def get_tensors(self) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        self.assign_sibling_indices()
        nodes = list(self.node_dict.values())
        N = len(nodes)
        # --- raw abundance ---
        raw = torch.tensor([n.get_abundance() for n in nodes], dtype=torch.float32)
        log_abun   = torch.log1p(raw)
        quant_abun = self.quantile_norm(raw)         # 0-1
        is_present = (raw > 0).float()               # 0/1
        # --- depth & sibling ---
        depth_raw  = torch.tensor([n.get_layer() for n in nodes], dtype=torch.float32)
        depth_sc   = self.scale_minmax(depth_raw)    # [-1,1]
        sibling_idx= torch.tensor([getattr(n,'sibling_index',0) for n in nodes],
                                  dtype=torch.float32)
        sibling_sc = self.scale_minmax(sibling_idx)

        # --- rank & z-score ---
        rank = torch.argsort(torch.argsort(log_abun)).float() / (N-1)
        zscr = (log_abun - log_abun.mean()) / (log_abun.std()+1e-6)

        # --- parent / sibling mean (quantile) ---
        parent_abun = torch.zeros_like(raw)
        sibling_mean= torch.zeros_like(raw)
        node2idx    = {n:i for i,n in enumerate(nodes)}
        for n,i in node2idx.items():
            if n.parent and n.parent in node2idx:
                p_idx = node2idx[n.parent]
                parent_abun[i] = raw[p_idx]
                sibs = [s for s in n.parent.children if s is not n]
                if sibs:
                    sibling_mean[i] = raw[[node2idx[s] for s in sibs]].mean()
        q_parent = self.quantile_norm(parent_abun)
        q_sib    = self.quantile_norm(sibling_mean)

        # ---------- assemble ----------
        x = torch.stack([
            quant_abun, is_present, log_abun, depth_sc,
            q_parent, q_sib, rank, zscr         # ← 新增两列
        ], dim=1)                                # (N,8)

        pos = torch.stack([
            depth_sc, sibling_sc,
            self.scale_minmax(log_abun)          # 把丰度塞进第 3 维
        ], dim=1)                                # (N,3)

        edges=[]
        for n in nodes:
            if n.parent and n.parent in node2idx:
                p=node2idx[n.parent]; c=node2idx[n]
                edges += [[p,c],[c,p]]
        edge_index=torch.tensor(edges,dtype=torch.long).t().contiguous()

        return x, pos, edge_index
    
    def fill_abundance(self,
                       otu2node: dict,
                       abundance_vec: np.ndarray,
                       otu_list: list[str]):
        for n in self.node_dict.values():
            n.set_abundance(0.0)

        for i, otu in enumerate(otu_list):
            node = otu2node.get(otu, None)
            if node is not None:
                node.set_abundance(float(abundance_vec[i]))

        def agg(node):
            if not node.children:
                return node.get_abundance()
            s = sum(agg(c) for c in node.children)
            node.set_abundance(s);  return s
        if self.graph.root is not None:
            agg(self.graph.root)
    
    def process(self):
        tensors = self.get_tensors()
        if tensors is None:
            raise RuntimeError("preprocess failed")
        x, pos, edge_index = tensors
        assert x.shape[0] == pos.shape[0] == edge_index.max().item()+1
        return x, pos, edge_index