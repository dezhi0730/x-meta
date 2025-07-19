# gutclip/data/gut_dataset.py
from __future__ import annotations
import torch, torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

# 你的子 Dataset
from .dna_dataset  import DNABERTEmbeddingDataset
from .tree_dataset import TreeEGNNDataset


# ------------------------------------------------------------------
# 1. collate_fn  —— 负责：padding、随机遮挡、图合批
# ------------------------------------------------------------------
def gutclip_collate_fn(batch: list[Data]):
    """
    输入 batch 内的 Data：
        .dna : (N_i, 768)   可变长度 DNA 嵌入
        其余为树的 x/pos/edge_index
    返回的 Batch 增加了：
        .dna            (B, L_max, 768)
        .dna_pad_mask   (B, L_max)   True=有效 token
        .dna_rand_mask  (B, L_max)   True=随机被 mask（0.20）
    """
    # ---- 1) padding 到同一长度 ----
    dna_list = [d.dna for d in batch]
    max_len  = max(t.size(0) for t in dna_list)
    pad_dnas, dna_lens = [], []
    for t in dna_list:
        pad = (0, 0, 0, max_len - t.size(0))          # 右侧 0-padding
        pad_dnas.append(F.pad(t, pad, value=0.0))
        dna_lens.append(t.size(0))
    dna_tensor = torch.stack(pad_dnas, 0)             # (B, L, 768)
    dna_lens   = torch.tensor(dna_lens, device=dna_tensor.device)  # (B,)

    # ---- 2) padding-mask：True = 有效 token ----
    pad_mask = torch.arange(max_len, device=dna_tensor.device)[None, :] < dna_lens[:, None]  # (B,L)

    # ---- 3) 随机遮挡-mask：True = 将被随机遮掉的 token ----
    prob, keep_min = 0.20, 128                        # 20 %；每序列至少留 128
    rand_mask = torch.zeros_like(pad_mask)
    for b, N in enumerate(dna_lens):
        N = int(N.item())
        if N <= keep_min:
            continue                                  # 太短就不遮
        n_mask = int(N * prob)
        if N - n_mask < keep_min:
            n_mask = N - keep_min
        idx = torch.randperm(N, device=dna_tensor.device)[: n_mask]
        rand_mask[b, idx] = True

    # ---- 4) Debug 打印（2 % 采样）----
    if torch.rand(1).item() < 0.02:
        print(f"[Debug] DNA pad-mask mean={pad_mask.float().mean():.3f}  "
              f"rand-mask mean={ (rand_mask & pad_mask).float().mean():.3f}  "
              f"min_len={dna_lens.min()}  max_len={dna_lens.max()}  "
              f"avg_len={dna_lens.float().mean():.1f}")

    # ---- 5) 图合批 ----
    pyg_batch = Batch.from_data_list(batch)
    pyg_batch.dna            = dna_tensor
    pyg_batch.dna_pad_mask   = pad_mask
    pyg_batch.dna_rand_mask  = rand_mask
    
    # 计算每个样本的0丰度节点比例（如果存在node_zero属性）
    if hasattr(pyg_batch, 'node_zero'):
        # 使用PyTorch Geometric的batch属性来正确计算每个样本的0丰度比例
        node_zero = pyg_batch.node_zero.float()  # (总节点数,)
        batch_idx = pyg_batch.batch  # (总节点数,) 每个节点属于哪个样本
        
        # 为每个样本计算0丰度比例
        batch_size = batch_idx.max().item() + 1
        zero_ratios = []
        
        for i in range(batch_size):
            mask = (batch_idx == i)
            if mask.any():
                sample_zero_ratio = node_zero[mask].mean()
                zero_ratios.append(sample_zero_ratio)
            else:
                # 如果某个样本没有节点，使用默认值
                zero_ratios.append(torch.tensor(0.0, device=node_zero.device))
        
        pyg_batch.zero_ratio = torch.stack(zero_ratios)  # (B,)
        
    return pyg_batch


# ------------------------------------------------------------------
# 2. 主 Dataset：DNA + Tree 对齐
# ------------------------------------------------------------------
class GutCLIPDataset(Dataset):
    """只保留 DNA 与树都存在的样本"""
    def __init__(self,
                 dna_dataset : DNABERTEmbeddingDataset,
                 tree_dataset):
        self.dna_dataset  = dna_dataset
        self.tree_dataset = tree_dataset

        # 检查样本ID匹配情况
        dna_sample_ids = set(dna_dataset.sample_ids)
        # 适配不同的树数据集类型
        if hasattr(tree_dataset, 'sample_id_to_idx'):
            # TreeEGNNDataset
            tree_sample_ids = set(tree_dataset.sample_id_to_idx.keys())
        elif hasattr(tree_dataset, 'records'):
            # TreeSplitDataset
            tree_sample_ids = set(rec['sid'] for rec in tree_dataset.records)
        else:
            raise RuntimeError(f"Unknown tree dataset type: {type(tree_dataset)}")

        common_ids = dna_sample_ids & tree_sample_ids
        print(f"[INFO] DNA samples: {len(dna_sample_ids)}")
        print(f"[INFO] Tree samples: {len(tree_sample_ids)}")
        print(f"[INFO] Common samples: {len(common_ids)}")

        self.valid_indices = [
            i for i in range(len(dna_dataset))
            if tree_dataset.get_index_by_sample_id(dna_dataset[i]["sample_id"]) is not None
        ]
        if not self.valid_indices:
            print(f"[ERROR] No paired samples found!")
            print(f"[ERROR] DNA dataset type: {type(dna_dataset)}")
            print(f"[ERROR] Tree dataset type: {type(tree_dataset)}")
            print(f"[ERROR] Tree dataset has get_index_by_sample_id: {hasattr(tree_dataset, 'get_index_by_sample_id')}")
            if hasattr(tree_dataset, 'sample_id_to_idx'):
                print(f"[ERROR] Tree sample_id_to_idx keys: {list(tree_dataset.sample_id_to_idx.keys())[:10]}")
            elif hasattr(tree_dataset, 'records'):
                print(f"[ERROR] Tree records sample IDs: {[rec['sid'] for rec in tree_dataset.records[:10]]}")
            raise RuntimeError("No paired samples found! Cannot proceed with training.")

        print(f"[INFO] Paired samples: {len(self.valid_indices)}/{len(dna_dataset)}")

    def __len__(self): return len(self.valid_indices)

    def __getitem__(self, idx):
        # 尝试获取合法样本（最多尝试10次）
        for _ in range(10):
            dna_idx = self.valid_indices[idx]
            dna     = self.dna_dataset[dna_idx]
            sid     = dna["sample_id"]

            tree_idx = self.tree_dataset.get_index_by_sample_id(sid)
            tree     = self.tree_dataset[tree_idx]

            # 构建Data对象
            data = Data(
                x          = tree.x,
                edge_index = tree.edge_index,
                pos        = tree.pos,
                dna        = dna["embedding"],  # (N, 768)
                sample_id  = sid
            )

            if hasattr(tree, 'node_zero'):
                data.node_zero = tree.node_zero

            # --- 检查是否包含NaN/Inf ---
            has_invalid = False
            for attr_name in ['x', 'pos', 'dna']:
                attr = getattr(data, attr_name, None)
                if attr is not None and torch.is_tensor(attr):
                    if torch.isnan(attr).any() or torch.isinf(attr).any():
                        has_invalid = True
                        break
            if hasattr(data, 'node_zero') and torch.is_tensor(data.node_zero):
                if torch.isnan(data.node_zero).any() or torch.isinf(data.node_zero).any():
                    has_invalid = True

            if not has_invalid:
                return data  # 合法样本，正常返回
            
            # 否则跳过当前 idx，尝试下一个
            idx = (idx + 1) % len(self)  # 环形前进防止越界

        raise ValueError(f"[ERROR] 连续10个样本含有 NaN 或 Inf，可能数据集存在严重异常。")