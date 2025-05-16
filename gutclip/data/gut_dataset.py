# gutclip/data/gut_dataset.py
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

from .dna_dataset    import DNABERTEmbeddingDataset
from .tree_dataset import TreeEGNNDataset   

# --------------------------- collate_fn ---------------------------------
def gutclip_collate_fn(batch: list[Data]):
    # 1) DNA padding 到 batch 内最长
    dna_list   = [d.dna for d in batch]                  # list[(N_i, 768)]
    max_len    = max(t.size(0) for t in dna_list)
    pad_dnas   = []
    for t in dna_list:
        pad = (0, 0, 0, max_len - t.size(0))             # pad to (max_len, 768)
        pad_dnas.append(F.pad(t, pad, value=0))
    dna_tensor = torch.stack(pad_dnas, 0)                # (B, max_len, 768)

    # 2) 图合批
    pyg_batch  = Batch.from_data_list(batch)             # .x / .edge_index / .pos
    pyg_batch.dna      = dna_tensor                      # 增加 dna 张量
    pyg_batch.dna_len  = torch.tensor([t.size(0) for t in dna_list])

    return pyg_batch


# ------------------------- GutCLIPDataset --------------------------------
class GutCLIPDataset(Dataset):
    """
    DNADataset 决定 train/val；TreeEGNNDataset 只加载一次共用。
    只保留同时有 DNA 和树的样本。
    """
    def __init__(self,
                 dna_dataset:  DNABERTEmbeddingDataset,
                 tree_dataset: 'TreeEGNNDataset'):
        self.dna_dataset  = dna_dataset
        self.tree_dataset = tree_dataset
        
        # 过滤掉没有对应树的样本
        self.valid_indices = []
        for idx in range(len(dna_dataset)):
            sid = dna_dataset[idx]["sample_id"]
            if tree_dataset.get_index_by_sample_id(sid) is not None:
                self.valid_indices.append(idx)
        
        if not self.valid_indices:
            raise ValueError("No valid samples found! All DNA samples are missing corresponding trees.")
        
        print(f"[INFO] Found {len(self.valid_indices)} valid samples out of {len(dna_dataset)} total samples")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        dna_idx = self.valid_indices[idx]
        dna_sample = self.dna_dataset[dna_idx]
        sid = dna_sample["sample_id"]
        
        tree_idx = self.tree_dataset.get_index_by_sample_id(sid)
        tree_data = self.tree_dataset[tree_idx]          # PyG Data
        
        return Data(
            x=tree_data.x,
            edge_index=tree_data.edge_index,
            pos=tree_data.pos,
            dna=dna_sample["embedding"],                 # (N, 768) variable length
            sample_id=sid
        )