import os, random
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Union, Optional

class DNABERTEmbeddingDataset(Dataset):
    """
    每个 sample 一行：sample_id, 其他列可忽略。
    文件路径:  <dna_dir>/<sample_id><dna_ext>  (内容为 (N, 768) tensor / npy)
    """
    def __init__(self,
                 meta_csv: str,
                 dna_dir: str,
                 dna_ext: str = ".pt",
                 max_genes: Optional[int] = None):
        super().__init__()
        self.dna_dir   = dna_dir
        self.dna_ext   = dna_ext
        self.max_genes = max_genes

        self.sample_ids = pd.read_csv(meta_csv)["sample_id"].tolist()

    # ------------------------------------------
    def __len__(self):
        return len(self.sample_ids)

    def _load_tensor(self, path: str) -> torch.Tensor:
        if path.endswith(".pt"):
            return torch.load(path, map_location="cpu")      # (N, 768)
        else:
            raise ValueError(f"Unsupported DNA ext: {path}")

    def __getitem__(self, idx):
        sid  = self.sample_ids[idx]
        path = os.path.join(self.dna_dir, sid + self.dna_ext)
        dna  = self._load_tensor(path)                       # (N, 768)

        if self.max_genes and dna.size(0) > self.max_genes:  # 随机截断
            sel = torch.randperm(dna.size(0))[: self.max_genes]
            dna = dna[sel]

        return {"sample_id": sid, "embedding": dna}