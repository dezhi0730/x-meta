import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from typing import Dict, Any, Optional

class DiffusionDataset(Dataset):
    """
    直接读取 export_embeddings.py 导出的 .pt：
      {'z': FloatTensor[B,D], 'y': FloatTensor[B,N], 'sample_ids': list[str]}

    可选：若 export 里没有 sample_ids，也自动生成顺序 id。
    """
    def __init__(self, embed_pt: str, normalize: bool = False):
        obj = torch.load(embed_pt, map_location="cpu", weights_only=True)
        self.z: torch.Tensor = obj["z"].float()      # (B,D)
        self.y: torch.Tensor = obj["y"].float()      # (B,N)
        self.sample_ids = obj.get("sample_ids", None)
        if self.sample_ids is None:
            self.sample_ids = [f"idx_{i}" for i in range(len(self.z))]

        # 可选：再次归一化（一般不需要；你前面已 log1p + minmax 到 [-1,1]）
        if normalize:
            # 这里演示按样本维做 [-1,1]，如不需要可关掉
            v = self.y
            v = (v - v.min(dim=1, keepdim=True).values) / \
                (v.max(dim=1, keepdim=True).values - v.min(dim=1, keepdim=True).values + 1e-6)
            self.y = v * 2 - 1

    def __len__(self) -> int:
        return self.z.size(0)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return {
            "z_dna": self.z[i],          # (D,)
            "y":     self.y[i],          # (N,)
            "sid":   self.sample_ids[i], # str
            "index": i                   # int，全局顺序
        }


def diffusion_collate(batch: list[Dict[str,Any]]) -> Dict[str, Any]:
    z = torch.stack([b["z_dna"] for b in batch], 0)       # (B,D)
    y = torch.stack([b["y"]     for b in batch], 0)       # (B,N)
    sids  = [b["sid"]   for b in batch]
    idxes = torch.tensor([b["index"] for b in batch], dtype=torch.long)
    return {"z_dna": z, "y": y, "sid": sids, "index": idxes}


class DiffusionDataModule:
    """
    只负责把 train/val 两个 .pt 变成 DataLoader。
    配置需要包含：
      data:
        train_embed_pt: <path>
        val_embed_pt:   <path>
      train:
        batch_size: 64
      num_workers: 4
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.train_ds: Optional[DiffusionDataset] = None
        self.val_ds:   Optional[DiffusionDataset] = None

    def setup(self, stage: Optional[str] = None):
        train_pt = self.cfg.data.train_embed_pt
        val_pt   = self.cfg.data.val_embed_pt
        self.train_ds = DiffusionDataset(train_pt, normalize=False)
        self.val_ds   = DiffusionDataset(val_pt,   normalize=False)

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        sampler = DistributedSampler(ds, shuffle=shuffle) if self.world > 1 else None
        return DataLoader(
            ds,
            batch_size = self.cfg.train.batch_size,
            shuffle    = (False if sampler else shuffle),
            sampler    = sampler,
            num_workers= getattr(self.cfg, "num_workers", 4),
            pin_memory = True,
            collate_fn = diffusion_collate,
            drop_last  = False,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return self._loader(self.val_ds, shuffle=False)