# gutclip/data/_data_module.py
from __future__ import annotations
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from .dna_dataset      import DNABERTEmbeddingDataset
from .tree_dataset import TreeEGNNDataset
from .gut_dataset      import GutCLIPDataset, gutclip_collate_fn


class GutDataModule:
    """
    * DNA train/val 拆分
    * TreeEGNNDataset 只实例化一次共用
    * 只保留同时有 DNA 和树的样本
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_cfg = cfg.data

        # -------- 共享 TreeEGNNDataset ------------------------
        self.tree_ds = TreeEGNNDataset(dataset_dir=self.data_cfg.tree_dir)

        # -------- DNA train / val -----------------------------
        self.train_dna = DNABERTEmbeddingDataset(
            meta_csv=self.data_cfg.train_meta,
            dna_dir=self.data_cfg.dna_dir,
            dna_ext=self.data_cfg.dna_ext,
            max_genes=self.data_cfg.max_genes,
        )
        self.val_dna   = DNABERTEmbeddingDataset(
            meta_csv=self.data_cfg.val_meta,
            dna_dir=self.data_cfg.dna_dir,
            dna_ext=self.data_cfg.dna_ext,
            max_genes=self.data_cfg.max_genes,
        )

        # -------- 拼接最终 Dataset ----------------------------
        self.train_set = GutCLIPDataset(self.train_dna, self.tree_ds)
        self.val_set   = GutCLIPDataset(self.val_dna,   self.tree_ds)

        # DDP
        self.world = dist.get_world_size() if dist.is_initialized() else 1

    # ---------------------------------------------------------
    def _loader(self, dataset, shuffle):
        sampler = DistributedSampler(dataset, shuffle=shuffle) \
                  if self.world > 1 else None
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=(False if sampler else shuffle),
            sampler=sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=gutclip_collate_fn,
        )

    def train_dataloader(self):
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_set,   shuffle=False)