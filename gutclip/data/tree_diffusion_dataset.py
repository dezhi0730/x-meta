import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from .dna_dataset  import DNABERTEmbeddingDataset
from .tree_dataset import TreeEGNNDataset
from .split_dataset import TreeSplitDataset
from typing import List, Dict, Any, Optional
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from gutclip.diffusion import linear_beta_schedule, cosine_beta_schedule

class TreeDiffusionDataset(Dataset):
    """Paired DNA / Tree samples for fixed‑topology abundance diffusion.

    * separates noisy features (log_abun, presence) from static features (depth, q_parent, q_sib, degree)
    * DNA embedding is kept as full (L_i, 768) tensor – *no pooling here*
    """
    
    # Node features列索引
    NOISY_COLS   = [2, 1]            # log_abun, presence
    STATIC_COLS  = [3, 8]            # depth_sc, degree_sc (只保留这两个)

    def __init__(
        self,
        dna_dataset: DNABERTEmbeddingDataset, 
        tree_dataset: TreeEGNNDataset, 
    ) -> None:
        super().__init__()
        self.dna_dataset = dna_dataset
        self.tree_dataset = tree_dataset

        # ========= build paired list =========
        dna_sids = set(dna_dataset.sample_ids)
        
        # 适配不同的树数据集类型
        if hasattr(tree_dataset, 'sample_ids'):
            # TreeEGNNDataset
            tree_sids = set(tree_dataset.sample_ids)
        elif hasattr(tree_dataset, 'sample_id_to_idx'):
            # 其他可能有 sample_id_to_idx 的数据集
            tree_sids = set(tree_dataset.sample_id_to_idx.keys())
        elif hasattr(tree_dataset, 'records'):
            # TreeSplitDataset
            tree_sids = set(rec['sid'] for rec in tree_dataset.records)
        else:
            raise RuntimeError(f"Unknown tree dataset type: {type(tree_dataset)}")
            
        self.valid_sids: List[str] = sorted(dna_sids & tree_sids)
        assert self.valid_sids, "No paired samples found!"

        self.dna_sid2idx = {
            sid: i for i, sid in enumerate(dna_dataset.sample_ids)
        }
        
        # 创建树数据集的样本ID到索引的映射
        if hasattr(tree_dataset, 'sample_id_to_idx'):
            self.tree_sid2idx = tree_dataset.sample_id_to_idx
        elif hasattr(tree_dataset, 'records'):
            # TreeSplitDataset
            self.tree_sid2idx = {
                rec['sid']: i for i, rec in enumerate(tree_dataset.records)
            }
        else:
            # 如果没有 sample_id_to_idx，创建一个
            self.tree_sid2idx = {
                sid: i for i, sid in enumerate(tree_dataset.sample_ids)
            }

    # --------------------------------------------------
    def __len__(self):
        return len(self.valid_sids)

    # --------------------------------------------------
    def __getitem__(self, idx: int) -> Data:
        sid = self.valid_sids[idx]
        dna_rec = self.dna_dataset[self.dna_sid2idx[sid]]
        tree    = self.tree_dataset[self.tree_sid2idx[sid]]

        # 按照用户要求只选特定列
        x_abun   = tree.x[:, [2]].float()     # log_abun
        x_pres   = tree.x[:, [1]].long()      # presence 0/1
        x_static = tree.x[:, [3,8]].float()   # depth_sc, degree_sc

        data = Data(
            x0_abun   = x_abun,          # log_abun
            x0_pres   = x_pres,          # presence 0/1
            x_static  = x_static,        # 静态特征
            pos       = tree.pos,
            edge_index = tree.edge_index,
            batch     = torch.zeros(x_abun.size(0), dtype=torch.long),
            dna       = dna_rec["embedding"].float(),  # (L_i,768)
            sample_id = sid,
        )
        return data


class GaussianDiffusionCollate:
    """Collate function that pads DNA, builds masks, then adds Gaussian noise
    according to randomly sampled timesteps.
    """

    def __init__(
        self,
        betas: torch.Tensor,  # (T,)
        rand_mask_prob: float = 0.20,
        keep_min: int = 128,
    ) -> None:
        self.betas = betas.float()
        self.alphas_bar = (1.0 - self.betas).cumprod(0).cpu()  # 确保在CPU上
        self.rand_mask_prob = rand_mask_prob
        self.keep_min = keep_min

    # ---------------- internal helpers ----------------
    def _pad_dna(self, dna_list: List[torch.Tensor]):
        L_max = max(t.size(0) for t in dna_list)
        pad_dnas, pad_mask = [], []
        for t in dna_list:
            pad = (0, 0, 0, L_max - t.size(0))  # pad rows on dim 0
            pad_dnas.append(F.pad(t, pad, value=0.0))
            m = torch.zeros(L_max, dtype=torch.bool)
            m[: t.size(0)] = True  # valid positions
            pad_mask.append(m)
        return torch.stack(pad_dnas, 0), torch.stack(pad_mask, 0)  # (B,L,768) / (B,L)

    def _random_mask(self, pad_mask: torch.Tensor) -> torch.Tensor:
        """Generate random‑mask (bool) on valid positions only."""
        B, L = pad_mask.shape
        rand_mask = torch.zeros_like(pad_mask)
        for b in range(B):
            valid_len = pad_mask[b].sum().item()
            if valid_len <= self.keep_min:
                continue
            n_mask = int(valid_len * self.rand_mask_prob)
            if valid_len - n_mask < self.keep_min:
                n_mask = valid_len - self.keep_min
            idx = torch.randperm(valid_len, device=pad_mask.device)[: n_mask]
            rand_mask[b, idx] = True
        return rand_mask & pad_mask

    def _add_noise(self, x0: torch.Tensor, t_idx: torch.Tensor, batch: torch.Tensor):
        # 将计算移到GPU上进行，避免CPU-GPU数据传输瓶颈
        device = x0.device
        a_bar = self.alphas_bar[t_idx].sqrt().unsqueeze(1).to(device)  # (B,1)
        one_m = (1.0 - self.alphas_bar[t_idx]).sqrt().unsqueeze(1).to(device)  # (B,1)
        noise = torch.randn_like(x0, device=device)  # 直接在GPU上生成噪声
        result = a_bar[batch] * x0 + one_m[batch] * noise
        return result, noise
    
    def _bernoulli_noisy(self, x0_pres: torch.Tensor, t_idx: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """为离散流添加 Bernoulli 噪声"""
        # x0_pres: (ΣN,1) int64
        beta = self.betas[t_idx].view(-1, 1)            # (B,1)
        flip_prob = beta[batch] * x0_pres.float()       # 只对1节点生效
        rand = torch.rand_like(flip_prob)
        x_t_pres = (rand > flip_prob).long()            # 1保持/0翻转
        return x_t_pres                                  # (ΣN,1)

    # --------------------------------------------------
    def __call__(self, batch: List[Data]):
        # 1) Graph merge (keeps .dna list for later)
        pyg = Batch.from_data_list(batch)

        # 2) Pad DNA → (B,L,768) + pad_mask
        dna_tensor, pad_mask = self._pad_dna([d.dna for d in batch])
        rand_mask = self._random_mask(pad_mask)

        pyg.dna           = dna_tensor
        pyg.dna_pad_mask  = pad_mask
        pyg.dna_rand_mask = rand_mask

        # 3) 拼回完整节点特征，供 EGNN 使用
        pyg.x_full = torch.cat([pyg.x0_abun, pyg.x0_pres.float(), pyg.x_static], dim=1)   # (ΣN,4)

        # 4) 加噪 **分离建模** ---
        B = len(batch)
        t_idx = torch.randint(0, len(self.betas), (B,), device=pyg.x0_abun.device)
        
        # --- 连续流：log_abun ---
        pyg.x_t, pyg.noise = self._add_noise(pyg.x0_abun, t_idx, pyg.batch)
        
        # --- 离散流：presence Bernoulli 扩散 ---
        pyg.x_t_pres = self._bernoulli_noisy(pyg.x0_pres, t_idx, pyg.batch)
        
        # 拼显式 mask 特征（float）
        pyg.mask_feat = pyg.x_t_pres.float()

        pyg.t_idx = t_idx
        return pyg
    
class TreeDiffusionDataModule:

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg       = cfg
        self.data_cfg  = cfg["data"]
        self.world     = dist.get_world_size() if dist.is_initialized() else 1

        # ---------- betas ----------
        T = cfg["T"]
        if cfg.get("beta_schedule", "linear") == "cosine":
            self.betas = cosine_beta_schedule(T)  # 保持在 CPU
        else:
            self.betas = linear_beta_schedule(
                T, cfg["beta_start"], cfg["beta_end"]
            )  # 保持在 CPU

        # ---------- collate instance ----------
        self.collate_fn = GaussianDiffusionCollate(
            self.betas,
            rand_mask_prob = cfg.get("rand_mask_p", 0.20),
            keep_min       = cfg.get("keep_min", 128)
        )

        # place-holders set in setup()
        self.train_set: Optional[TreeDiffusionDataset] = None
        self.val_set  : Optional[TreeDiffusionDataset] = None

    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None):
        # 1) 共享树数据集（pkl or split 目录）
        if self.data_cfg.get("type", "pkl") == "split":
            tree_ds = TreeSplitDataset(root_dir=self.data_cfg["tree_dir"])
        else:
            tree_ds = TreeEGNNDataset(dataset_dir=self.data_cfg["tree_dir"])

        # 2) DNA datasets
        train_dna = DNABERTEmbeddingDataset(
            meta_csv = self.data_cfg["train_meta"],
            dna_dir  = self.data_cfg["dna_dir"],
            dna_ext  = self.data_cfg["dna_ext"],
            max_genes= self.data_cfg.get("max_genes")
        )
        val_dna   = DNABERTEmbeddingDataset(
            meta_csv = self.data_cfg["val_meta"],
            dna_dir  = self.data_cfg["dna_dir"],
            dna_ext  = self.data_cfg["dna_ext"],
            max_genes= self.data_cfg.get("max_genes")
        )

        # 3) Pair to final datasets
        self.train_set = TreeDiffusionDataset(train_dna, tree_ds)
        self.val_set   = TreeDiffusionDataset(val_dna,   tree_ds)

    # ------------------------------------------------------------------
    def _loader(self, ds, shuffle: bool):
        sampler = (
            DistributedSampler(ds, shuffle=shuffle) if self.world > 1 else None
        )
        return DataLoader(
            ds,
            batch_size = self.cfg["train"]["batch_size"],
            shuffle    = False if sampler else shuffle,
            sampler    = sampler,
            num_workers= self.cfg.get("num_workers", 4),
            pin_memory = self.cfg.get("pin_memory", False),
            prefetch_factor = self.cfg.get("prefetch_factor", 2),
            collate_fn = self.collate_fn,
        )

    # Lightning-style hooks
    def train_dataloader(self): return self._loader(self.train_set, True)
    def val_dataloader  (self): return self._loader(self.val_set,   False)