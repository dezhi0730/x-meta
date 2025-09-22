import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from .dna_dataset import DNABERTEmbeddingDataset
from .tree_dataset import TreeEGNNDataset
from .split_dataset import TreeSplitDataset
from .tree_diffusion_dataset import TreeDiffusionDataset, GaussianDiffusionCollate
from typing import List, Dict, Any, Optional
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import numpy as np
import pickle
import os


class PromptTreeDiffusionDataset(TreeDiffusionDataset):
    """带Prompt条件的Tree Diffusion Dataset
    
    继承自TreeDiffusionDataset，添加prompt embedding支持
    """
    
    def __init__(
        self,
        dna_dataset: DNABERTEmbeddingDataset, 
        tree_dataset: TreeEGNNDataset,
        prompt_bundle_path: str,
        prompt_id2idx_path: str,
        prompt_manifest_path: str,
        prompt_dim: int = 3072,
    ) -> None:
        super().__init__(dna_dataset, tree_dataset)
        
        self.prompt_bundle_path = prompt_bundle_path
        self.prompt_id2idx_path = prompt_id2idx_path
        self.prompt_manifest_path = prompt_manifest_path
        self.prompt_dim = prompt_dim
        
        # 加载prompt embeddings
        self._load_prompt_data()
        
        # 统计prompt可用性
        self._count_prompt_availability()
        
        print(f"[PromptTreeDiffusionDataset] 初始化完成")
        print(f"  - 有效样本数: {len(self.valid_sids)}")
        print(f"  - Prompt embedding维度: {self.prompt_dim}")
        print(f"  - 有prompt的样本数: {len(self.prompt_sid2idx)}")
        print(f"  - 无prompt的样本数: {len(self.valid_sids) - len(self.prompt_sid2idx)}")
        print(f"  - Prompt覆盖率: {len(self.prompt_sid2idx) / len(self.valid_sids) * 100:.1f}%")

    def _load_prompt_data(self):
        """加载prompt embeddings和相关映射"""
        # 加载prompt bundle
        bundle = np.load(self.prompt_bundle_path, allow_pickle=False)
        self.prompt_sample_ids = bundle['sample_id']
        self.prompt_embeddings = bundle['embedding']  # (N, 3072)
        
        # 加载id2idx映射
        with open(self.prompt_id2idx_path, 'rb') as f:
            self.prompt_id2idx = pickle.load(f)
        
        # 加载manifest
        import pandas as pd
        self.prompt_manifest = pd.read_csv(self.prompt_manifest_path)
        
        # 创建sample_id到prompt embedding的映射
        self.prompt_sid2idx = {}
        for sid in self.valid_sids:
            if sid in self.prompt_id2idx:
                self.prompt_sid2idx[sid] = self.prompt_id2idx[sid]
        
        print(f"  - Prompt bundle: {self.prompt_embeddings.shape}")
        print(f"  - Prompt ID2IDX: {len(self.prompt_id2idx)} 个条目")
        print(f"  - 匹配的样本: {len(self.prompt_sid2idx)} 个")

    def _count_prompt_availability(self):
        """统计prompt可用性（这个方法主要是为了保持接口一致性）"""
        # 对于 PromptTreeDiffusionDataset，prompt 可用性已经在 _load_prompt_data 中统计了
        # 这里不需要额外统计，因为 prompt_sid2idx 已经包含了所有有 prompt 的样本
        pass

    def __getitem__(self, idx: int) -> Data:
        """获取带prompt的数据项"""
        sid = self.valid_sids[idx]
        dna_rec = self.dna_dataset[self.dna_sid2idx[sid]]
        tree = self.tree_dataset[self.tree_sid2idx[sid]]

        # 按照用户要求只选特定列
        x_abun = tree.x[:, [2]].float()     # log_abun
        x_pres = tree.x[:, [1]].long()      # presence 0/1
        x_static = tree.x[:, [3,8]].float() # depth_sc, degree_sc

        # 获取prompt embedding
        if sid in self.prompt_sid2idx:
            prompt_idx = self.prompt_sid2idx[sid]
            prompt_embedding = torch.from_numpy(
                self.prompt_embeddings[prompt_idx]
            ).float()  # (3072,)
        else:
            # 如果没有prompt，使用零向量（静默处理，不打印警告）
            prompt_embedding = torch.zeros(self.prompt_dim, dtype=torch.float32)

        data = Data(
            x0_abun=x_abun,          # log_abun
            x0_pres=x_pres,          # presence 0/1
            x_static=x_static,       # 静态特征
            pos=tree.pos,
            edge_index=tree.edge_index,
            batch=torch.zeros(x_abun.size(0), dtype=torch.long),
            dna=dna_rec["embedding"].float(),  # (L_i,768)
            prompt=prompt_embedding,  # (3072,) prompt embedding
            sample_id=sid,
        )
        return data


class PromptGaussianDiffusionCollate(GaussianDiffusionCollate):
    """带Prompt的Gaussian Diffusion Collate函数"""
    
    def __init__(
        self,
        betas: torch.Tensor,  # (T,)
        rand_mask_prob: float = 0.20,
        keep_min: int = 128,
        prompt_dim: int = 3072,
    ) -> None:
        super().__init__(betas, rand_mask_prob, keep_min)
        self.prompt_dim = prompt_dim

    def __call__(self, batch: List[Data]):
        """处理带prompt的batch"""
        # 1) Graph merge (keeps .dna and .prompt lists for later)
        pyg = Batch.from_data_list(batch)

        # 2) Pad DNA → (B,L,768) + pad_mask
        dna_tensor, pad_mask = self._pad_dna([d.dna for d in batch])
        rand_mask = self._random_mask(pad_mask)

        pyg.dna = dna_tensor
        pyg.dna_pad_mask = pad_mask
        pyg.dna_rand_mask = rand_mask

        # 3) 处理prompt embeddings
        prompt_tensor = torch.stack([d.prompt for d in batch], dim=0)  # (B, 3072)
        pyg.prompt = prompt_tensor

        # 4) 拼回完整节点特征，供 EGNN 使用
        pyg.x_full = torch.cat([pyg.x0_abun, pyg.x0_pres.float(), pyg.x_static], dim=1)   # (ΣN,4)

        # 5) 加噪 **分离建模** ---
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


class PromptTreeDiffusionDataModule:
    """带Prompt的Tree Diffusion Data Module"""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.data_cfg = cfg["data"]
        self.world = dist.get_world_size() if dist.is_initialized() else 1

        # Prompt相关配置
        self.prompt_cfg = cfg.get("prompt", {})
        self.prompt_bundle_path = self.prompt_cfg.get("bundle_path")
        self.prompt_id2idx_path = self.prompt_cfg.get("id2idx_path") 
        self.prompt_manifest_path = self.prompt_cfg.get("manifest_path")
        self.prompt_dim = self.prompt_cfg.get("dim", 3072)

        # 验证prompt配置
        if not all([self.prompt_bundle_path, self.prompt_id2idx_path, self.prompt_manifest_path]):
            raise ValueError("Prompt配置不完整，需要bundle_path, id2idx_path, manifest_path")

        # ---------- betas ----------
        from gutclip.diffusion import linear_beta_schedule, cosine_beta_schedule
        T = cfg["T"]
        if cfg.get("beta_schedule", "linear") == "cosine":
            self.betas = cosine_beta_schedule(T)  # 保持在 CPU
        else:
            self.betas = linear_beta_schedule(
                T, cfg["beta_start"], cfg["beta_end"]
            )  # 保持在 CPU

        # ---------- collate instance ----------
        self.collate_fn = PromptGaussianDiffusionCollate(
            self.betas,
            rand_mask_prob=cfg.get("rand_mask_p", 0.20),
            keep_min=cfg.get("keep_min", 128),
            prompt_dim=self.prompt_dim
        )

        # place-holders set in setup()
        self.train_set: Optional[PromptTreeDiffusionDataset] = None
        self.val_set: Optional[PromptTreeDiffusionDataset] = None

    def setup(self, stage: Optional[str] = None):
        """设置数据集"""
        # 1) 共享树数据集（pkl or split 目录）
        if self.data_cfg.get("type", "pkl") == "split":
            tree_ds = TreeSplitDataset(root_dir=self.data_cfg["tree_dir"])
        else:
            tree_ds = TreeEGNNDataset(dataset_dir=self.data_cfg["tree_dir"])

        # 2) DNA datasets
        train_dna = DNABERTEmbeddingDataset(
            meta_csv=self.data_cfg["train_meta"],
            dna_dir=self.data_cfg["dna_dir"],
            dna_ext=self.data_cfg["dna_ext"],
            max_genes=self.data_cfg.get("max_genes")
        )
        val_dna = DNABERTEmbeddingDataset(
            meta_csv=self.data_cfg["val_meta"],
            dna_dir=self.data_cfg["dna_dir"],
            dna_ext=self.data_cfg["dna_ext"],
            max_genes=self.data_cfg.get("max_genes")
        )

        # 3) Pair to final datasets with prompt support
        self.train_set = PromptTreeDiffusionDataset(
            train_dna, tree_ds,
            prompt_bundle_path=self.prompt_bundle_path,
            prompt_id2idx_path=self.prompt_id2idx_path,
            prompt_manifest_path=self.prompt_manifest_path,
            prompt_dim=self.prompt_dim
        )
        self.val_set = PromptTreeDiffusionDataset(
            val_dna, tree_ds,
            prompt_bundle_path=self.prompt_bundle_path,
            prompt_id2idx_path=self.prompt_id2idx_path,
            prompt_manifest_path=self.prompt_manifest_path,
            prompt_dim=self.prompt_dim
        )

    def _loader(self, ds, shuffle: bool):
        """创建DataLoader"""
        sampler = (
            DistributedSampler(ds, shuffle=shuffle) if self.world > 1 else None
        )
        return DataLoader(
            ds,
            batch_size=self.cfg["train"]["batch_size"],
            shuffle=False if sampler else shuffle,
            sampler=sampler,
            num_workers=self.cfg.get("num_workers", 4),
            pin_memory=self.cfg.get("pin_memory", False),
            prefetch_factor=self.cfg.get("prefetch_factor", 2),
            collate_fn=self.collate_fn,
        )

    # Lightning-style hooks
    def train_dataloader(self): 
        return self._loader(self.train_set, True)
    
    def val_dataloader(self): 
        return self._loader(self.val_set, False)
