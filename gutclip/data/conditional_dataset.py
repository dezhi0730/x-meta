#!/usr/bin/env python3
"""
条件扩散数据集
基于TreeDiffusionDataset，添加条件编码器支持
"""

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
import json
import os
from pathlib import Path


class ConditionalTreeDiffusionDataset(TreeDiffusionDataset):
    """带条件编码器的Tree Diffusion Dataset"""
    
    def __init__(
        self,
        dna_dataset: DNABERTEmbeddingDataset, 
        tree_dataset: TreeEGNNDataset,
        condition_encoder,
        condition_file: Optional[str] = None,
    ) -> None:
        super().__init__(dna_dataset, tree_dataset)
        
        self.condition_encoder = condition_encoder
        self.condition_file = condition_file
        
        # 加载条件数据（如果提供）
        self.conditions = self._load_conditions()
        
        # 统计有条件向量和无条件向量的样本数量
        self._count_prompt_availability()
        
        print(f"[ConditionalTreeDiffusionDataset] 初始化完成")
        print(f"  - 有效样本数: {len(self.valid_sids)}")
        print(f"  - 条件编码器: {type(condition_encoder).__name__}")
        print(f"  - 有条件向量样本: {self.conditional_count}")
        print(f"  - 无条件向量样本: {self.unconditional_count}")
        print(f"  - 条件向量覆盖率: {self.conditional_count / len(self.valid_sids) * 100:.1f}%")

    def _load_conditions(self) -> Dict[str, str]:
        """加载条件数据"""
        conditions = {}
        
        if self.condition_file and os.path.exists(self.condition_file):
            try:
                with open(self.condition_file, 'r') as f:
                    conditions = json.load(f)
                print(f"加载条件文件: {len(conditions)} 个条件")
            except Exception as e:
                print(f"[WARNING] 无法加载条件文件 {self.condition_file}: {e}")
        
        return conditions
    
    def _count_prompt_availability(self):
        """统计有条件向量和无条件向量的样本数量"""
        self.conditional_count = 0
        self.unconditional_count = 0
        
        print(f"[INFO] 正在统计样本的条件向量可用性...")
        
        for sid in self.valid_sids:
            try:
                # 尝试获取Bio prompt embedding
                if hasattr(self.condition_encoder, '_get_bio_prompt_embedding'):
                    embedding = self.condition_encoder._get_bio_prompt_embedding(sid)
                    if embedding is not None and not torch.allclose(embedding, torch.zeros_like(embedding)):
                        self.conditional_count += 1
                    else:
                        self.unconditional_count += 1
                else:
                    self.unconditional_count += 1
            except Exception:
                self.unconditional_count += 1
        
        print(f"[INFO] 统计完成: {self.conditional_count} 个有条件向量, {self.unconditional_count} 个无条件向量")
    
    def _get_bio_prompt_embedding(self, sample_id: str) -> torch.Tensor:
        """获取Bio prompt embedding"""
        try:
            # 使用条件编码器获取Bio prompt embedding
            if hasattr(self.condition_encoder, '_get_bio_prompt_embedding'):
                return self.condition_encoder._get_bio_prompt_embedding(sample_id)
            else:
                # 如果没有Bio prompt，返回零向量
                return torch.zeros(3072, dtype=torch.float32)
        except Exception:
            # 静默处理，不打印每个样本的警告
            return torch.zeros(3072, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Data:
        """获取带条件的数据项"""
        sid = self.valid_sids[idx]
        dna_rec = self.dna_dataset[self.dna_sid2idx[sid]]
        tree = self.tree_dataset[self.tree_sid2idx[sid]]

        # 按照用户要求只选特定列
        x_abun = tree.x[:, [2]].float()     # log_abun
        x_pres = tree.x[:, [1]].long()      # presence 0/1
        x_static = tree.x[:, [3,8]].float() # depth_sc, degree_sc

        # 获取条件文本
        condition_text = self.conditions.get(sid, "")
        
        # 获取Bio prompt embedding
        prompt_embedding = self._get_bio_prompt_embedding(sid)

        data = Data(
            x0_abun=x_abun,          # log_abun
            x0_pres=x_pres,          # presence 0/1
            x_static=x_static,       # 静态特征
            pos=tree.pos,
            edge_index=tree.edge_index,
            # batch索引由Batch.from_data_list自动生成，不要手动填写
            dna=dna_rec["embedding"].float(),  # (L_i,768)
            sample_id=sid,
            condition_text=condition_text,  # 条件文本
            prompt=prompt_embedding,  # Bio prompt embedding (3072,)
        )
        return data


class ConditionalGaussianDiffusionCollate(GaussianDiffusionCollate):
    """带条件编码的Gaussian Diffusion Collate函数"""
    
    def __init__(
        self,
        betas: torch.Tensor,
        condition_encoder,
        rand_mask_prob: float = 0.20,
        keep_min: int = 128,
        unconditional_prob: float = 0.1,
    ) -> None:
        super().__init__(betas, rand_mask_prob, keep_min)
        self.condition_encoder = condition_encoder
        self.unconditional_prob = unconditional_prob

    def __call__(self, batch: List[Data]):
        """处理带条件的batch"""
        B = len(batch)  # 批次大小
        
        # 1) Graph merge (keeps .dna list for later)
        pyg = Batch.from_data_list(batch)

        # 2) Pad DNA → (B,L,768) + pad_mask
        dna_tensor, pad_mask = self._pad_dna([d.dna for d in batch])
        rand_mask = self._random_mask(pad_mask)

        pyg.dna = dna_tensor
        pyg.dna_pad_mask = pad_mask
        pyg.dna_rand_mask = rand_mask

        # 3) 处理Bio prompt embeddings（直接使用raw prompt）
        prompt_embeddings = []
        unconditional_mask = []
        
        # 获取设备信息（从第一个tensor推断）
        device = batch[0].x0_abun.device
        
        for d in batch:
            # 获取Bio prompt embedding
            if hasattr(d, 'prompt') and d.prompt is not None:
                prompt_emb = d.prompt  # (3072,)
            else:
                # 如果没有prompt，使用零向量
                prompt_emb = torch.zeros(3072, dtype=torch.float32)
            
            prompt_embeddings.append(prompt_emb)
            
            # 随机选择无条件样本（用于Classifier-free Guidance）
            unconditional_mask.append(torch.rand(1, device=device) < self.unconditional_prob)
        
        # 堆叠prompt embeddings并移动到正确设备
        pyg.prompt_embeddings = torch.stack(prompt_embeddings).to(device)  # (B, 3072)
        pyg.unconditional_mask = torch.stack(unconditional_mask).squeeze().bool()  # (B,) bool类型
        
        # 添加sample_ids用于日志/审计（可选）
        pyg.sample_ids = [d.sample_id for d in batch]

        # 4) 拼回完整节点特征，供 EGNN 使用
        pyg.x_full = torch.cat([pyg.x0_abun, pyg.x0_pres.float(), pyg.x_static], dim=1)   # (ΣN,4)

        # 5) 加噪 **分离建模** ---
        t_idx = torch.randint(0, len(self.betas), (B,), device=pyg.x0_abun.device)
        
        # --- 连续流：log_abun ---
        pyg.x_t, pyg.noise = self._add_noise(pyg.x0_abun, t_idx, pyg.batch)
        
        # --- 离散流：presence Bernoulli 扩散 ---
        pyg.x_t_pres = self._bernoulli_noisy(pyg.x0_pres, t_idx, pyg.batch)
        
        # 注意：不再构造mask_feat，避免维度混乱

        pyg.t_idx = t_idx
        return pyg


class ConditionalTreeDiffusionDataModule:
    """带条件编码的Tree Diffusion Data Module"""

    def __init__(
        self, 
        cfg: Dict[str, Any],
        condition_encoder,
        condition_file: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.data_cfg = cfg["data"]
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.condition_encoder = condition_encoder
        self.condition_file = condition_file

        # ---------- betas ----------
        from gutclip.diffusion import linear_beta_schedule, cosine_beta_schedule
        T = cfg["T"]
        if cfg.get("beta_schedule", "linear") == "cosine":
            self.betas = cosine_beta_schedule(T)
        else:
            self.betas = linear_beta_schedule(
                T, cfg["beta_start"], cfg["beta_end"]
            )

        # ---------- collate instance ----------
        self.collate_fn = ConditionalGaussianDiffusionCollate(
            self.betas,
            condition_encoder=self.condition_encoder,
            rand_mask_prob=cfg.get("rand_mask_p", 0.20),
            keep_min=cfg.get("keep_min", 128),
            unconditional_prob=cfg.get("condition", {}).get("unconditional_prob", 0.1)
        )

        # place-holders set in setup()
        self.train_set: Optional[ConditionalTreeDiffusionDataset] = None
        self.val_set: Optional[ConditionalTreeDiffusionDataset] = None

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

        # 3) Pair to final datasets with condition support
        self.train_set = ConditionalTreeDiffusionDataset(
            train_dna, tree_ds, self.condition_encoder, self.condition_file
        )
        self.val_set = ConditionalTreeDiffusionDataset(
            val_dna, tree_ds, self.condition_encoder, self.condition_file
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


def create_sample_condition_file(output_path: str) -> str:
    """创建示例条件文件"""
    sample_conditions = {
        "MV_FEI1_t1Q14": "0y_infant / female / body_site=stool / disease=healthy / condition=control / normal_weight",
        "MV_FEI2_t1Q14": "0y_infant / male / body_site=stool / disease=healthy / condition=control / normal_weight",
        "MV_FEI3_t1Q14": "0y_infant / male / body_site=stool / disease=healthy / condition=control / normal_weight",
        "MV_FEI4_t1Q14": "1y_toddler / male / body_site=stool / disease=healthy / condition=control / normal_weight",
        "MV_FEM1_t1Q14": "34y_middle_adult / female / body_site=stool / disease=healthy / condition=control / normal_weight"
    }
    
    with open(output_path, 'w') as f:
        json.dump(sample_conditions, f, indent=2)
    
    return output_path
