#!/usr/bin/env python3
"""
多条件编码器
组合DNA embedding + Bio prompt embedding + Tech prompt embedding
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pickle
from typing import Optional, Dict, Any
from pathlib import Path


class MultiConditionEncoder(nn.Module):
    """
    灵活的多条件编码器
    
    支持任意组合的条件：
    1. DNA embedding (768维) - 可选
    2. Bio prompt embedding (3072维) - 可选
    3. Tech prompt embedding (待定维数) - 可选
    
    输出统一的条件向量 (256维)
    """
    
    def __init__(
        self,
        # Bio prompt配置 (可选)
        bio_bundle_path: Optional[str] = None,
        bio_id2idx_path: Optional[str] = None,
        bio_manifest_path: Optional[str] = None,
        
        # Tech prompt配置 (可选)
        tech_bundle_path: Optional[str] = None,
        tech_id2idx_path: Optional[str] = None,
        tech_manifest_path: Optional[str] = None,
        
        # 输出配置
        output_dim: int = 256,
        dropout: float = 0.1,
        
        # DNA配置
        dna_dim: int = 768,
        use_dna: bool = True,
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.dna_dim = dna_dim
        self.use_dna = use_dna
        
        # 加载Bio prompt数据（如果提供）
        self.use_bio_prompt = bio_bundle_path is not None
        if self.use_bio_prompt:
            self._load_bio_prompt_data(bio_bundle_path, bio_id2idx_path, bio_manifest_path)
        
        # 加载Tech prompt数据（如果提供）
        self.use_tech_prompt = tech_bundle_path is not None
        self.tech_prompt_dim = 0  # 默认值
        if self.use_tech_prompt:
            self._load_tech_prompt_data(tech_bundle_path, tech_id2idx_path, tech_manifest_path)
        
        # 计算总输入维度
        total_input_dim = 0
        if self.use_dna:
            total_input_dim += dna_dim
        if self.use_bio_prompt:
            total_input_dim += 3072
        if self.use_tech_prompt:
            total_input_dim += self.tech_prompt_dim
        
        # 如果没有提供任何条件，使用默认维度
        if total_input_dim == 0:
            total_input_dim = 256  # 默认维度
        
        # 投影层：组合所有embedding -> 统一输出
        self.projection = nn.Sequential(
            nn.Linear(total_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # 无条件嵌入（用于Classifier-free Guidance）
        self.unconditional_embedding = nn.Parameter(
            torch.randn(output_dim) * 0.1
        )
        
        print(f"[MultiConditionEncoder] 初始化完成")
        print(f"  - 使用DNA: {self.use_dna} (维度: {dna_dim if self.use_dna else 0})")
        print(f"  - 使用Bio prompt: {self.use_bio_prompt} (维度: {3072 if self.use_bio_prompt else 0})")
        if self.use_tech_prompt:
            print(f"  - 使用Tech prompt: {self.use_tech_prompt} (维度: {self.tech_prompt_dim})")
        print(f"  - 总输入维度: {total_input_dim}")
        print(f"  - 输出维度: {output_dim}")
        
        # 打印prompt数据统计信息
        if self.use_bio_prompt:
            print(f"  - Bio prompt样本数: {len(self.bio_id2idx)}")
        if self.use_tech_prompt:
            print(f"  - Tech prompt样本数: {len(self.tech_id2idx)}")
    
    def _load_bio_prompt_data(self, bundle_path: str, id2idx_path: str, manifest_path: str):
        """加载Bio prompt数据"""
        # 尝试加载安全格式的bundle
        safe_bundle_path = bundle_path.replace('.npz', '_safe.npz')
        safe_sample_ids_path = bundle_path.replace('.npz', '_safe_sample_ids.pkl')
        
        if os.path.exists(safe_bundle_path):
            # 使用安全格式
            bundle = np.load(safe_bundle_path, allow_pickle=False)
            self.bio_embeddings = bundle['embedding']
            
            # 加载sample_ids
            with open(safe_sample_ids_path, 'rb') as f:
                self.bio_sample_ids = pickle.load(f)
            
            print(f"✅ 使用安全格式加载Bio prompt数据: {self.bio_embeddings.shape}")
        else:
            # 回退到原始格式
            bundle = np.load(bundle_path, allow_pickle=True)
            self.bio_sample_ids = bundle['sample_id']
            self.bio_embeddings = bundle['embedding']
            print(f"⚠️ 使用原始格式加载Bio prompt数据: {self.bio_embeddings.shape}")
        
        # 加载id2idx映射
        with open(id2idx_path, 'rb') as f:
            self.bio_id2idx = pickle.load(f)
        
        # 加载manifest
        import pandas as pd
        self.bio_manifest = pd.read_csv(manifest_path)
    
    def _load_tech_prompt_data(self, bundle_path: str, id2idx_path: str, manifest_path: str):
        """加载Tech prompt数据"""
        # 加载bundle
        bundle = np.load(bundle_path, allow_pickle=False)
        self.tech_sample_ids = bundle['sample_id']
        self.tech_embeddings = bundle['embedding']
        self.tech_prompt_dim = self.tech_embeddings.shape[1]
        
        # 加载id2idx映射
        with open(id2idx_path, 'rb') as f:
            self.tech_id2idx = pickle.load(f)
        
        # 加载manifest
        import pandas as pd
        self.tech_manifest = pd.read_csv(manifest_path)
        
        print(f"Tech prompt数据: {self.tech_embeddings.shape}")
    
    def _get_bio_prompt_embedding(self, sample_id: str) -> torch.Tensor:
        """获取Bio prompt embedding"""
        if sample_id in self.bio_id2idx:
            idx = self.bio_id2idx[sample_id]
            return torch.from_numpy(self.bio_embeddings[idx]).float()
        else:
            return torch.zeros(3072, dtype=torch.float32)
    
    def _get_tech_prompt_embedding(self, sample_id: str) -> torch.Tensor:
        """获取Tech prompt embedding"""
        if not self.use_tech_prompt:
            return torch.zeros(0, dtype=torch.float32)  # 空tensor
        
        if sample_id in self.tech_id2idx:
            idx = self.tech_id2idx[sample_id]
            return torch.from_numpy(self.tech_embeddings[idx]).float()
        else:
            # 如果没有找到，使用零向量（静默处理，不打印警告）
            return torch.zeros(self.tech_prompt_dim, dtype=torch.float32)
    
    def encode(
        self, 
        dna_embedding: Optional[torch.Tensor] = None,
        sample_id: Optional[str] = None,
        condition_text: Optional[str] = None
    ) -> torch.Tensor:
        """
        编码多条件
        
        Args:
            dna_embedding: DNA embedding (768,) - 可选
            sample_id: 样本ID - 可选
            condition_text: 条件文本（可选，用于调试）
        
        Returns:
            组合条件嵌入 (output_dim,)
        """
        embeddings = []
        
        # 添加DNA embedding（如果提供且启用）
        if self.use_dna and dna_embedding is not None:
            embeddings.append(dna_embedding)
        elif self.use_dna:
            # 如果没有提供DNA embedding，使用零向量
            embeddings.append(torch.zeros(self.dna_dim, dtype=torch.float32))
        
        # 添加Bio prompt embedding（如果启用）
        if self.use_bio_prompt:
            bio_embedding = self._get_bio_prompt_embedding(sample_id or "")
            embeddings.append(bio_embedding)
        
        # 添加Tech prompt embedding（如果启用）
        if self.use_tech_prompt:
            tech_embedding = self._get_tech_prompt_embedding(sample_id or "")
            embeddings.append(tech_embedding)
        
        # 如果没有提供任何embedding，使用默认向量
        if not embeddings:
            default_embedding = torch.randn(256, dtype=torch.float32) * 0.1
            return self.projection(default_embedding)
        
        # 确保所有embedding在同一设备上
        device = embeddings[0].device
        embeddings = [emb.to(device) for emb in embeddings]
        
        # 确保投影层也在同一设备上
        self.projection = self.projection.to(device)
        
        # 组合所有embedding
        combined_embedding = torch.cat(embeddings, dim=0)
        
        # 投影到输出维度
        return self.projection(combined_embedding)
    
    def encode_batch(
        self, 
        dna_embeddings: Optional[torch.Tensor] = None,  # (batch_size, 768) - 可选
        sample_ids: Optional[list] = None,
        condition_texts: Optional[list] = None
    ) -> torch.Tensor:
        """
        批量编码多条件
        
        Args:
            dna_embeddings: DNA embeddings (batch_size, 768) - 可选
            sample_ids: 样本ID列表 - 可选
            condition_texts: 条件文本列表（可选）
        
        Returns:
            组合条件嵌入 (batch_size, output_dim)
        """
        # 确定batch_size
        if dna_embeddings is not None:
            batch_size = dna_embeddings.size(0)
        elif sample_ids is not None:
            batch_size = len(sample_ids)
        else:
            batch_size = 1  # 默认batch_size
        
        combined_embeddings = []
        
        for i in range(batch_size):
            dna_emb = dna_embeddings[i] if dna_embeddings is not None else None
            sample_id = sample_ids[i] if sample_ids is not None else None
            condition_text = condition_texts[i] if condition_texts else None
            
            combined_emb = self.encode(dna_emb, sample_id, condition_text)
            combined_embeddings.append(combined_emb)
        
        return torch.stack(combined_embeddings)
    
    def get_unconditional_embedding(self) -> torch.Tensor:
        """获取无条件嵌入（用于Classifier-free Guidance）"""
        return self.unconditional_embedding
    
    def forward(
        self, 
        dna_embedding: torch.Tensor,
        sample_id: str,
        condition_text: Optional[str] = None
    ) -> torch.Tensor:
        """前向传播"""
        return self.encode(dna_embedding, sample_id, condition_text)


def create_multi_condition_encoder(
    bio_bundle_path: Optional[str] = None,
    bio_id2idx_path: Optional[str] = None,
    bio_manifest_path: Optional[str] = None,
    tech_bundle_path: Optional[str] = None,
    tech_id2idx_path: Optional[str] = None,
    tech_manifest_path: Optional[str] = None,
    output_dim: int = 256,
    **kwargs
) -> MultiConditionEncoder:
    """
    创建多条件编码器
    
    Args:
        bio_bundle_path: Bio prompt bundle路径
        bio_id2idx_path: Bio prompt id2idx路径
        bio_manifest_path: Bio prompt manifest路径
        tech_bundle_path: Tech prompt bundle路径（可选）
        tech_id2idx_path: Tech prompt id2idx路径（可选）
        tech_manifest_path: Tech prompt manifest路径（可选）
        output_dim: 输出维度
        **kwargs: 其他参数
    
    Returns:
        MultiConditionEncoder实例
    """
    return MultiConditionEncoder(
        bio_bundle_path=bio_bundle_path,
        bio_id2idx_path=bio_id2idx_path,
        bio_manifest_path=bio_manifest_path,
        tech_bundle_path=tech_bundle_path,
        tech_id2idx_path=tech_id2idx_path,
        tech_manifest_path=tech_manifest_path,
        output_dim=output_dim,
        **kwargs
    )


# 测试函数
def test_multi_condition_encoder():
    """测试多条件编码器的各种组合"""
    print("=== 测试多条件编码器 ===")
    
    # Bio prompt路径
    bio_bundle = "/data/home/wudezhi/project/school/x-meta/datasets/raw/prompt/bio_bundle_v1.npz"
    bio_id2idx = "/data/home/wudezhi/project/school/x-meta/datasets/raw/prompt/bio_id2idx_v1.pkl"
    bio_manifest = "/data/home/wudezhi/project/school/x-meta/datasets/raw/prompt/bio_manifest_v1.csv"
    
    try:
        # 测试1: 只有Bio prompt
        print("\n1. 测试只有Bio prompt")
        encoder1 = create_multi_condition_encoder(
            bio_bundle_path=bio_bundle,
            bio_id2idx_path=bio_id2idx,
            bio_manifest_path=bio_manifest,
            use_dna=False,
            output_dim=256
        )
        
        embedding1 = encoder1.encode(sample_id="MV_FEI1_t1Q14")
        print(f"只有Bio prompt编码形状: {embedding1.shape}")
        
        # 测试2: DNA + Bio prompt
        print("\n2. 测试DNA + Bio prompt")
        encoder2 = create_multi_condition_encoder(
            bio_bundle_path=bio_bundle,
            bio_id2idx_path=bio_id2idx,
            bio_manifest_path=bio_manifest,
            use_dna=True,
            output_dim=256
        )
        
        dna_embedding = torch.randn(768)
        embedding2 = encoder2.encode(dna_embedding, "MV_FEI1_t1Q14")
        print(f"DNA + Bio prompt编码形状: {embedding2.shape}")
        
        # 测试3: 只有DNA
        print("\n3. 测试只有DNA")
        encoder3 = create_multi_condition_encoder(
            use_dna=True,
            output_dim=256
        )
        
        embedding3 = encoder3.encode(dna_embedding)
        print(f"只有DNA编码形状: {embedding3.shape}")
        
        # 测试4: 没有任何条件
        print("\n4. 测试没有任何条件")
        encoder4 = create_multi_condition_encoder(
            use_dna=False,
            output_dim=256
        )
        
        embedding4 = encoder4.encode()
        print(f"无条件编码形状: {embedding4.shape}")
        
        # 测试批量编码
        print("\n5. 测试批量编码")
        batch_size = 3
        dna_embeddings = torch.randn(batch_size, 768)
        sample_ids = ["MV_FEI1_t1Q14", "MV_FEI2_t1Q14", "MV_FEI3_t1Q14"]
        
        batch_embeddings = encoder2.encode_batch(dna_embeddings, sample_ids)
        print(f"批量编码形状: {batch_embeddings.shape}")
        
        # 测试无条件嵌入
        uncond_embedding = encoder2.get_unconditional_embedding()
        print(f"无条件嵌入形状: {uncond_embedding.shape}")
        
        print("\n✅ 多条件编码器测试通过!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_multi_condition_encoder()
