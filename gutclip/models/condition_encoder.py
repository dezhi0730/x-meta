#!/usr/bin/env python3
"""
条件编码器工厂函数
专门为DNA + Bio prompt + Tech prompt三种embedding组合设计
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class BaseConditionEncoder(nn.Module, ABC):
    """条件编码器基类"""
    
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
    
    @abstractmethod
    def encode(self, dna_embedding: torch.Tensor, sample_id: str, condition_text: Optional[str] = None) -> torch.Tensor:
        """编码条件"""
        pass
    
    @abstractmethod
    def encode_batch(self, dna_embeddings: torch.Tensor, sample_ids: list, condition_texts: Optional[list] = None) -> torch.Tensor:
        """批量编码条件"""
        pass
    
    @abstractmethod
    def get_unconditional_embedding(self) -> torch.Tensor:
        """获取无条件嵌入"""
        pass


class SimpleConditionEncoder(BaseConditionEncoder):
    """简单条件编码器（用于测试）"""
    
    def __init__(self, embedding_dim: int = 256, seed: int = 42):
        super().__init__(embedding_dim)
        torch.manual_seed(seed)
        self.unconditional_embedding = nn.Parameter(
            torch.randn(embedding_dim) * 0.1
        )
    
    def encode(self, dna_embedding: torch.Tensor, sample_id: str, condition_text: Optional[str] = None) -> torch.Tensor:
        """简单编码：基于DNA embedding"""
        return dna_embedding[:self.embedding_dim] if dna_embedding.size(0) >= self.embedding_dim else torch.randn(self.embedding_dim) * 0.1
    
    def encode_batch(self, dna_embeddings: torch.Tensor, sample_ids: list, condition_texts: Optional[list] = None) -> torch.Tensor:
        """批量编码"""
        batch_size = dna_embeddings.size(0)
        if dna_embeddings.size(1) >= self.embedding_dim:
            return dna_embeddings[:, :self.embedding_dim]
        else:
            return torch.randn(batch_size, self.embedding_dim) * 0.1
    
    def get_unconditional_embedding(self) -> torch.Tensor:
        """获取无条件嵌入"""
        return self.unconditional_embedding


def create_condition_encoder(encoder_type: str = "multi_condition", config_file: str = None, **kwargs) -> BaseConditionEncoder:
    """工厂函数：根据配置创建条件编码器"""
    
    # 加载编码器配置文件
    encoder_configs = {}
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                import yaml
                encoder_configs = yaml.safe_load(f)
        except Exception as e:
            print(f"[WARNING] 无法加载编码器配置文件 {config_file}: {e}")
    
    # 默认配置
    default_configs = {
        "simple": {
            "class": SimpleConditionEncoder,
            "default_params": {"embedding_dim": 256, "seed": 42}
        },
        "multi_condition": {
            "class": None,  # 需要特殊处理
            "default_params": {
                "output_dim": 256,
                "bio_bundle_path": "/data/home/wudezhi/project/school/x-meta/datasets/raw/prompt/bio_bundle_v1.npz",
                "bio_id2idx_path": "/data/home/wudezhi/project/school/x-meta/datasets/raw/prompt/bio_id2idx_v1.pkl",
                "bio_manifest_path": "/data/home/wudezhi/project/school/x-meta/datasets/raw/prompt/bio_manifest_v1.csv",
                "tech_bundle_path": None,
                "tech_id2idx_path": None,
                "tech_manifest_path": None,
                "dna_dim": 768
            }
        }
    }
    
    # 合并配置文件中的参数
    if encoder_type in encoder_configs:
        config = encoder_configs[encoder_type]
        params = {**default_configs[encoder_type]["default_params"], **config}
    else:
        params = default_configs[encoder_type]["default_params"]
    
    # 合并传入的参数
    params.update(kwargs)
    
    # 特殊处理multi_condition类型
    if encoder_type == "multi_condition":
        from gutclip.models.multi_condition_encoder import create_multi_condition_encoder
        return create_multi_condition_encoder(**params)
    
    # 创建编码器
    encoder_class = default_configs[encoder_type]["class"]
    return encoder_class(**params)


# 测试函数
def test_condition_encoders():
    """测试各种条件编码器"""
    print("=== 测试条件编码器 ===")
    
    # 测试简单编码器
    print("\n1. 测试简单编码器")
    encoder = create_condition_encoder("simple", embedding_dim=256)
    dna_embedding = torch.randn(768)
    embedding = encoder.encode(dna_embedding, "test_sample")
    print(f"简单编码器输出形状: {embedding.shape}")
    
    # 测试多条件编码器
    print("\n2. 测试多条件编码器")
    try:
        encoder = create_condition_encoder("multi_condition", output_dim=256)
        dna_embedding = torch.randn(768)
        embedding = encoder.encode(dna_embedding, "MV_FEI1_t1Q14")
        print(f"多条件编码器输出形状: {embedding.shape}")
    except Exception as e:
        print(f"多条件编码器测试失败: {e}")
    
    print("\n✅ 条件编码器测试完成")


if __name__ == "__main__":
    test_condition_encoders()
