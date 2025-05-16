# tests/conftest.py
import os
import pytest
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from pathlib import Path

@pytest.fixture(scope="session")
def cfg():
    """加载默认配置"""
    # 加载默认配置
    default_cfg = OmegaConf.load("gutclip/configs/default.yaml")
    
    # 为测试环境修改一些配置
    test_cfg = OmegaConf.create({
        "batch_size": 2,  # 测试时使用小批量
        "num_workers": 0,  # 测试时使用单进程
    })
    
    # 合并配置
    cfg = OmegaConf.merge(default_cfg, test_cfg)
    return cfg

@pytest.fixture(scope="session")
def device():
    """获取设备"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="session", autouse=True)
def setup_ddp():
    """设置分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)
    yield
    if dist.is_initialized():
        dist.destroy_process_group()
