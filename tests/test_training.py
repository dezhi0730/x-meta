# tests/test_training.py
"""
pytest -q tests/test_training.py
测试：
1. 模型前向传播
2. 损失计算
3. 反向传播
4. 优化器更新
5. 一个完整的训练循环
"""
import pytest
import torch
import numpy as np
from pathlib import Path
from gutclip.data import GutDataModule
from gutclip.models import GutCLIPModel
from gutclip.engine.train import train_one_epoch, evaluate
from gutclip.loss import CLIPLoss
import os
import torch.distributed as dist


def test_model_forward(cfg, device):
    """测试模型前向传播"""
    # 创建数据加载器
    dm = GutDataModule(cfg)
    train_loader = dm.train_dataloader()
    
    # 获取一个批次的数据
    batch = next(iter(train_loader))
    batch = batch.to(device)  # PyG Batch 对象可以直接移到设备上
    
    # 创建模型
    model = GutCLIPModel(tree_dim=cfg.tree_dim, dna_dim=cfg.dna_dim, output_dict=True).to(device)
    
    # 前向传播
    out = model(batch)
    
    # 检查输出
    assert "tree_emb" in out
    assert "dna_emb" in out
    assert "logit_scale" in out
    assert out["tree_emb"].shape == (cfg.batch_size, cfg.embed_dim)
    assert out["dna_emb"].shape == (cfg.batch_size, cfg.embed_dim)
    assert out["logit_scale"].shape == ()


def test_loss_computation(cfg, device):
    """测试损失计算"""
    # 创建数据加载器
    dm = GutDataModule(cfg)
    train_loader = dm.train_dataloader()
    
    # 获取一个批次的数据
    batch = next(iter(train_loader))
    batch = batch.to(device)
    
    # 创建模型
    model = GutCLIPModel(tree_dim=cfg.tree_dim, dna_dim=cfg.dna_dim, output_dict=True).to(device)
    
    # 前向传播
    out = model(batch)
    
    # 计算损失
    loss_fn = CLIPLoss(local_loss=cfg.local_loss).to(device)
    loss = loss_fn(out)
    
    # 检查损失
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # 标量
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_backprop(cfg, device):
    """测试反向传播和优化器更新"""
    # 创建数据加载器
    dm = GutDataModule(cfg)
    train_loader = dm.train_dataloader()
    
    # 获取一个批次的数据
    batch = next(iter(train_loader))
    batch = batch.to(device)
    
    # 创建模型和优化器
    model = GutCLIPModel(tree_dim=cfg.tree_dim, dna_dim=cfg.dna_dim, output_dict=True).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    
    # 前向传播
    out = model(batch)
    loss_fn = CLIPLoss(local_loss=cfg.local_loss).to(device)
    loss = loss_fn(out)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 检查梯度
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
        assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
    
    # 优化器更新
    optimizer.step()


def test_training_loop(cfg, device):
    """测试完整的训练循环"""
    # 创建数据加载器
    dm = GutDataModule(cfg)
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    
    # 创建模型和优化器
    model = GutCLIPModel(tree_dim=cfg.tree_dim, dna_dim=cfg.dna_dim, output_dict=True).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    # 修复 GradScaler 警告
    scaler = torch.amp.GradScaler('cuda') if cfg.precision == "amp" else None
    
    # 训练一个epoch
    train_loss = train_one_epoch(model, train_loader, optimizer, 0, device, cfg, scaler)
    assert isinstance(train_loss, float)
    assert not torch.isnan(torch.tensor(train_loss))
    assert not torch.isinf(torch.tensor(train_loss))
    
    # 验证
    val_loss = evaluate(model, val_loader, device, cfg)
    assert isinstance(val_loss, float)
    assert not torch.isnan(torch.tensor(val_loss))
    assert not torch.isinf(torch.tensor(val_loss))

