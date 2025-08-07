#!/usr/bin/env python3
"""
训练分离建模扩散模型：Bernoulli + Gaussian 分离建模

使用方法：
python -m gutclip.cmdline.train_separated_diffusion --config configs/train_separated_diffusion.yaml
"""

import os
import torch
import argparse
from pathlib import Path
from omegaconf import OmegaConf

from gutclip.data.tree_diffusion_dataset import TreeDiffusionDataset, TreeDiffusionDataModule
from gutclip.models.diffusion.separated_unet import SeparatedDiffusionModel, SeparatedDiffusionModelWithDNA
from gutclip.engine.trainer_tree_diffusion import TreeDiffusionTrainer
from gutclip.diffusion import linear_beta_schedule, cosine_beta_schedule


def main():
    parser = argparse.ArgumentParser(description="训练分离建模扩散模型")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--pretrained_ckpt", type=str, default=None, help="预训练GutCLIP模型路径")
    parser.add_argument("--load_tree_encoder", action="store_true", help="是否也加载预训练的tree encoder")
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 加载配置
    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device)
    
    print(f"[INFO] 使用设备: {device}")
    print(f"[INFO] 配置: {cfg}")
    
    # 创建数据模块
    data_module = TreeDiffusionDataModule(cfg)
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    print(f"[INFO] 训练集大小: {len(train_loader.dataset)}")
    print(f"[INFO] 验证集大小: {len(val_loader.dataset)}")
    
    # 加载模型配置
    model_cfg_path = cfg.get("model_config", "gutclip/configs/model/separated_diffusion.yaml")
    if os.path.exists(model_cfg_path):
        model_cfg = OmegaConf.load(model_cfg_path)
        print(f"[INFO] 加载模型配置: {model_cfg_path}")
    else:
        model_cfg = {}
        print(f"[WARN] 模型配置文件不存在: {model_cfg_path}，使用默认配置")
    
    # 创建模型
    if cfg.get("use_dna", False):
        model = SeparatedDiffusionModelWithDNA(
            input_dim=4,  # 基础输入维度，DNA编码会在模型内部处理
            dna_dim=cfg.get("dna_dim", 768),
            hidden_dim=cfg.model.hidden_dim,
            out_dim=cfg.model.out_dim,
            num_layers=cfg.model.num_layers,
            dropout_rate=cfg.model.dropout_rate,
            pretrained_dna_encoder=True,
            dna_output_dim=cfg.get("dna_output_dim", None),  # 动态DNA输出维度
            model_cfg=model_cfg  # 传递模型配置
        )
        
        # 加载预训练编码器
        pretrained_ckpt = args.pretrained_ckpt or cfg.get("pretrained_ckpt", None)
        if pretrained_ckpt:
            print(f"[INFO] 加载预训练检查点: {pretrained_ckpt}")
            model.load_pretrained_encoders(pretrained_ckpt, load_tree_encoder=args.load_tree_encoder)
        else:
            print("[WARN] 未提供预训练检查点，编码器将使用随机初始化")
        
        print("[INFO] 使用带DNA条件的分离建模模型")
    else:
        model = SeparatedDiffusionModel(
            input_dim=4,
            hidden_dim=cfg.model.hidden_dim,
            out_dim=cfg.model.out_dim,
            num_layers=cfg.model.num_layers,
            dropout_rate=cfg.model.dropout_rate
        )
        print("[INFO] 使用基础分离建模模型")
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay
    )
    
    # 创建betas
    T = cfg.T
    if cfg.get("beta_schedule", "linear") == "cosine":
        betas = cosine_beta_schedule(T)
    else:
        betas = linear_beta_schedule(
            T, 
            cfg.get("beta_start", 1e-4), 
            cfg.get("beta_end", 0.02)
        )
    
    # 创建训练器
    trainer = TreeDiffusionTrainer(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        epochs=cfg.train.epochs,
        betas=betas,
        cfg=cfg,
        device=device
    )
    
    # 开始训练
    print("[INFO] 开始训练...")
    trainer.fit()
    
    print("[INFO] 训练完成!")


if __name__ == "__main__":
    main() 