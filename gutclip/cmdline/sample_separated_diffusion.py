#!/usr/bin/env python3
"""
分离建模扩散模型采样示例

使用方法：
python -m gutclip.cmdline.sample_separated_diffusion --ckpt checkpoints/tree_diffusion/latest.pt --config configs/train_separated_diffusion.yaml
"""

import os
import torch
import argparse
from pathlib import Path
from omegaconf import OmegaConf
import json

from gutclip.data.tree_diffusion_dataset import TreeDiffusionDataModule
from gutclip.models.diffusion.separated_unet import SeparatedDiffusionModel, SeparatedDiffusionModelWithDNA
from gutclip.diffusion.separated_sampler import SeparatedDiffusionSampler
from gutclip.diffusion import linear_beta_schedule, cosine_beta_schedule


def main():
    parser = argparse.ArgumentParser(description="分离建模扩散模型采样")
    parser.add_argument("--ckpt", type=str, required=True, help="检查点路径")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--num_steps", type=int, default=50, help="采样步数")
    parser.add_argument("--temperature", type=float, default=1.0, help="温度参数")
    parser.add_argument("--guidance_scale", type=float, default=0.0, help="引导强度")
    parser.add_argument("--output_dir", type=str, default="samples", help="输出目录")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # 加载配置
    cfg = OmegaConf.load(args.config)
    
    # 创建数据模块
    data_module = TreeDiffusionDataModule(cfg)
    data_module.setup()
    
    val_loader = data_module.val_dataloader()
    
    # 创建模型
    if cfg.get("use_dna", False):
        model = SeparatedDiffusionModelWithDNA(
            input_dim=4,
            dna_dim=cfg.get("dna_dim", 768),
            hidden_dim=cfg.model.hidden_dim,
            out_dim=cfg.model.out_dim,
            num_layers=cfg.model.num_layers,
            dropout_rate=cfg.model.dropout_rate
        )
    else:
        model = SeparatedDiffusionModel(
            input_dim=4,
            hidden_dim=cfg.model.hidden_dim,
            out_dim=cfg.model.out_dim,
            num_layers=cfg.model.num_layers,
            dropout_rate=cfg.model.dropout_rate
        )
    
    # 加载检查点
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    
    print(f"[INFO] 加载检查点: {args.ckpt}")
    print(f"[INFO] 使用设备: {device}")
    
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
    
    # 创建采样器
    sampler = SeparatedDiffusionSampler(betas, device)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 开始采样
    print(f"[INFO] 开始采样，步数: {args.num_steps}")
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 5:  # 只采样前5个批次
                break
                
            # 移动数据到设备
            batch = batch.to(device, non_blocking=True)
            
            print(f"[INFO] 采样批次 {i+1}/5")
            
            # 采样
            if args.guidance_scale > 0:
                result = sampler.sample_with_guidance(
                    model, batch, 
                    guidance_scale=args.guidance_scale,
                    num_steps=args.num_steps,
                    temperature=args.temperature
                )
            else:
                result = sampler.sample(
                    model, batch,
                    num_steps=args.num_steps,
                    temperature=args.temperature
                )
            
            # 保存结果
            sample_path = output_dir / f"sample_{i:03d}.pt"
            torch.save({
                "x0_abun": result["x0_abun"].cpu(),
                "x0_pres": result["x0_pres"].cpu(),
                "presence_mask": result["presence_mask"].cpu(),
                "original_abun": batch.x0_abun.cpu(),
                "original_pres": batch.x0_pres.cpu(),
                "sample_id": getattr(batch, 'sample_ids', [])
            }, sample_path)
            
            print(f"[INFO] 保存样本到: {sample_path}")
            
            # 计算一些统计信息
            abun_mean = result["x0_abun"].mean().item()
            pres_ratio = result["presence_mask"].mean().item()
            print(f"[INFO] 样本统计 - Abundance均值: {abun_mean:.4f}, Presence比例: {pres_ratio:.4f}")
    
    print(f"[INFO] 采样完成! 结果保存在: {output_dir}")


if __name__ == "__main__":
    main() 