#!/usr/bin/env python3
"""
训练脚本 v2 - 使用 TreeDiffusionDataModule
"""

import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

# ===== local project imports =====
from gutclip.data.tree_diffusion_dataset import TreeDiffusionDataModule
from gutclip.models.diffusion.tree_noise_predictor import TreeNoisePredictor
from gutclip.engine.trainer_tree_diffusion import TreeDiffusionTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="gutclip/configs/train_tree_diffusion.yaml")
    p.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    p.add_argument("--tb-dir", type=str, default="runs", help="TensorBoard log directory")
    p.add_argument("--exp-name", type=str, default=None, help="Experiment name for TensorBoard")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.cfg))

    # 覆盖配置中的epochs
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"使用设备: {device}")
    print(f"配置: {cfg['name']}")
    print(f"训练轮数: {cfg['epochs']}")

    # ---------- TensorBoard 设置 ----------
    if args.exp_name is None:
        exp_name = f"{cfg['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        exp_name = args.exp_name
    
    tb_dir = os.path.join(args.tb_dir, exp_name)
    tb_writer = SummaryWriter(tb_dir)
    print(f"TensorBoard日志目录: {tb_dir}")
    
    # 记录配置到TensorBoard
    tb_writer.add_text("config", str(cfg), 0)

    # ---------- 数据模块 ----------
    print("正在设置数据模块...")
    data_module = TreeDiffusionDataModule(cfg)
    data_module.setup()
    
    print(f"训练集大小: {len(data_module.train_set)}")
    print(f"验证集大小: {len(data_module.val_set)}")

    # ---------- 模型 ----------
    print("正在创建模型...")
    model = TreeNoisePredictor(
        node_dim=2,  # 输出维度仍然是2，匹配noisy特征
        hid=128,
        dna_dim=768,
        t_emb_dim=128
    ).to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 记录模型信息到TensorBoard
    tb_writer.add_scalar("model/total_params", total_params, 0)
    tb_writer.add_scalar("model/trainable_params", trainable_params, 0)

    # ---------- 优化器 ----------
    print("正在设置优化器...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["wd"]
    )

    # ---------- 训练器 ----------
    print("正在创建训练器...")
    trainer = TreeDiffusionTrainer(
        model=model,
        loader=data_module.train_dataloader(),
        optimizer=optimizer,
        epochs=cfg["epochs"],
        betas=torch.linspace(cfg["beta_start"], cfg["beta_end"], cfg["T"]),
        cfg=cfg,
        device=device
    )

    # ---------- 开始训练 ----------
    print("开始训练...")
    try:
        trainer.fit()
        print("训练完成！")
    except KeyboardInterrupt:
        print("训练被用户中断")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        raise
    finally:
        # 关闭TensorBoard writer
        tb_writer.close()
        print(f"TensorBoard日志已保存到: {tb_dir}")


if __name__ == "__main__":
    main() 