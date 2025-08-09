#!/usr/bin/env python3
import os
import argparse
from omegaconf import OmegaConf
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from gutclip.data.tree_diffusion_dataset import TreeDiffusionDataModule
from gutclip.models.diffusion.separated_unet import (
    SeparatedDiffusionModel, SeparatedDiffusionModelWithDNA
)
from gutclip.engine.trainer_tree_diffusion import TreeDiffusionTrainer
from gutclip.diffusion import linear_beta_schedule, cosine_beta_schedule


def is_dist():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1

def setup_ddp(args_device: str):
    if is_dist():
        # 这两行能避免 NCCL “沉默挂死”
        os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "OFF")

        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device(args_device)
    return rank, world_size, local_rank, device

def main():
    parser = argparse.ArgumentParser(description="训练分离建模扩散模型")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--device", type=str, default="cuda", help="设备（单机单卡时使用）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--pretrained_ckpt", type=str, default=None, help="预训练GutCLIP模型路径")
    parser.add_argument("--load_tree_encoder", action="store_true", help="是否也加载预训练的tree encoder")
    args = parser.parse_args()

    rank, world, local_rank, device = setup_ddp(args.device)
    is_main = (rank == 0)

    # 种子：不同 rank 不同
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    cfg = OmegaConf.load(args.config)
    if is_main:
        print(f"[INFO] DDP world={world}, rank={rank}, local_rank={local_rank}")
        print(f"[INFO] 使用设备: {device}")
        print(f"[INFO] 配置: {cfg}")

    # === Data ===
    # 注意：DataModule 内部会根据 dist.is_initialized() 决定是否使用 DistributedSampler
    data_module = TreeDiffusionDataModule(cfg)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader   = data_module.val_dataloader()

    if is_main:
        print(f"[INFO] 训练集大小: {len(train_loader.dataset)}")
        print(f"[INFO] 验证集大小: {len(val_loader.dataset)}")

    # === Model ===
    # 先构建并 to(device)，再（如需）包 DDP；优化器在 DDP 之后创建
    if cfg.get("use_dna", False):
        model = SeparatedDiffusionModelWithDNA(
            input_dim=4,
            dna_dim=cfg.get("dna_dim", 768),
            hidden_dim=cfg.model.hidden_dim,
            out_dim=cfg.model.out_dim,
            num_layers=cfg.model.num_layers,
            dropout_rate=cfg.model.dropout_rate,
            pretrained_dna_encoder=True,
            dna_output_dim=cfg.get("dna_output_dim", None),
            model_cfg=OmegaConf.load(cfg.get("model_config", "gutclip/configs/model/separated_diffusion.yaml"))
                     if Path(cfg.get("model_config", "")).exists() else {}
        )
        pretrained_ckpt = args.pretrained_ckpt or cfg.get("pretrained_ckpt", None)
        if pretrained_ckpt and is_main:
            print(f"[INFO] 加载预训练检查点: {pretrained_ckpt}")
        if pretrained_ckpt:
            model.load_pretrained_encoders(pretrained_ckpt, load_tree_encoder=args.load_tree_encoder)
        elif is_main:
            print("[WARN] 未提供预训练检查点，编码器将使用随机初始化")
        if is_main:
            print("[INFO] 使用带DNA条件的分离建模模型")
    else:
        model = SeparatedDiffusionModel(
            input_dim=4,
            hidden_dim=cfg.model.hidden_dim,
            out_dim=cfg.model.out_dim,
            num_layers=cfg.model.num_layers,
            dropout_rate=cfg.model.dropout_rate
        )
        if is_main:
            print("[INFO] 使用基础分离建模模型")

    model.to(device)
    if is_dist():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # === betas ===
    T = cfg.T
    if cfg.get("beta_schedule", "linear") == "cosine":
        betas = cosine_beta_schedule(T)
    else:
        betas = linear_beta_schedule(T, cfg.get("beta_start", 1e-4), cfg.get("beta_end", 0.02))

    # === Optimizer（在 DDP 之后创建）===
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # === Trainer ===
    trainer = TreeDiffusionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        epochs=cfg.train.epochs,
        betas=betas,
        cfg=cfg,
        device=device,
    )

    if is_main:
        print("[INFO] 开始训练...")
    trainer.fit()
    if is_main:
        print("[INFO] 训练完成!")

    if is_dist():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()