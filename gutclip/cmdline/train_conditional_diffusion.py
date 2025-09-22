#!/usr/bin/env python3
import os
import argparse
from omegaconf import OmegaConf
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from gutclip.data.conditional_dataset import (
    ConditionalTreeDiffusionDataModule, 
    create_sample_condition_file
)
from gutclip.models.diffusion.conditional_separated_unet import (
    ConditionalSeparatedDiffusionModel, 
    ConditionalSeparatedDiffusionModelWithDNA
)
from gutclip.engine.conditional_trainer import ConditionalTreeDiffusionTrainer
from gutclip.models.condition_encoder import create_condition_encoder
from gutclip.diffusion import linear_beta_schedule, cosine_beta_schedule


def is_dist():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1

def setup_ddp(args_device: str):
    if is_dist():
        # 这两行能避免 NCCL "沉默挂死"
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
    parser = argparse.ArgumentParser(description="训练条件分离建模扩散模型")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--device", type=str, default="cuda", help="设备（单机单卡时使用）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--pretrained_ckpt", type=str, required=True, help="预训练separated diffusion模型路径")
    parser.add_argument("--condition_file", type=str, default=None, help="条件标注文件路径")
    parser.add_argument("--create_sample_conditions", action="store_true", help="创建示例条件文件")
    parser.add_argument("--unconditional_prob", type=float, default=0.1, help="Classifier-free guidance无条件概率")
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

    # === 创建示例条件文件（如果需要） ===
    if args.create_sample_conditions:
        condition_file = create_sample_condition_file("sample_conditions.json")
        if is_main:
            print(f"[INFO] 已创建示例条件文件: {condition_file}")
    else:
        condition_file = args.condition_file

    # === Data ===
    # 创建条件编码器
    condition_config = cfg.condition
    encoder_type = condition_config.get("encoder_type", "simple")
    output_dim = condition_config.get("output_dim", 256)
    
    # 使用编码器配置文件
    encoder_config_file = "gutclip/configs/condition_encoders.yaml"
    
    condition_encoder = create_condition_encoder(
        encoder_type=encoder_type,
        config_file=encoder_config_file,
        output_dim=output_dim
    )
    
    if is_main:
        print(f"[INFO] 使用条件编码器: {encoder_type}")
        print(f"[INFO] 编码器配置文件: {encoder_config_file}")
        print(f"[INFO] 编码器维度: {output_dim}")
    
    # 创建条件数据模块
    data_module = ConditionalTreeDiffusionDataModule(
        cfg, 
        condition_file=condition_file,
        condition_encoder=condition_encoder
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    if is_main:
        print(f"[INFO] 训练集大小: {len(train_loader.dataset)}")
        print(f"[INFO] 验证集大小: {len(val_loader.dataset)}")

    # === Model ===
    # 加载预训练模型
    if is_main:
        print(f"[INFO] 加载预训练模型: {args.pretrained_ckpt}")
    
    pretrained_checkpoint = torch.load(args.pretrained_ckpt, map_location='cpu')
    
    # 构建条件模型
    if cfg.get("use_dna", False):
        model = ConditionalSeparatedDiffusionModelWithDNA(
            input_dim=4,
            dna_dim=cfg.get("dna_dim", 768),
            hidden_dim=cfg.model.hidden_dim,
            out_dim=cfg.model.out_dim,
            num_layers=cfg.model.num_layers,
            dropout_rate=cfg.model.dropout_rate,
            condition_dim=output_dim,
            pretrained_dna_encoder=True,
            dna_output_dim=cfg.get("dna_output_dim", None),
            model_cfg=OmegaConf.load(cfg.get("model_config", "gutclip/configs/model/separated_diffusion.yaml"))
                     if Path(cfg.get("model_config", "")).exists() else {}
        )
        if is_main:
            print("[INFO] 使用带DNA条件的条件分离建模模型")
    else:
        model = ConditionalSeparatedDiffusionModel(
            input_dim=4,
            hidden_dim=cfg.model.hidden_dim,
            out_dim=cfg.model.out_dim,
            num_layers=cfg.model.num_layers,
            dropout_rate=cfg.model.dropout_rate,
            condition_dim=output_dim,
            model_cfg=OmegaConf.load(cfg.get("model_config", "gutclip/configs/model/separated_diffusion.yaml"))
                     if Path(cfg.get("model_config", "")).exists() else {}
        )
        if is_main:
            print("[INFO] 使用条件分离建模模型")

    # 加载预训练权重（只加载基础部分，不加载条件相关部分）
    if is_main:
        print("[INFO] 加载预训练权重...")
    
    # 这里需要根据你的预训练模型结构来调整加载逻辑
    # 暂时跳过，等模型结构确定后再完善
    if is_main:
        print("[WARNING] 预训练权重加载逻辑需要根据具体模型结构调整")

    # 设置外部条件编码器
    model.set_external_condition_encoder(condition_encoder)
    
    # 模型移动到设备
    model = model.to(device)

    # DDP包装
    if is_dist():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if is_main:
            print("[INFO] 模型已包装为DDP")

    # === Optimizer & Scheduler ===
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay
    )

    # === Beta Schedule ===
    if cfg.beta_schedule == "linear":
        betas = linear_beta_schedule(cfg.T)
    elif cfg.beta_schedule == "cosine":
        betas = cosine_beta_schedule(cfg.T)
    else:
        raise ValueError(f"Unknown scheduler type: {cfg.beta_schedule}")

    # === Trainer ===
    trainer = ConditionalTreeDiffusionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        condition_encoder=condition_encoder,
        unconditional_prob=condition_config.get("unconditional_prob", 0.1),
        epochs=cfg.train.epochs,
        betas=betas,
        cfg=cfg,
        device=device
    )

    if is_main:
        print("[INFO] 开始条件扩散训练...")

    # === Train ===
    trainer.fit()

    if is_main:
        print("[INFO] 条件扩散训练完成!")

if __name__ == "__main__":
    main() 