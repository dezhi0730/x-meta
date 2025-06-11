# gutclip/cmdline/main.py
# -----------------------------------------------------------
# 项目入口：读取 cfg → 初始化 DDP / wandb → 训练循环
# -----------------------------------------------------------
import os, sys, argparse, yaml
from omegaconf import OmegaConf
import torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from pathlib import Path
import torch.nn as nn

from gutclip.utils.seed import set_seed
from gutclip.data import GutDataModule
from gutclip.models import GutCLIPModel
from gutclip.engine import train_one_epoch, evaluate


def parse_args():
    p = argparse.ArgumentParser(description="GutCLIP training")
    p.add_argument("--cfg", type=str, default="gutclip/configs/default.yaml",
                   help="yaml config path")
    p.add_argument("--ddp", action="store_true",
                   help="use DistributedDataParallel")
    p.add_argument("opts", nargs=argparse.REMAINDER,
                   help="override yaml (e.g. epochs=10 lr=1e-4)")
    return p.parse_args()


def load_cfg(cfg_path, cli_opts):
    cfg = OmegaConf.load(cfg_path)
    if cli_opts:
        cli_cfg = OmegaConf.from_dotlist(cli_opts)
        cfg = OmegaConf.merge(cfg, cli_cfg)
    return cfg


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
    else:
        dist.init_process_group("gloo", rank=0, world_size=1)
    return dist.get_rank()


def save_checkpoint(model, optimizer, scheduler,
                    epoch, val_metrics, cfg,
                    tag: str):
    """
    tag = 'latest' / 'best' / f'e{epoch:03d}_t{top1:.3f}'
    """
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
    ckpt = {
        "epoch": epoch + 1,
        "model": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "metrics": val_metrics
    }
    path = ckpt_dir / f"{cfg.name}_{tag}.pt"
    torch.save(ckpt, path)
    print(f"[Checkpoint] Saved → {path}")


def main():
    args = parse_args()
    cfg  = load_cfg(args.cfg, args.opts)

    # 根据参数决定是否使用 DDP
    if args.ddp:
        rank = setup_ddp()
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        set_seed(cfg.seed + rank)  # 保证多卡 determinism
    else:
        rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        set_seed(cfg.seed)

    if (not args.ddp or rank == 0) and cfg.wandb.mode != "disabled":
        wandb.init(project=cfg.wandb.project,
                   name=cfg.name,
                   mode=cfg.wandb.mode,
                   config=OmegaConf.to_container(cfg, resolve=True))

    # -------- 数据加载器 -----------------------------
    dm = GutDataModule(cfg)
    train_loader = dm.train_dataloader()
    val_loader   = dm.val_dataloader()

    # -------- 模型 & 优化器 --------------------------
    model = GutCLIPModel(embed_dim=cfg.embed_dim,
                         tree_dim=cfg.tree_dim,
                         dna_dim=cfg.dna_dim,
                         output_dict=True).to(device)
    
    # 根据参数决定是否使用 DDP
    if args.ddp:
        model = DDP(model, device_ids=[rank])

    base_lr   = cfg.lr              # 例如 5e-4
    logit_lr  = base_lr * 0.1       # 5e-5

    other_params  = [p for n, p in model.named_parameters()
                    if n != "logit_scale"]

    optimizer = torch.optim.AdamW(
        [
            {"params": other_params,           "lr": base_lr},
            {"params": [model.logit_scale],    "lr": logit_lr}
        ],
        weight_decay=cfg.wd
    )

    # === ② 学习率调度器照常对两个 group 都生效 ===
    total_steps  = len(train_loader) * cfg.epochs
    warmup_steps = len(train_loader) * cfg.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 0.5 * (1 + torch.cos(torch.pi *
                    (step - warmup_steps) / (total_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(init_scale=2.0) if cfg.precision == "amp" else None

    # -------- 训练循环 ------------------------------
    best_top1 = 0.0
    patience_counter = 0
    patience = cfg.patience

    for epoch in range(cfg.epochs):
        if args.ddp:
            train_loader.sampler.set_epoch(epoch)

        # 1) 训练
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                    epoch, device, cfg, scaler)

        # 2) 验证
        val_loss, val_metrics = evaluate(model, val_loader, device, cfg)

        if args.ddp:
            top1_tensor = torch.tensor(
                [val_metrics["top1"]] if rank == 0 else [0.0],
                device=device
            )
            dist.broadcast(top1_tensor, src=0)
            top1 = top1_tensor.item()
        else:
            top1 = val_metrics["top1"]

        # 3) 学习率
        scheduler.step()

        # 4) ─── 保存 latest，覆盖写 ───
        if not args.ddp or rank == 0:
            save_checkpoint(model, optimizer, scheduler, epoch,
                            val_metrics, cfg, tag="latest")

        # 5) ─── 保存 best (Top-1 提升) ───
        if top1 > best_top1 + cfg.min_delta:
            best_top1 = top1
            patience_counter = 0
            if not args.ddp or rank == 0:
                save_checkpoint(model, optimizer, scheduler, epoch,
                                val_metrics, cfg, tag="best")
        else:
            patience_counter += 1

        # 6) 早停
        if patience_counter >= patience:
            if not args.ddp or rank == 0:
                print(f"[EarlyStop] Top-1 连续 {patience} 轮无提升，停止训练。")
            break

        # 7) wandb 记录（仅 rank0）
        if (not args.ddp or rank == 0) and wandb.run is not None:
            wandb.log({
                "epoch":        epoch,
                "train/loss":   train_loss,
                "val/loss":     val_loss,
                "val/top1":     top1,
                "val/recall@5": val_metrics["recall@5"],
                "val/mrr":      val_metrics["mrr"],
            })

    if (not args.ddp or rank == 0) and cfg.wandb.mode != "disabled":
        wandb.finish()
    
    if args.ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())