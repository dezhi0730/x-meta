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

from gutclip.utils.seed import set_seed
from gutclip.data import GutDataModule
from gutclip.models import GutCLIPModel
from gutclip.engine import train_one_epoch, evaluate


def parse_args():
    p = argparse.ArgumentParser(description="GutCLIP DDP training")
    p.add_argument("--cfg", type=str, default="gutclip/configs/default.yaml",
                   help="yaml config path")
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


def save_checkpoint(model, optimizer, scheduler, epoch, loss, cfg, is_best=False):
    """保存检查点"""
    if dist.get_rank() != 0:
        return

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    ckpt = {
        "epoch": epoch + 1,
        "model": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }

    # 保存最新检查点
    torch.save(ckpt, checkpoint_dir / f"{cfg.name}_latest.pt")
    
    # 如果是最佳模型，额外保存一份
    if is_best:
        torch.save(ckpt, checkpoint_dir / f"{cfg.name}_best.pt")
        print(f"[Checkpoint] Saved best model to checkpoints/{cfg.name}_best.pt")


def main():
    args = parse_args()
    cfg  = load_cfg(args.cfg, args.opts)

    rank = setup_ddp()                       # → world_size & rank
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed + rank)                # 保证多卡 determinism

    if rank == 0 and cfg.wandb.mode != "disabled":
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
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)
    scaler = torch.amp.GradScaler(init_scale=2.0) if cfg.precision == "amp" else None

    # -------- 训练循环 ------------------------------
    best_val_loss = float('inf')
    patience = getattr(cfg, 'patience', 10)  # 早停耐心值，默认10个epoch
    patience_counter = 0

    for epoch in range(cfg.epochs):
        if dist.is_initialized(): 
            train_loader.sampler.set_epoch(epoch)

        # 训练一个epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, epoch,
                                    device, cfg, scaler)
        
        # 验证
        val_loss = evaluate(model, val_loader, device, cfg)
        
        # 学习率调整
        scheduler.step()

        # 检查是否是最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, cfg, is_best=True)
        else:
            patience_counter += 1
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, cfg, is_best=False)
            
            # 早停检查
            if patience_counter >= patience:
                if rank == 0:
                    print(f"[Early Stopping] No improvement for {patience} epochs. Stopping training.")
                break

    if rank == 0 and cfg.wandb.mode != "disabled":
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())