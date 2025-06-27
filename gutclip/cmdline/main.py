# gutclip/cmdline/main.py
# -----------------------------------------------------------
# 项目入口：读取 cfg → 初始化 DDP / wandb / tensorboard → 训练循环
# -----------------------------------------------------------
import os, sys, argparse, yaml
import torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR
from omegaconf import OmegaConf
from pathlib import Path
import wandb

from torch.utils.tensorboard import SummaryWriter

from gutclip.utils.seed  import set_seed
from gutclip.data        import GutDataModule
from gutclip.models       import GutCLIPModel
from gutclip.engine import train_one_epoch, evaluate     # ★ 新版

# -----------------------------------------------------------
# 1. CLI & cfg
# -----------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="GutCLIP training")
    p.add_argument("--cfg", type=str, default="gutclip/configs/default.yaml")
    p.add_argument("--ddp", action="store_true", help="use DistributedDataParallel")
    p.add_argument("opts", nargs=argparse.REMAINDER, help="override yaml (e.g. epochs=10 lr=1e-4)")
    return p.parse_args()

def load_cfg(cfg_path, cli_opts):
    cfg = OmegaConf.load(cfg_path)
    if cli_opts:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(cli_opts))
    return cfg

# -----------------------------------------------------------
# 2. DDP helper
# -----------------------------------------------------------
def setup_ddp():
    if "RANK" in os.environ:          # torchrun 启动
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        print(f"[DDP] rank={rank}/{dist.get_world_size()} ready")
    else:                             # 单机
        dist.init_process_group("gloo", rank=0, world_size=1)
        rank = 0
    return rank

# -----------------------------------------------------------
# 3. Checkpoint helper
# -----------------------------------------------------------
def save_checkpoint(model, optimizer, scheduler,
                    epoch, metrics, cfg, tag):
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
    ckpt = {
        "epoch": epoch + 1,
        "model": (model.module if isinstance(model, DDP) else model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "cfg":  OmegaConf.to_container(cfg, resolve=True),
        "metrics": metrics
    }
    path = ckpt_dir / f"{cfg.name}_{tag}.pt"
    torch.save(ckpt, path)
    print(f"[Checkpoint] Saved → {path}")

# -----------------------------------------------------------
# 4. Main
# -----------------------------------------------------------
def main():
    args = parse_args()
    cfg  = load_cfg(args.cfg, args.opts)

    # -------- DDP / device / seed --------
    use_ddp = args.ddp or ("RANK" in os.environ)
    rank = setup_ddp() if use_ddp else 0
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed + rank)

    # -------- W&B (仅 rank0) --------------
    if rank == 0 and cfg.wandb.mode != "disabled":
        wandb.init(project=cfg.wandb.project,
                   name=cfg.name,
                   mode=cfg.wandb.mode,
                   config=OmegaConf.to_container(cfg, resolve=True))

    # -------- TensorBoard (仅 rank0) -------
    tb_writer = SummaryWriter(log_dir=cfg.get("log_dir", "runs/"+cfg.name)) if rank == 0 else None

    # -------- DataModule ------------------
    dm = GutDataModule(cfg)
    train_loader = dm.train_dataloader()
    val_loader   = dm.val_dataloader()

    # -------- Model -----------------------
    model = GutCLIPModel(tree_dim=cfg.tree_dim,
                         dna_dim=cfg.dna_dim,
                         output_dict=True).to(device)
    if use_ddp:
        model = DDP(model, device_ids=[rank])

    # -------- Optim / Scheduler ----------
    base_lr   = cfg.lr
    logit_lr  = base_lr * 0.1
    other = [p for n,p in model.named_parameters() if n!="logit_scale"]
    optimizer = torch.optim.AdamW(
        [{"params": other,               "lr": base_lr},
         {"params": [model.logit_scale], "lr": logit_lr}],
        weight_decay=cfg.wd
    )

    steps_total  = len(train_loader) * cfg.epochs
    steps_warmup = len(train_loader) * cfg.warmup_epochs

    scheduler = LambdaLR(
        optimizer,
        lambda step: step/steps_warmup if step<steps_warmup
        else 0.5*(1+torch.cos(torch.pi*(step-steps_warmup)/(steps_total-steps_warmup)))
    )

    scaler = GradScaler() if cfg.precision=="amp" else None

    # -------- Train loop ------------------
    best_top1, patience_cnt = 0.0, 0
    global_step = 0
    for epoch in range(cfg.epochs):
        if use_ddp: train_loader.sampler.set_epoch(epoch)

        train_one_epoch(model, train_loader, optimizer,
                        epoch, device, cfg, scaler, tb_writer)

        global_step += len(train_loader)

        val_loss, metrics = evaluate(model, val_loader, device, cfg,
                                     tb_writer if rank==0 else None,
                                     global_step)

        top1 = metrics["top1"] if metrics is not None else 0.0
        if use_ddp:
            top1_tensor = torch.tensor([top1], device=device)
            dist.broadcast(top1_tensor, src=0); top1 = top1_tensor.item()

        # lr schedule
        scheduler.step()

        # checkpoint
        if rank==0:
            save_checkpoint(model, optimizer, scheduler, epoch, metrics, cfg, tag="latest")
            if top1 > best_top1 + cfg.min_delta:
                best_top1, patience_cnt = top1, 0
                save_checkpoint(model, optimizer, scheduler, epoch, metrics, cfg, tag="best")
            else:
                patience_cnt += 1

        if patience_cnt >= cfg.patience:
            if rank==0: print(f"[EarlyStop] no improvement for {cfg.patience} epochs, stop.")
            break

        # wandb
        if rank==0 and wandb.run is not None and metrics is not None:
            wandb.log({"epoch": epoch,
                       "train/lr": optimizer.param_groups[0]["lr"],
                       "val/top1": top1,
                       "val/recall@5": metrics["recall@5"],
                       "val/mrr": metrics["mrr"]})

    if rank==0:
        if tb_writer: tb_writer.close()
        if cfg.wandb.mode != "disabled": wandb.finish()
    if use_ddp: dist.destroy_process_group()

if __name__ == "__main__":
    sys.exit(main())