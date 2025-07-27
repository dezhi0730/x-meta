# gutclip/cmdline/main.py
# ============================================================
import os, sys, argparse
import torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR
from omegaconf import OmegaConf
from pathlib import Path
import wandb
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from typing import Union
import math

from gutclip.utils.seed  import set_seed
from gutclip.data        import GutDataModule
from gutclip.models      import GutCLIPModel

# ---- 可学习损失 ----
from gutclip.loss import SparseCLIPLoss, add_loss_params_to_optimizer

from gutclip.engine import train_one_epoch, evaluate
# -----------------------------------------------------------
# CLI / cfg
# -----------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, default="gutclip/configs/default.yaml")
    p.add_argument("--ddp", action="store_true")
    p.add_argument("opts", nargs=argparse.REMAINDER)
    return p.parse_args()

def load_cfg(path, opts):
    cfg = OmegaConf.load(path)
    if opts: cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(opts))
    return cfg
# -----------------------------------------------------------
def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank(); torch.cuda.set_device(rank)
    else:
        dist.init_process_group("gloo", rank=0, world_size=1); rank=0
    return rank
# -----------------------------------------------------------
def save_ckpt(model, opt, sch, epoch, metrics, cfg, tag,
              return_path: bool = False) -> Union[Path, None]:
    if dist.is_initialized() and dist.get_rank() != 0:
        return None

    p = Path("checkpoints/gutclip"); p.mkdir(exist_ok=True)

    if tag == "latest":
        fname = f"{cfg.name}_latest.pt"
    else:
        date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        metric_suffix = (
            "_" + "_".join(f"{k}{metrics[k]:.4f}"
                           for k in ("top1","recall@5","mrr")
                           if metrics and k in metrics and metrics[k] is not None)
        ) if metrics else ""
        fname = f"{cfg.name}_{tag}_{date_str}{metric_suffix}.pt"

    fpath = p / fname
    torch.save({
        "epoch": epoch + 1,
        "model": (model.module if isinstance(model, DDP) else model).state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sch.state_dict(),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "metrics": metrics
    }, fpath)

    return fpath if return_path else None
# -----------------------------------------------------------
def main():
    args = parse_args(); cfg = load_cfg(args.cfg, args.opts)
    use_ddp = args.ddp or ("RANK" in os.environ)
    rank = setup_ddp() if use_ddp else 0
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed + rank)

    if rank==0 and cfg.wandb.mode!="disabled":
        wandb.init(project=cfg.wandb.project, name=cfg.name,
                   mode=cfg.wandb.mode,
                   config=OmegaConf.to_container(cfg, resolve=True))
    tb = SummaryWriter("runs/"+cfg.name) if rank==0 else None

    dm = GutDataModule(cfg)
    train_loader, val_loader = dm.train_dataloader(), dm.val_dataloader()

    model = GutCLIPModel(tree_dim=cfg.tree_dim,
                         dna_dim=cfg.dna_dim,
                         output_dict=True).to(device)
    if use_ddp: model = DDP(model, device_ids=[rank])

    criterion = SparseCLIPLoss(
        local_loss = cfg.local_loss,
        init_zero_w= cfg.sparse.zero_weight_init,
        var_gamma  = cfg.var_gamma,
        var_weight = cfg.var_weight,
        normalize  = True,
    ).to(device)

    optimizer = add_loss_params_to_optimizer(model, criterion,
                                             base_lr=cfg.lr,
                                             weight_decay=cfg.wd,
                                             logit_lr_factor=cfg.logit_lr_factor)

    steps_total  = len(train_loader)*cfg.epochs
    steps_warmup = len(train_loader)*cfg.warmup_epochs
    scheduler = LambdaLR(
        optimizer,
        lambda s: s/steps_warmup if s<steps_warmup else
                  0.5*(1+math.cos(math.pi*(s-steps_warmup)/(steps_total-steps_warmup)))
    )
    scaler = GradScaler() if cfg.precision=="amp" else None

    best_top1, patience = 0.0,0; global_step=0
    best_ckpt_path   = None 
    for ep in range(cfg.epochs):
        if use_ddp: train_loader.sampler.set_epoch(ep)
        train_one_epoch(model, train_loader, optimizer,
                        ep, device, cfg, scaler, tb, criterion)
        global_step += len(train_loader)

        val_loss, metrics = evaluate(model, val_loader, device, cfg,
                                     tb if rank==0 else None,
                                     global_step, criterion)
        top1 = metrics["top1"] if metrics else 0.0
        if use_ddp:
            t = torch.tensor([top1], device=device); dist.broadcast(t,0); top1=t.item()

        scheduler.step()
        if rank==0:
            save_ckpt(model, optimizer, scheduler, ep, metrics, cfg, "latest")
            if top1 > best_top1 + cfg.min_delta:
                best_top1, patience = top1,0
                if best_ckpt_path and best_ckpt_path.exists():
                    best_ckpt_path.unlink(missing_ok=True)
                best_ckpt_path = save_ckpt(
                    model, optimizer, scheduler, ep, metrics, cfg, "best",
                    return_path=True             
                )
        else:
            patience += 1

        if patience >= cfg.patience:
            if rank == 0:
                print(f"[EarlyStop] no improvement for {cfg.patience} epochs — stop.")
            break

        # ---------- W&B log (rank‑0) ----------
        if rank == 0 and wandb.run is not None and metrics:
            wandb.log({
                "epoch": ep,
                "train/lr": optimizer.param_groups[0]["lr"],
                "val/top1":       top1,
                "val/recall@5":   metrics["recall@5"],
                "val/mrr":        metrics["mrr"],
                "learned_zero_w": float(criterion.zero_weight)
            })

    # ------------ 收尾 ------------
    if rank == 0:
        if tb: tb.close()
        if cfg.wandb.mode != "disabled":
            wandb.finish()
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())