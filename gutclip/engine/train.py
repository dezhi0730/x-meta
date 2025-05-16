# gutclip/engine/train.py
# -----------------------------------------------------------
# 训练与评估循环（含 wandb 日志 / 自动混合精度 / DDP all-reduce）
# -----------------------------------------------------------
import time
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler
import wandb
from typing import Union

from gutclip.loss import CLIPLoss


# ---------- 小工具 ---------------------------------------------------------
class AverageMeter:
    """仅做 epoch 累积均值（rank0 打印用）"""
    def __init__(self):
        self.sum = 0.0
        self.cnt = 0
    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.cnt += n
    @property
    def avg(self):
        return self.sum / max(self.cnt, 1)


def _reduce_tensor(t: torch.Tensor):
    """多卡均值；单卡原样返回"""
    if dist.is_initialized():
        dist.all_reduce(t)
        t /= dist.get_world_size()
    return t


# ---------- 训练一步 -------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, epoch: int,
                    device, cfg, scaler: Union[GradScaler, None]):
    model.train()
    loss_meter = AverageMeter()
    clip_loss = CLIPLoss(local_loss=cfg.local_loss).to(device)

    global_step_base = epoch * len(dataloader)  # wandb x 轴
    for it, batch in enumerate(dataloader):
        batch = batch.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            out  = model(batch)        # 需返回 tree_emb / dna_emb / logit_scale
            loss = clip_loss(out)

        optimizer.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # DDP 梯度同步
            if dist.is_initialized():
                for param in model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                        param.grad.data /= dist.get_world_size()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # DDP 梯度同步
            if dist.is_initialized():
                for param in model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                        param.grad.data /= dist.get_world_size()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

        # ---- 日志 ---------------------------------------------------------
        loss_detach = _reduce_tensor(loss.detach()).item()
        loss_meter.update(loss_detach, n=batch.dna.size(0))

        if (not dist.is_initialized()) or dist.get_rank() == 0:
            if wandb.run is not None:  # 只在 wandb 初始化后记录
                wandb.log(
                    {
                        "train/loss_step": loss_detach,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/epoch": epoch,
                    },
                    step=global_step_base + it,
                    commit=False,
                )

        # 控制台进度条（可选：tqdm）
        if it % cfg.log_interval == 0 and ((not dist.is_initialized()) or dist.get_rank() == 0):
            print(f"[Train] Epoch {epoch:03d} | Iter {it:04d}/{len(dataloader)} "
                  f"| loss {loss_detach:.4f}", flush=True)

    # ---- epoch 级日志 ------------------------------------------------------
    if (not dist.is_initialized()) or dist.get_rank() == 0:
        if wandb.run is not None:  # 只在 wandb 初始化后记录
            wandb.log({"train/loss_epoch": loss_meter.avg},
                      step=(epoch + 1) * len(dataloader))
        print(f"[Train] Epoch {epoch:03d}  avg_loss={loss_meter.avg:.4f}", flush=True)

    return loss_meter.avg


# ---------- 评估 -----------------------------------------------------------
@torch.no_grad()
def evaluate(model, dataloader, device, cfg):
    model.eval()
    loss_meter = AverageMeter()
    clip_loss = CLIPLoss(local_loss=cfg.local_loss).to(device)

    for batch in dataloader:
        batch = batch.to(device, non_blocking=True)
        out   = model(batch)
        loss  = clip_loss(out)

        loss_detach = _reduce_tensor(loss.detach()).item()
        loss_meter.update(loss_detach, n=batch.dna.size(0))

    if (not dist.is_initialized()) or dist.get_rank() == 0:
        if wandb.run is not None:  # 只在 wandb 初始化后记录
            wandb.log({"val/loss": loss_meter.avg})
        print(f"[Eval ] avg_loss={loss_meter.avg:.4f}", flush=True)

    return loss_meter.avg