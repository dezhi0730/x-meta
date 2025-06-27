# gutclip/engine/train_tb.py
# -*- coding: utf-8 -*-
import torch, time, wandb
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from typing import Union

from gutclip.loss import MarginCLIPLoss, CLIPLoss   # 你的实现
# ---------------------------------------------------------
# 小工具
# ---------------------------------------------------------
class AverageMeter:
    def __init__(self):
        self.sum, self.cnt = 0.0, 0
    def update(self, val, n=1):
        self.sum += val * n; self.cnt += n
    @property
    def avg(self): return self.sum / max(self.cnt, 1)

def _reduce_tensor(t: torch.Tensor):
    if dist.is_initialized():
        dist.all_reduce(t); t /= dist.get_world_size()
    return t

# ---------------------------------------------------------
# 1. 单 epoch 训练
# ---------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer,
                    epoch: int, device,
                    cfg, scaler: Union[GradScaler, None],
                    tb: SummaryWriter):
    model.train()
    loss_meter = AverageMeter()
    loss_fn = MarginCLIPLoss(margin=0.1,
                             local_loss=cfg.local_loss,
                             gamma=0.5).to(device)

    step_base = epoch * len(dataloader)

    for it, batch in enumerate(dataloader):
        batch = batch.to(device, non_blocking=True)

        # ---------- 前向 ----------
        with torch.amp.autocast(device_type='cuda',
                                enabled=scaler is not None):
            out  = model(batch)              # dict: tree_emb / dna_emb / logit_scale
            loss = loss_fn(out)

        # ---------- 度量正负分数 ----------
        with torch.no_grad():
            sim = out["tree_emb"] @ out["dna_emb"].T      # (B,B)
            pos = sim.diag()
            neg = sim[~torch.eye(sim.size(0),
                                 dtype=torch.bool,
                                 device=sim.device)]
            delta = pos.mean() - neg.mean()
            global_step = step_base + it

            # 写 TensorBoard（每 step）
            if not dist.is_initialized() or dist.get_rank() == 0:
                tb.add_scalar("train/loss_step", loss.item(), global_step)
                tb.add_scalars("train/score_mean",
                               {"pos":   pos.mean().item(),
                                "neg":   neg.mean().item(),
                                "delta": delta.item()},
                               global_step)
                # 每 500 step 画直方图
                if it % 500 == 0:
                    tb.add_histogram("train/pos_scores", pos.cpu(), global_step)
                    tb.add_histogram("train/neg_scores", neg.cpu(), global_step)

        # ---------- 反向 ----------
        optimizer.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

        # ---------- 统计 ----------
        loss_meter.update(_reduce_tensor(loss.detach()).item(),
                          n=batch.dna.size(0))

        if (not dist.is_initialized() or dist.get_rank() == 0) and \
           it % cfg.log_interval == 0:
            print(f"[Train] Ep{epoch:03d} It{it:04d}/{len(dataloader)} "
                  f"loss={loss.item():.4f} Δ={delta:+.4f}", flush=True)

    # epoch 末写一次平均 loss
    if not dist.is_initialized() or dist.get_rank() == 0:
        tb.add_scalar("train/loss_epoch",
                      loss_meter.avg, (epoch + 1) * len(dataloader))
        print(f"[Train] Ep{epoch:03d} AVG_LOSS={loss_meter.avg:.4f}", flush=True)

    return loss_meter.avg

# ---------------------------------------------------------
# 2. 评估
# ---------------------------------------------------------
def _retrieval_metrics(sim, ks=(1,5,10)):
    ranks   = (-sim).argsort(dim=1)
    target  = torch.arange(sim.size(0), device=sim.device).unsqueeze(1)
    correct = ranks.eq(target)
    out = {"top1": correct[:,0].float().mean()}
    for k in ks[1:]:
        out[f"recall@{k}"] = correct[:,:k].any(1).float().mean()
    pos_rank = torch.nonzero(correct)[:,1] + 1
    out["mrr"]         = (1./pos_rank.float()).mean()
    out["median_rank"] = pos_rank.median()
    return {k: v.item() for k,v in out.items()}

@torch.no_grad()
def evaluate(model, dataloader, device, cfg, tb: SummaryWriter, global_step:int):
    model.eval()
    loss_meter = AverageMeter()
    clip_loss  = CLIPLoss(local_loss=cfg.local_loss).to(device)
    all_tree, all_dna = [], []

    for batch in dataloader:
        batch = batch.to(device, non_blocking=True)
        out   = model(batch)
        loss  = clip_loss(out)
        loss_meter.update(loss.item(), n=batch.dna.size(0))
        all_tree.append(out["tree_emb"].cpu())
        all_dna .append(out["dna_emb"].cpu())

    # 仅主进程计算指标
    metrics = None
    if not dist.is_initialized() or dist.get_rank() == 0:
        tree_emb = torch.cat(all_tree, 0)
        dna_emb  = torch.cat(all_dna , 0)
        sim = tree_emb @ dna_emb.T

        metrics = _retrieval_metrics(sim.T)
        tb.add_scalar("val/loss", loss_meter.avg, global_step)
        tb.add_scalars("val/metrics", metrics, global_step)

        if wandb.run is not None:
            wandb.log({"val/loss": loss_meter.avg, **metrics})

        print(f"[Eval] loss={loss_meter.avg:.4f} | "
              f"top1={metrics['top1']:.3f} "
              f"r5={metrics['recall@5']:.3f} "
              f"mrr={metrics['mrr']:.3f}", flush=True)

    return loss_meter.avg, metrics