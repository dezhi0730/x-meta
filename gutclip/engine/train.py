# -*- coding: utf-8 -*-
import torch, time, wandb
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from typing import Union

from gutclip.loss import  SparseCLIPLoss, add_loss_params_to_optimizer  

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


def train_one_epoch(model, dataloader, optimizer,
                    epoch: int, device,
                    cfg, scaler: Union[GradScaler, None],
                    tb: SummaryWriter,
                    criterion: SparseCLIPLoss):

    model.train()
    loss_meter = AverageMeter()
    step_base  = epoch * len(dataloader)

    for it, batch in enumerate(dataloader):
        # ------------------ 数据检查 ------------------
        if not hasattr(batch, "dna") or batch.dna.size(0) == 0:
            print(f"[WARN] Step {it}: empty batch, skip");  continue

        # 可选：更多 NaN/Inf 检查（略）
        batch = batch.to(device, non_blocking=True)

        # ------------------ 前向 ------------------
        with torch.amp.autocast(device_type='cuda', enabled=scaler is not None):
            out  = model(batch)   # 必须返回 tree_emb / dna_emb / logit_scale / zero_ratio
            loss = criterion(out)

        # 跳过无效 loss
        if not torch.isfinite(loss):
            print(f"[WARN] Step {it}: non‑finite loss, skip");  continue

        # ------------------ 反向 ------------------
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

        # ------------------ 统计 / 日志 ------------------
        loss_meter.update(_reduce_tensor(loss.detach()).item(), n=batch.dna.size(0))
        global_step = step_base + it

        if (not dist.is_initialized() or dist.get_rank()==0) and it % cfg.log_interval == 0:
            zb = float(criterion.zero_weight)
            tb.add_scalar("train/loss_step", loss.item(), global_step)
            tb.add_scalar("train/learned_zero_weight", zb, global_step)
            print(f"[Train] Ep{epoch:03d} It{it:04d}/{len(dataloader)} "
                  f"loss={loss.item():.4f} | zero_w={zb:.3f}", flush=True)

    if not dist.is_initialized() or dist.get_rank()==0:
        tb.add_scalar("train/loss_epoch", loss_meter.avg,
                      (epoch+1)*len(dataloader))
        print(f"[Train] Ep{epoch:03d} AVG_LOSS={loss_meter.avg:.4f}", flush=True)

    return loss_meter.avg

@torch.no_grad()
def _retrieval_metrics(sim, ks=(1,5,10)):
    ranks   = (-sim).argsort(1)
    tgt     = torch.arange(sim.size(0), device=sim.device)[:,None]
    corr    = ranks.eq(tgt)
    out = {"top1": corr[:,0].float().mean()}
    for k in ks[1:]:
        out[f"recall@{k}"] = corr[:,:k].any(1).float().mean()
    pos_r   = torch.nonzero(corr)[:,1] + 1
    out["mrr"]         = (1./pos_r.float()).mean()
    out["median_rank"] = pos_r.median()
    return {k: v.item() for k,v in out.items()}


# @torch.no_grad()
# def evaluate(model, dataloader, device, cfg,
#              tb: SummaryWriter, global_step: int,
#              criterion: SparseCLIPLoss):

#     model.eval()
#     loss_meter = AverageMeter()
#     all_tree, all_dna = [], []

#     for batch in dataloader:
#         batch = batch.to(device, non_blocking=True)
#         out   = model(batch)
#         loss  = criterion(out)
#         loss_meter.update(loss.item(), n=batch.dna.size(0))
#         all_tree.append(out["tree_emb"].cpu())
#         all_dna .append(out["dna_emb"].cpu())

#     metrics = None
#     if not dist.is_initialized() or dist.get_rank()==0:
#         tree = torch.cat(all_tree);  dna = torch.cat(all_dna)
#         sim  = tree @ dna.T
#         metrics = _retrieval_metrics(sim.T)
#         tb.add_scalar("val/loss", loss_meter.avg, global_step)
#         tb.add_scalars("val/metrics", metrics, global_step)
#         print(f"[Eval] loss={loss_meter.avg:.4f} | "
#               f"top1={metrics['top1']:.3f} "
#               f"r5={metrics['recall@5']:.3f} "
#               f"mrr={metrics['mrr']:.3f}", flush=True)
#     return loss_meter.avg, metrics
@torch.no_grad()
def evaluate(model, dataloader, device, cfg,
             tb: SummaryWriter, global_step: int,
             criterion: SparseCLIPLoss):

    model.eval()
    loss_meter, clip_meter, var_meter = AverageMeter(), AverageMeter(), AverageMeter()
    all_tree, all_dna = [], []

    for batch in dataloader:
        batch = batch.to(device, non_blocking=True)
        out   = model(batch)

        # ---------- 1. loss 拆分 ----------
        total_loss = criterion(out)                     # 原来返回的是 clip+var
        clip_loss  = criterion.last_clip_loss           # 需在 SparseCLIPLoss.forward 设置
        var_loss   = criterion.last_var_loss            #   self.last_clip_loss = clip_loss
        loss_meter.update(total_loss.item(), n=batch.dna.size(0))
        clip_meter.update(clip_loss.item(), n=batch.dna.size(0))
        var_meter .update(var_loss.item(),  n=batch.dna.size(0))

        all_tree.append(out["tree_emb"].cpu())
        all_dna .append(out["dna_emb"].cpu())

    metrics = None
    if not dist.is_initialized() or dist.get_rank() == 0:
        tree = torch.cat(all_tree); dna  = torch.cat(all_dna)
        sim  = tree @ dna.T
        metrics = _retrieval_metrics(sim.T)

        # ---------- 2. 诊断量 ----------
        diag_mean  = sim.diag().mean().item()
        sim_mean   = sim.mean().item()
        zt_std     = tree.std().item()
        zd_std     = dna .std().item()
        logit_exp  = out['logit_scale'].exp().item()

        # ---------- 3. TensorBoard ----------
        tb.add_scalar("val/loss_total", loss_meter.avg, global_step)
        tb.add_scalar("val/loss_clip",  clip_meter.avg, global_step)
        tb.add_scalar("val/loss_var",   var_meter.avg,  global_step)
        tb.add_scalars("val/metrics",   metrics,        global_step)
        tb.add_scalars("val/debug", {
            "sim_diag_mean": diag_mean,
            "sim_mean":      sim_mean,
            "zt_std":        zt_std,
            "zd_std":        zd_std,
            "logit_scale":   logit_exp
        }, global_step)

        print(f"[Eval] loss={loss_meter.avg:.4f}  "
              f"(clip {clip_meter.avg:.3f} | var {var_meter.avg:.3f}) | "
              f"top1={metrics['top1']:.3f}  r5={metrics['recall@5']:.3f} "
              f"mrr={metrics['mrr']:.3f} || "
              f"diag={diag_mean:.3f}  mean={sim_mean:.3f}  "
              f"σ_t={zt_std:.3f} σ_d={zd_std:.3f}  logit={logit_exp:.2f}",
              flush=True)

    return loss_meter.avg, metrics