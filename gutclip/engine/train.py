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
    clip_loss = CLIPLoss(local_loss=cfg.local_loss, gamma=1.0).to(device)

    # 温度控制 - 使用更大的学习率
    if epoch == 0:
        # 第一个epoch：降温
        model.logit_scale.data = torch.tensor([1.0], device=device)  # exp≈2.7
        print("[Info] Epoch 0: Temperature cooled down")
        
        # 为温度参数设置更大的学习率
        for param_group in optimizer.param_groups:
            # 检查参数组中是否包含 logit_scale 参数
            if any('logit_scale' in str(param) for param in param_group['params']):
                param_group['lr'] = 1e-3  # 增大温度的学习率
                print(f"[Info] Increased learning rate for temperature to {param_group['lr']}")

    global_step_base = epoch * len(dataloader)  # wandb x 轴
    for it, batch in enumerate(dataloader):
        batch = batch.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            out  = model(batch)        # 需返回 tree_emb / dna_emb / logit_scale
            
            # 添加调试信息
            if it % cfg.log_interval == 0 and ((not dist.is_initialized()) or dist.get_rank() == 0):
                tree_norm = out["tree_emb"].norm(dim=1).mean().item()
                dna_norm = out["dna_emb"].norm(dim=1).mean().item()
                logit_scale = out["logit_scale"].item()
                
                # 检查梯度
                grad_norm = 0.0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.norm().item()
                
                print(f"[Debug] tree_norm={tree_norm:.4f} dna_norm={dna_norm:.4f} "
                      f"logit_scale={logit_scale:.4f} grad_norm={grad_norm:.4f}")
                
                # 添加余弦相似度监控
                with torch.no_grad():
                    # 确保维度匹配
                    tree_emb = out["tree_emb"]  # (B, D)
                    dna_emb = out["dna_emb"]    # (B, D)
                    
                    # 计算相似度矩阵
                    cos = tree_emb @ dna_emb.T  # (B, B)
                    
                    # 计算对角线（正样本）和非对角线（负样本）的相似度
                    diag = cos.diag().mean().item()
                    
                    # 创建掩码矩阵
                    mask = ~torch.eye(cos.size(0), dtype=torch.bool, device=cos.device)
                    off = cos[mask].mean().item()
                    
                    print(f"[DebugCos] diag={diag:.3f}, off={off:.3f}")
                    
                    # 添加标准差检查
                    tree_std = tree_emb.std(dim=0).mean().item()
                    dna_std = dna_emb.std(dim=0).mean().item()
                    print(f"[DebugStd] tree_std={tree_std:.4f}, dna_std={dna_std:.4f}")
                
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
            # 梯度裁剪移到同步后
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
            # 梯度裁剪移到同步后
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


def _retrieval_metrics(sim, ks=(1,5,10)):
    """sim: (B,B) 本批相似度矩阵，行=tree，列=dna"""
    ranks   = (-sim).argsort(dim=1)
    target  = torch.arange(sim.size(0), device=sim.device).unsqueeze(1)
    correct = ranks.eq(target)
    metrics = {
        "top1": correct[:,0].float().mean()
    }
    for k in ks[1:]:
        metrics[f"recall@{k}"] = correct[:,:k].any(dim=1).float().mean()
    pos_rank = torch.nonzero(correct)[:,1] + 1
    metrics["mrr"]         = (1./pos_rank.float()).mean()
    metrics["median_rank"] = pos_rank.median()
    return {k: v.item() for k,v in metrics.items()}


# ---------- 评估 -----------------------------------------------------------
@torch.no_grad()
def evaluate(model, dataloader, device, cfg):
    model.eval()
    loss_meter = AverageMeter()
    clip_loss  = CLIPLoss(local_loss=cfg.local_loss).to(device)

    # 用于收集本 epoch 的所有 tree/dna 向量
    all_tree, all_dna = [], []

    for batch in dataloader:
        batch = batch.to(device, non_blocking=True)
        out   = model(batch)                      # dict

        # CE 损失
        loss  = clip_loss(out)
        loss_meter.update(loss.detach().item(), n=batch.dna.size(0))

        # 收集对齐向量
        all_tree.append(out["tree_emb"].cpu())    # (B,D)
        all_dna .append(out["dna_emb"].cpu())

    # ------- 只在 rank==0 计算检索指标 -------
    if (not dist.is_initialized()) or dist.get_rank() == 0:
        tree_emb = torch.cat(all_tree, 0)         # (N,D)
        dna_emb  = torch.cat(all_dna , 0)
        sim = tree_emb @ dna_emb.T                # (N,N)

        metrics = _retrieval_metrics(sim.T)         # dict

        if wandb.run is not None:
            wandb.log({"val/loss": loss_meter.avg,
                       "val/top1": metrics["top1"],
                       "val/recall@5": metrics["recall@5"],
                       "val/recall@10": metrics["recall@10"],
                       "val/mrr": metrics["mrr"],
                       "val/median_rank": metrics["median_rank"]})
        print(f"[Eval] loss={loss_meter.avg:.4f} | "
              f"top1={metrics['top1']:.3f} r5={metrics['recall@5']:.3f} "
              f"mrr={metrics['mrr']:.3f}", flush=True)
    else:
        metrics = None   # 其它 rank 返回 None

    return loss_meter.avg, metrics