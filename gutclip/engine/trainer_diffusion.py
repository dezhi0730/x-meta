# gutclip/engine/trainer_diffusion.py
# -----------------------------------------------------------
import os
import math
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from omegaconf import OmegaConf

from gutclip.diffusion.schedulers import get_scheduler
from gutclip.diffusion.utils import add_noise
from gutclip.models.diffusion.unet1d_film import FiLMUNet1D
from gutclip.loss.diffusion_loss import DiffusionLoss


# ─────────────────── Retrieval 轻量封装（支持 LOO） ────────────────────
class RetrievalIndex:
    def __init__(self, index_file: str, y_file: str, ids_file: Optional[str],
                 gpu: int = 0, k: int = 5):
        import faiss, numpy as np
        from gutclip.data.retrieval_index import _to_gpu

        self.k = int(k)
        self.idx = faiss.read_index(index_file)
        if gpu >= 0:
            self.idx = _to_gpu(self.idx, gpu)

        self.y = np.load(y_file)  # (B_all, N)
        self.ids = None
        if ids_file is not None and os.path.exists(ids_file):
            self.ids = np.load(ids_file)

        self.norm = bool(np.load(y_file.replace(".y.npy", ".norm.npy")).item())
        self.device = torch.device(f"cuda:{gpu}" if gpu >= 0 else "cpu")

        # 预计算全局均值，作为极端回退
        self.y_mean = torch.from_numpy(self.y.mean(axis=0)).float().to(self.device)

        self._printed_loo_check = False
        self._warned_all_mask_rows = 0

    @torch.no_grad()
    def query(self, z: torch.Tensor, sids: Optional[List[str]] = None) -> torch.Tensor:
        import numpy as np

        B = z.size(0)
        z_np = z.detach().cpu().numpy()
        if self.norm:
            from gutclip.data.retrieval_index import _l2_normalize
            z_np = _l2_normalize(z_np)

        # 取 k+1 便于丢弃自身
        D, I = self.idx.search(z_np.astype("float32"), self.k + 1)  # (B, k+1)

        D_t = torch.from_numpy(D).float()  # (B, k+1)
        I_t = torch.from_numpy(I)          # (B, k+1)

        # 无效索引置 -inf 分数
        invalid_mask = (I_t < 0)
        if invalid_mask.any():
            D_t[invalid_mask] = -1e9

        # —— LOO —— #
        if sids is not None and self.ids is not None:
            ids_np = self.ids
            hit_before = 0
            mask_self = torch.zeros_like(D_t, dtype=torch.bool)
            for b, sid in enumerate(sids):
                if sid is None:
                    continue
                neigh_ids = ids_np[I[b]]              # (k+1,)
                hit_before += (neigh_ids == sid).sum()
                mask_self[b] = torch.from_numpy(neigh_ids == sid)

            # 屏蔽自身
            D_t[mask_self] = -1e9

            # 重新选 top-k
            D_top, top_idx = torch.topk(D_t, k=self.k, dim=1)     # (B,k)
            I_top = torch.gather(I_t, 1, top_idx)                 # (B,k)

            # 统计 after
            hit_after = 0
            for b, sid in enumerate(sids):
                if sid is None:
                    continue
                neigh_ids = ids_np[I_top[b].cpu().numpy()]
                hit_after += (neigh_ids == sid).sum()

            if not self._printed_loo_check:
                total_slots = len(sids) * (self.k + 1)
                print(f"[CHECK] LOO: raw self-hit: {int(hit_before)}/{total_slots}, after mask: {int(hit_after)}")
                self._printed_loo_check = True
        else:
            # 无 ids：丢弃 top-1
            D_top = D_t[:, 1 : self.k + 1]
            I_top = I_t[:, 1 : self.k + 1]

        # 处理 top-k 中的无效索引（-1）
        valid_top = (I_top >= 0)                                # (B,k)
        any_valid = valid_top.any(dim=1)                        # (B,)

        # 为 softmax 做权重；先把无效位置的分数设得非常小
        D_top_masked = D_top.clone()
        D_top_masked[~valid_top] = -1e9
        W = torch.softmax(D_top_masked, dim=1)                  # (B,k)

        # 将 torch 索引转成 numpy
        I_top_np = I_top.cpu().numpy()                          # (B,k)

        # 取邻居 y
        y_neighbors = torch.from_numpy(self.y[I_top_np]).float()  # (B,k,N)

        # 对无效位置将权重置 0
        W = W * valid_top.float()

        # 有效权重和为 0 的行（全无效）：回退为全局均值
        W_sum = W.sum(dim=1, keepdim=True)                      # (B,1)
        need_fallback = (W_sum.squeeze(1) == 0)

        # 避免除零：对有有效权重的行做归一化
        W_norm = torch.where(W_sum > 0, W / (W_sum + 1e-8), W)  # (B,k)

        y_prior = (y_neighbors * W_norm.unsqueeze(-1)).sum(1).to(self.device)  # (B,N)

        if need_fallback.any():
            cnt = int(need_fallback.sum().item())
            y_prior[need_fallback] = self.y_mean
            # 只打印一次累计告警
            self._warned_all_mask_rows += cnt
            if self._warned_all_mask_rows > 0 and self._warned_all_mask_rows <= 1:
                print(f"[WARN] {cnt} rows had no valid neighbors; fell back to global mean y.")
        return y_prior


# ─────────────────── TrainerDiffusion 类 ───────────────────
class TrainerDiffusion:
    def __init__(self, cfg, dataloader, val_loader, retrieval, device=None):
        self.device = device if device is not None else torch.device("cuda:0")
        self.cfg = cfg
        self.train_loader = dataloader
        self.val_loader = val_loader
        self.ret = retrieval

        # 显式维度
        self.z_dna_dim = int(cfg.model.z_dna_dim)
        self.y_dim = int(cfg.model.y_dim)
        self.proj_dim = int(cfg.model.proj_dim)
        self.cond_dim = self.z_dna_dim + self.proj_dim

        # 对齐 cond_dim
        if hasattr(cfg.model, "cond_dim") and cfg.model.cond_dim != self.cond_dim:
            print(
                f"[WARN] Config cond_dim ({cfg.model.cond_dim}) != calculated ({self.cond_dim}), updating config"
            )
            cfg.model.cond_dim = self.cond_dim

        layers_per_block = getattr(cfg.model, "layers_per_block", getattr(cfg.model, "num_res_blocks", 2))

        # 模型
        self.model = FiLMUNet1D(
            y_dim=self.y_dim,
            cond_dim=self.cond_dim,
            base_channels=cfg.model.base_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=cfg.model.norm_num_groups,
        ).to(self.device)

        # 调度器 / β / ᾱ_t
        self.scheduler = get_scheduler(cfg.train.scheduler_type, cfg.train.num_timesteps)
        self.betas = torch.tensor(self.scheduler.betas, device=self.device, dtype=torch.float32)
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)  # (T,)

        # 优化器
        self.optim = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )

        # y_prior 投影
        self.proj_y = nn.Sequential(
            nn.Linear(self.y_dim, self.proj_dim),
            nn.GELU(),
            nn.Linear(self.proj_dim, self.proj_dim),
        ).to(self.device)

        # 损失
        self.criterion = DiffusionLoss(cfg.train.lambda_rank)

        # AMP
        use_amp = str(getattr(self.cfg.train, "precision", "amp")).lower() == "amp"
        self.scaler = GradScaler(enabled=use_amp and torch.cuda.is_available())

        print(
            f"[INFO] Dimensions: z_dna={self.z_dna_dim}, y={self.y_dim}, "
            f"proj={self.proj_dim}, cond={self.cond_dim}"
        )
        print(f"[INFO] Device: {self.device}, AMP: {self.scaler.is_enabled()}")

    def train_one_epoch(self, epoch, tb_writer=None):
        self.model.train()
        meters = {"total": 0.0, "mse": 0.0, "rank": 0.0}
        n = 0
        pbar = tqdm(self.train_loader, desc=f"Train Ep{epoch:03d}")

        for it, batch in enumerate(pbar):
            y0 = batch["y"].to(self.device)        # (B, N)
            z_dna = batch["z_dna"].to(self.device) # (B, D)
            sids = batch.get("sid", None)          # list[str] or None

            if it == 0:
                print(f"[DEBUG] First batch shapes: y0={y0.shape}, z_dna={z_dna.shape}")

            assert z_dna.shape[1] == self.z_dna_dim, f"z_dna dim mismatch: expected {self.z_dna_dim}, got {z_dna.shape[1]}"
            assert y0.shape[1] == self.y_dim, f"y0 dim mismatch: expected {self.y_dim}, got {y0.shape[1]}"

            with torch.no_grad():
                y_prior = self.ret.query(z_dna, sids=sids)

            assert y_prior.shape[1] == self.y_dim, f"y_prior dim mismatch: expected {self.y_dim}, got {y_prior.shape[1]}"

            y_prior_proj = self.proj_y(y_prior)  # (B, proj_dim)
            cond_vec = torch.cat([z_dna, y_prior_proj], dim=-1)  # (B, cond_dim)

            B = y0.size(0)
            t = torch.randint(0, self.cfg.train.num_timesteps, (B,), device=self.device)
            noise = torch.randn_like(y0)
            y_t = add_noise(y0, noise, t, self.betas)  # (B, N)

            with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
                eps_pred = self.model(y_t, t, cond_vec)  # (B, N)

                # y0 反推
                alphabar_t = self.alphas_cumprod[t]             # (B,)
                sqrt_ab = torch.sqrt(alphabar_t).unsqueeze(-1)  # (B,1)
                sqrt_omab = torch.sqrt(1.0 - alphabar_t).unsqueeze(-1)
                y0_pred = (y_t - sqrt_omab * eps_pred) / (sqrt_ab + 1e-8)

                loss_dict = self.criterion(
                    eps_pred, noise, y_prior, y0, y0_pred=y0_pred
                )
                loss = loss_dict["total"]

            # 首个 batch 打印诊断
            if it == 0:
                zero_eps_mse = F.mse_loss(torch.zeros_like(noise), noise).item()
                prior_mse = F.mse_loss(y_prior, y0).item()
                eps_mse = F.mse_loss(eps_pred, noise).item()
                
                # 计算先验强度指标
                y0_var = y0.var().item()
                prior_correlation = torch.corrcoef(torch.stack([y_prior.flatten(), y0.flatten()]))[0, 1].item()
                
                print(
                    f"[DEBUG] zero_eps_mse={zero_eps_mse:.4f}  "
                    f"prior_mse={prior_mse:.4f}  eps_mse={eps_mse:.4f}"
                )
                print(
                    f"[CHECK] 先验强度: y0_var={y0_var:.4f}, "
                    f"prior_corr={prior_correlation:.4f}, "
                    f"prior_mse/y0_var={prior_mse/y0_var:.4f}"
                )

            # 反传
            self.optim.zero_grad(set_to_none=True)
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), getattr(self.cfg.train, "max_grad_norm", 1.0)
                )
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), getattr(self.cfg.train, "max_grad_norm", 1.0)
                )
                self.optim.step()

            # 统计
            meters["total"] += loss.item() * B
            meters["mse"] += loss_dict["mse"].item() * B
            meters["rank"] += loss_dict["rank"].item() * B
            n += B

            pbar.set_postfix(loss=f"{meters['total'] / n:.4f}")
            if tb_writer and it % 20 == 0:
                tb_writer.add_scalar(
                    "train/loss_step", loss.item(), epoch * len(self.train_loader) + it
                )
                # 每20步记录一次先验强度
                if it % 100 == 0:  # 减少频率，避免日志过多
                    prior_corr = torch.corrcoef(torch.stack([y_prior.flatten(), y0.flatten()]))[0, 1].item()
                    tb_writer.add_scalar("train/prior_correlation", prior_corr, epoch * len(self.train_loader) + it)

        avg = {k: v / n for k, v in meters.items()}
        if tb_writer:
            tb_writer.add_scalar("train/loss_epoch", avg["total"], epoch)
            tb_writer.add_scalar("train/mse_epoch", avg["mse"], epoch)
            tb_writer.add_scalar("train/rank_epoch", avg["rank"], epoch)
        return avg

    @torch.no_grad()
    def evaluate(self, tb_writer=None, epoch=None):
        self.model.eval()
        meters = {"mse": 0.0}
        n = 0
        for batch in self.val_loader:
            y0 = batch["y"].to(self.device)
            z = batch["z_dna"].to(self.device)
            sids = batch.get("sid", None)

            assert z.shape[1] == self.z_dna_dim, f"z dim mismatch: expected {self.z_dna_dim}, got {z.shape[1]}"
            assert y0.shape[1] == self.y_dim, f"y0 dim mismatch: expected {self.y_dim}, got {y0.shape[1]}"

            y_prior = self.ret.query(z, sids=sids)
            y_prior_proj = self.proj_y(y_prior)
            cond_vec = torch.cat([z, y_prior_proj], dim=-1)

            t = torch.randint(0, self.cfg.train.num_timesteps, (y0.size(0),), device=self.device)
            noise = torch.randn_like(y0)
            y_t = add_noise(y0, noise, t, self.betas)

            with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
                eps_pred = self.model(y_t, t, cond_vec)
                alphabar_t = self.alphas_cumprod[t]
                sqrt_ab = torch.sqrt(alphabar_t).unsqueeze(-1)
                sqrt_omab = torch.sqrt(1.0 - alphabar_t).unsqueeze(-1)
                y0_pred = (y_t - sqrt_omab * eps_pred) / (sqrt_ab + 1e-8)

                loss_dict = self.criterion(
                    eps_pred, noise, y_prior, y0, y0_pred=y0_pred
                )

            meters["mse"] += loss_dict["mse"].item() * y0.numel()
            n += y0.numel()

        avg_mse = meters["mse"] / n
        
        # 在评估结束时打印先验强度统计
        if epoch == 0:  # 只在第一个epoch打印
            print(f"[CHECK] 验证集先验强度: avg_mse={avg_mse:.6f}")
        
        if tb_writer and epoch is not None:
            tb_writer.add_scalar("val/mse", avg_mse, epoch)
        return avg_mse

    def save_ckpt(
        self,
        tag: str,
        epoch: Optional[int],
        metrics: Optional[dict],
        dir_: str = "checkpoints/diffusion",
        remove_old: bool = False,
    ) -> Path:
        Path(dir_).mkdir(exist_ok=True, parents=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        run = self.cfg.run_name if hasattr(self.cfg, "run_name") else "diffusion"

        metric_suffix = ""
        if metrics and "mse" in metrics:
            metric_suffix += f"_mse{metrics['mse']:.3f}"

        fpath = Path(dir_) / f"{run}_{tag}_{ts}{metric_suffix}.pt"

        if remove_old:
            for old in Path(dir_).glob(f"{run}_{tag}_*.pt"):
                if old != fpath:
                    old.unlink(missing_ok=True)
                    print(f"[INFO] 删除旧文件: {old.name}")

        # 兼容保存 scheduler
        scheduler_state = None
        if hasattr(self.scheduler, "state_dict"):
            scheduler_state = self.scheduler.state_dict()
        elif hasattr(self.scheduler, "config"):
            scheduler_state = self.scheduler.config

        ckpt = {
            "epoch": epoch,
            "metrics": metrics,
            "model": self.model.state_dict(),
            "optimizer": self.optim.state_dict(),
            "scheduler": scheduler_state,
            "cfg": OmegaConf.to_container(self.cfg, resolve=True),
        }
        torch.save(ckpt, fpath)
        return fpath

    def load(self, path: Union[str, Path]):
        ckpt = torch.load(path, map_location=self.device)

        self.model.load_state_dict(ckpt["model"])

        opt_state = ckpt.get("optimizer")
        if opt_state is not None:
            try:
                self.optim.load_state_dict(opt_state)
            except Exception as e:
                print(f"[WARN] optimizer state load failed: {e}")

        sched_state = ckpt.get("scheduler", None)
        try:
            if sched_state is None:
                self.scheduler = get_scheduler(self.cfg.train.scheduler_type, self.cfg.train.num_timesteps)
            else:
                # 你自己的 scheduler：dict + load_state_dict
                if isinstance(sched_state, dict) and hasattr(self.scheduler, "load_state_dict"):
                    self.scheduler.load_state_dict(sched_state)
                # diffusers：config 对象（带 to_dict）
                elif hasattr(sched_state, "to_dict"):
                    from diffusers import DDPMScheduler
                    self.scheduler = DDPMScheduler.from_config(sched_state)
                else:
                    self.scheduler = get_scheduler(self.cfg.train.scheduler_type, self.cfg.train.num_timesteps)
        except Exception as e:
            print(f"[WARN] scheduler restore failed: {e}")
            self.scheduler = get_scheduler(self.cfg.train.scheduler_type, self.cfg.train.num_timesteps)

        # 重新计算 betas / ᾱ_t
        self.betas = torch.tensor(self.scheduler.betas, device=self.device, dtype=torch.float32)
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        return ckpt