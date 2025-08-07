# -*- coding: utf-8 -*-
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

from gutclip.diffusion.schedulers import get_scheduler
from gutclip.models.diffusion.unet1d_film import FiLMUNet1D


# ======================= 工具函数 =======================

def _to_2d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() <= 1:
        return x.view(1, -1)
    if x.dim() == 2:
        return x
    b = x.size(0)
    return x.view(b, -1)

@torch.no_grad()
def _cos_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    a = _to_2d(a)
    b = _to_2d(b)
    num = (a * b).sum(-1)
    den = a.norm(dim=-1) * b.norm(dim=-1) + eps
    return num / den

@torch.no_grad()
def _k_scale(eps_hat: torch.Tensor, eps_true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    a = _to_2d(eps_hat)
    b = _to_2d(eps_true)
    dot = (a * b).sum(-1)
    den = (b * b).sum(-1) + eps
    return dot / den

# v/ε/x0 互转
@torch.no_grad()
def v_from_eps_x0(eps: torch.Tensor, x0: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return alpha * eps - sigma * x0

@torch.no_grad()
def eps_from_v_xt(v: torch.Tensor, xt: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    denom = alpha.pow(2) + sigma.pow(2)
    return (sigma * xt + alpha * v) / (denom + 1e-12)

@torch.no_grad()
def x0_from_v_xt(v: torch.Tensor, xt: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    denom = alpha.pow(2) + sigma.pow(2)
    return (alpha * xt - sigma * v) / (denom + 1e-12)


# ======================= 检索先验 =======================

class RetrievalIndex:
    """
    - leave-one-out 屏蔽自身
    - Softmax-加权聚合邻居
    - 返回 valid_k, wmax, 熵(entropy)
    """
    def __init__(
        self,
        index_file: str,
        y_file: str,
        ids_file: Optional[str],
        gpu: int = 0,
        k: int = 5,
        metric: str = "l2",
        softmax_temp: float = 0.1,
        sim_thresh: Optional[float] = None
    ):
        import faiss, numpy as np
        from gutclip.data.retrieval_index import _to_gpu

        self.k = int(k)
        self.metric = metric
        self.softmax_temp = float(softmax_temp)
        self.sim_thresh = sim_thresh

        self.idx = faiss.read_index(index_file)
        if gpu >= 0:
            self.idx = _to_gpu(self.idx, gpu)

        self.y = np.load(y_file)  # (B_all, N)
        self.ids = None
        if ids_file is not None and os.path.exists(ids_file):
            self.ids = np.load(ids_file)

        self.norm = bool(np.load(y_file.replace(".y.npy", ".norm.npy")).item())
        self.device = torch.device(f"cuda:{gpu}" if gpu >= 0 else "cpu")

        self.y_mean = torch.from_numpy(self.y.mean(axis=0)).float().to(self.device)

        self._printed_loo_check = False
        self._warned_all_mask_rows = 0

    @torch.no_grad()
    def query(self, z: torch.Tensor, sids: Optional[List[str]] = None):
        """
        返回：
        - y_prior: (B, N)
        - valid_k: (B,)
        - wmax:    (B,)
        - entropy: (B,)  (nats)
        """
        import numpy as np

        B = z.size(0)
        z_np = z.detach().cpu().numpy()
        if self.norm:
            from gutclip.data.retrieval_index import _l2_normalize
            z_np = _l2_normalize(z_np)

        D, I = self.idx.search(z_np.astype("float32"), self.k + 1)  # (B, k+1)
        D_t = torch.from_numpy(D).float()
        I_t = torch.from_numpy(I)

        invalid_mask = (I_t < 0)
        if invalid_mask.any():
            D_t[invalid_mask] = -1e9

        # Leave-one-out
        if sids is not None and self.ids is not None:
            ids_np = self.ids
            hit_before = 0
            mask_self = torch.zeros_like(D_t, dtype=torch.bool)
            for b, sid in enumerate(sids or []):
                if sid is None:
                    continue
                neigh_ids = ids_np[I[b]]
                hit_before += (neigh_ids == sid).sum()
                mask_self[b] = torch.from_numpy(neigh_ids == sid)
            D_t[mask_self] = -1e9
            D_top, top_idx = torch.topk(D_t, k=self.k, dim=1)
            I_top = torch.gather(I_t, 1, top_idx)
            if not self._printed_loo_check:
                total_slots = len(sids) * (self.k + 1)
                print(f"[CHECK] LOO: raw self-hit: {int(hit_before)}/{total_slots}, after mask: 0")
                self._printed_loo_check = True
        else:
            D_top = D_t[:, 1: self.k + 1]
            I_top = I_t[:, 1: self.k + 1]

        valid_top = (I_top >= 0)
        S = D_top.clone()

        # 距离 -> 分数（标准化）
        std = max(S[valid_top].std().item() if valid_top.any() else 1.0, 1e-6)
        if self.metric == "l2":
            S = -S / std
        elif self.metric == "ip":
            S =  S / std
        else:
            S =  S / std

        S[~valid_top] = -1e9
        S = S / self.softmax_temp
        W = torch.softmax(S, dim=1)

        # 阈值筛
        if self.sim_thresh is not None:
            valid_top = valid_top & (S > self.sim_thresh)
            W = W * valid_top.float()

        I_top_np = I_top.cpu().numpy()
        y_neighbors = torch.from_numpy(self.y[I_top_np]).float()  # (B,k,N)

        W_sum = W.sum(dim=1, keepdim=True)
        W_norm = torch.where(W_sum > 0, W / (W_sum + 1e-8), W)

        y_prior = (y_neighbors * W_norm.unsqueeze(-1)).sum(1).to(self.device)

        valid_k = valid_top.sum(1)
        wmax = W_norm.max(dim=1).values
        entropy = -(W_norm.clamp_min(1e-12) * W_norm.clamp_min(1e-12).log()).sum(1)

        # 极端无有效邻居：回退到全局均值
        need_fallback = (W_sum.squeeze(1) == 0) | (valid_k == 0)
        if need_fallback.any():
            cnt = int(need_fallback.sum().item())
            y_prior[need_fallback] = self.y_mean
            self._warned_all_mask_rows += cnt
            if self._warned_all_mask_rows <= 1:
                print(f"[WARN] {cnt} rows had no/low valid neighbors; fell back to global mean y.")

        return y_prior, valid_k.to(self.device), wmax.to(self.device), entropy.to(self.device)


# ======================= 损失（v-pred + 高SNR校准） =======================

class VPredLoss(nn.Module):
    def __init__(self,
                 lambda_rank: float = 0.0,
                 gamma_min_snr: float = 6.0,
                 lambda_high: float = 0.22,
                 lambda_mag: float = 3.0,
                 lambda_ang: float = 0.6,
                 high_center: float = 5.9,
                 high_tau: float = 0.6,
                 lambda_norm: float = 0.50,
                 lambda_ortho: float = 0.6,
                 loss_scale: float = 100.0):  # 添加loss缩放因子，默认100倍
        super().__init__()
        self.lambda_rank = float(lambda_rank)
        self.gamma = float(gamma_min_snr)
        self.lambda_high = float(lambda_high)
        self.lambda_mag = float(lambda_mag)
        self.lambda_ang = float(lambda_ang)
        self.high_center = float(high_center)
        self.high_tau = float(high_tau)
        self.lambda_norm = float(lambda_norm)
        self.lambda_ortho = float(lambda_ortho)
        self.loss_scale = float(loss_scale)  # 保存缩放因子

    def forward(self,
                v_pred: torch.Tensor,
                v_true: torch.Tensor,
                snr: torch.Tensor,
                xt: torch.Tensor,
                eps_true: torch.Tensor,
                alpha: torch.Tensor,
                sigma: torch.Tensor,
                y0_true: torch.Tensor,
                enable_high_cal: bool,
                epoch: int = 0) -> Dict[str, torch.Tensor]:

        # ---- 两阶段权重策略 ----
        if epoch < 2:
            # Ep<2: 统一权重，先把高段样本量"喂饱"
            w_main = torch.ones_like(snr)
        else:
            # Ep>=2: 温和上权高段
            w_main = snr / (snr + self.gamma)  # 单调递增，饱和于1
            w_main = w_main / (w_main.mean() + 1e-8)  # 归一到均值≈1

        mse_v = ((v_pred - v_true) ** 2).mean(dim=tuple(range(1, v_pred.ndim)))
        loss_main = (w_main * mse_v).mean()

        # 计算 ε̂ 与 \hat x0
        eps_hat = eps_from_v_xt(v_pred, xt, alpha, sigma)
        x0_hat  = x0_from_v_xt(v_pred, xt, alpha, sigma)

        # 高 SNR 校准（建议在 epoch>=3 再启用）
        if enable_high_cal and self.lambda_high > 0:
            logsnr = torch.log((alpha * alpha) / (sigma * sigma + 1e-12) + 1e-12).view(-1)
            gate = torch.sigmoid((logsnr - self.high_center) / self.high_tau)
            gate = torch.clamp((gate ** 1.5) * 1.3, 0.0, 1.0)

            k = _k_scale(eps_hat, eps_true)
            mag_loss = (k - 1.0) ** 2
            cos = _cos_sim(eps_hat, eps_true)
            ang_loss = (1.0 - cos).pow(2)  # 改为平方，更强地压小角误差

            eps_ = 1e-12
            n_hat = _to_2d(eps_hat).norm(dim=-1) + eps_
            n_true = _to_2d(eps_true).norm(dim=-1) + eps_
            norm_loss = (torch.log(n_hat) - torch.log(n_true)) ** 2

            # ---- 方向正交项：直接惩罚 eps_hat 在 eps_true 正交方向上的能量 ----
            a2d = _to_2d(eps_hat)
            b2d = _to_2d(eps_true)
            dot = (a2d * b2d).sum(-1, keepdim=True)
            den = (b2d * b2d).sum(-1, keepdim=True).clamp_min(1e-12)
            proj = (dot / den) * b2d                 # eps_hat 在 eps_true 上的投影
            orth = a2d - proj                        # 正交分量
            orth_loss = (orth.pow(2).sum(-1) / (b2d.pow(2).sum(-1).clamp_min(1e-12))).mean()

            cal = (gate * (self.lambda_mag * mag_loss +
                           self.lambda_ang * ang_loss +
                           self.lambda_norm * norm_loss +
                           self.lambda_ortho * orth_loss)).mean()
            loss_cal = self.lambda_high * cal
        else:
            loss_cal = v_pred.new_tensor(0.0)

        # Spearman 排名（可选）
        if self.lambda_rank > 0:
            rank = self._spearman_loss(x0_hat, y0_true)
        else:
            rank = v_pred.new_tensor(0.0)

        total = loss_main + loss_cal + self.lambda_rank * rank
        
        # 应用缩放因子以便可视化
        total_scaled = total * self.loss_scale
        loss_main_scaled = loss_main.detach() * self.loss_scale
        loss_cal_scaled = loss_cal.detach() * self.loss_scale
        rank_scaled = rank.detach() * self.loss_scale
        
        return {
            "total": total_scaled, 
            "v_mse": loss_main_scaled, 
            "cal": loss_cal_scaled, 
            "rank": rank_scaled,
            "total_raw": total,  # 保留原始值用于反向传播
            "v_mse_raw": loss_main.detach(),
            "cal_raw": loss_cal.detach(), 
            "rank_raw": rank.detach()
        }

    @staticmethod
    def _spearman_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        pred = _to_2d(pred).float()
        target = _to_2d(target).float()
        def _rank(x: torch.Tensor) -> torch.Tensor:
            _, idx = torch.sort(x, dim=-1)
            r = torch.zeros_like(x)
            arange = torch.arange(x.size(-1), device=x.device, dtype=x.dtype)
            r.scatter_(dim=-1, index=idx, src=arange.expand_as(x))
            return r
        rp = _rank(pred); rt = _rank(target)
        rp = rp - rp.mean(dim=-1, keepdim=True)
        rt = rt - rt.mean(dim=-1, keepdim=True)
        num = (rp * rt).sum(dim=-1)
        den = torch.sqrt((rp * rp).sum(dim=-1) * (rt * rt).sum(dim=-1) + eps)
        rho = torch.clamp(num / (den + eps), -1.0, 1.0)
        return (1.0 - rho).mean()


# ======================= 训练器 =======================

class TrainerDiffusion:
    def __init__(self, cfg, dataloader: DataLoader, val_loader: DataLoader, retrieval: RetrievalIndex, device=None):
        self.device = device if device is not None else torch.device("cuda:0")
        self.cfg = cfg
        self.train_loader = dataloader
        self.val_loader = val_loader
        self.ret = retrieval

        # 维度
        self.z_dna_dim = int(cfg.model.z_dna_dim)
        self.y_dim = int(cfg.model.y_dim)
        self.proj_dim = int(cfg.model.proj_dim)
        self.cond_dim = self.z_dna_dim + self.proj_dim
        if hasattr(cfg.model, "cond_dim") and cfg.model.cond_dim != self.cond_dim:
            print(f"[WARN] Config cond_dim ({cfg.model.cond_dim}) != calculated ({self.cond_dim}), updating config")
            cfg.model.cond_dim = self.cond_dim

        layers_per_block = getattr(cfg.model, "layers_per_block", getattr(cfg.model, "num_res_blocks", 2))

        # 模型 + 融合头
        self.model = FiLMUNet1D(
            y_dim=self.y_dim,
            cond_dim=self.cond_dim,
            base_channels=cfg.model.base_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=cfg.model.norm_num_groups,
        ).to(self.device)

        self.head = self._FusionHead(self.cond_dim, self.y_dim, r=8).to(self.device)

        # 调度器 & 预计算
        self.scheduler = get_scheduler(cfg.train.scheduler_type, cfg.train.num_timesteps)
        self.betas = torch.as_tensor(self.scheduler.betas, device=self.device, dtype=torch.float32)
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)  # (T,)
        self.alpha_t = torch.sqrt(self.alphas_cumprod)
        self.sigma_t = torch.sqrt(1.0 - self.alphas_cumprod)
        self.logsnr_t = torch.log((self.alpha_t ** 2) / (self.sigma_t ** 2 + 1e-12) + 1e-12)

        # 训练超参
        tr = cfg.train
        self.sigma_min = float(getattr(tr, "sigma_min", 0.03))
        self.logsnr_min = float(getattr(tr, "logsnr_min", -6.0))
        self.logsnr_max = float(getattr(tr, "logsnr_max", 8.0))
        self.phase1_epochs = int(getattr(tr, "phase1_epochs", 2))
        self.p_high_snr = float(getattr(tr, "p_high_snr", 0.20))

        # 优化器（包含 head 参数）
        params = list(self.model.parameters()) + list(self.head.parameters())
        self.optim = torch.optim.AdamW(
            params,
            lr=float(getattr(tr, "lr", 2e-4)),
            weight_decay=float(getattr(tr, "weight_decay", 0.01)),
        )

        # y_prior 投影
        self.proj_y = nn.Sequential(
            nn.Linear(self.y_dim, self.proj_dim), nn.GELU(), nn.Linear(self.proj_dim, self.proj_dim)
        ).to(self.device)

        # 损失器
        self.criterion = VPredLoss(
            lambda_rank=float(getattr(tr, "lambda_rank", 0.0)),
            gamma_min_snr=float(getattr(tr, "gamma_min_snr", 6.0)),
            lambda_high=float(getattr(tr, "lambda_high", 0.22)),
            lambda_mag=float(getattr(tr, "lambda_mag", 3.0)),
            lambda_ang=float(getattr(tr, "lambda_ang", 0.6)),
            high_center=float(getattr(tr, "high_snr_center", 5.9)),
            high_tau=float(getattr(tr, "high_snr_tau", 0.6)),
            lambda_norm=float(getattr(tr, "lambda_norm", 0.50)),
            lambda_ortho=float(getattr(tr, "lambda_ortho", 0.6)),
        )

        # 让训练器的中心/温度与 criterion 保持一致
        self.high_center = self.criterion.high_center
        self.high_tau = self.criterion.high_tau

        # 融合权重
        self.lambda_fuse = float(getattr(tr, "lambda_fuse", 0.05))
        self.lambda_reg  = float(getattr(tr, "lambda_reg", 1e-4))

        # AMP
        use_amp = str(getattr(tr, "precision", "amp")).lower() == "amp"
        self.scaler = GradScaler('cuda', enabled=use_amp and torch.cuda.is_available())

        print(f"[INFO] Dimensions: z_dna={self.z_dna_dim}, y={self.y_dim}, proj={self.proj_dim}, cond={self.cond_dim}")
        print(f"[INFO] Device: {self.device}, AMP: {self.scaler.is_enabled()}")

        # 有效 t
        self.valid_t_mask = (self.sigma_t >= self.sigma_min) & (self.logsnr_t >= self.logsnr_min) & (self.logsnr_t <= self.logsnr_max)
        if not torch.any(self.valid_t_mask):
            raise ValueError("No valid t after applying sigma_min/logSNR range.")

        # GateA 适配器
        self.gateA_scheduler = self._GateASchedulerAdapter(self.alpha_t, self.sigma_t, self.valid_t_mask, cfg.train.num_timesteps)

        # 训练策略
        self._high_toggle = False
        self.BOOSTER = False   # 重要：关闭强制高SNR
        self.last_gateA_worst_bucket = None  # 'low', 'mid', 'high'

        print(f"[CFG] high_center={self.high_center}  high_tau={self.high_tau}  p_high_snr={self.p_high_snr}  "
              f"lambda_high={self.criterion.lambda_high}  lambda_mag={self.criterion.lambda_mag}  "
              f"lambda_ang={self.criterion.lambda_ang}  lambda_norm={self.criterion.lambda_norm}  "
              f"lambda_ortho={self.criterion.lambda_ortho}  "
              f"lambda_fuse={self.lambda_fuse}  lambda_reg={self.lambda_reg}")

    # ------- 可学习融合头 -------
    class _FusionHead(nn.Module):
        def __init__(self, cond_dim, y_dim, r=8):
            super().__init__()
            self.k_head = nn.Sequential(nn.Linear(cond_dim, 64), nn.ReLU(), nn.Linear(64, 1))
            self.w_head = nn.Sequential(nn.Linear(cond_dim, 64), nn.ReLU(), nn.Linear(64, 1))
            self.U = nn.Linear(cond_dim, r)
            self.B = nn.Parameter(torch.randn(y_dim, r) * 1e-3)

        def forward(self, cond):
            k = 1.0 + 0.1 * torch.tanh(self.k_head(cond))  # 防发散
            w = torch.sigmoid(self.w_head(cond))
            u = self.U(cond)
            return k.squeeze(-1), w.squeeze(-1), u, self.B

    # ------- GateA 适配器 -------
    class _GateASchedulerAdapter:
        def __init__(self, alpha_t: torch.Tensor, sigma_t: torch.Tensor, valid_mask: torch.Tensor, T: int):
            self.alpha_t = alpha_t
            self.sigma_t = sigma_t
            self.valid_idx = torch.nonzero(valid_mask, as_tuple=False).view(-1)
            self.T = int(T)
        @torch.no_grad()
        def sample_t(self, B: int, device: torch.device):
            choice = torch.randint(0, self.valid_idx.numel(), (B,), device=device)
            return self.valid_idx[choice].to(device)
        @torch.no_grad()
        def alpha_sigma(self, t: torch.Tensor):
            a = self.alpha_t[t]
            s = self.sigma_t[t]
            return a.view(-1, 1), s.view(-1, 1)

    # ------- 冻结/解冻 head -------
    def _set_head_requires_grad(self, flag: bool):
        for p in self.head.parameters():
            p.requires_grad = flag

    # ------- t 采样 -------
    @torch.no_grad()
    def _sample_t_logsnr(self, B: int, force_high: bool = False) -> torch.Tensor:
        device = self.device
        s_low = self.logsnr_min
        s_high = self.logsnr_max
        if force_high:
            band = 0.5
            s_low = max(self.logsnr_min, self.high_center - band)
            s_high = min(self.logsnr_max, self.high_center + band)

        s = torch.empty(B, device=device).uniform_(s_low, s_high)
        idx_valid = torch.nonzero(self.valid_t_mask, as_tuple=False).view(-1)
        s_valid = self.logsnr_t[idx_valid]
        j = torch.argmin((s.view(-1, 1) - s_valid.view(1, -1)).abs(), dim=1)
        t = idx_valid[j]
        return t

    @torch.no_grad()
    def _sample_in_range(self, B: int, s_low: float, s_high: float):
        s = torch.empty(B, device=self.device).uniform_(s_low, s_high)
        idx_valid = torch.nonzero(self.valid_t_mask, as_tuple=False).view(-1)
        s_valid = self.logsnr_t[idx_valid]
        j = torch.argmin((s[:, None] - s_valid[None, :]).abs(), dim=1)
        return idx_valid[j]

    def _phase_flags(self, epoch: int) -> Tuple[bool, bool]:
        # epoch>=2 启用高SNR校准；采样开关我们总是使用三段概率
        enable_cal = epoch >= 2
        enable_high_sample = True
        return enable_cal, enable_high_sample

    def update_gateA_results(self, buckets: Dict[str, Any]):
        if not buckets or 'buckets' not in buckets:
            return
        worst_bucket, worst_nmse = None, -1.0
        for binfo in buckets['buckets']:
            if 'nmse_mean' in binfo:
                nmse = binfo['nmse_mean']
                if nmse > worst_nmse:
                    worst_nmse = nmse
                    worst_bucket = binfo.get('bucket_type', 'mid')
        if worst_bucket:
            self.last_gateA_worst_bucket = worst_bucket
            print(f"[GateA] Worst bucket: {worst_bucket} (NMSE: {worst_nmse:.4f})")

    @staticmethod
    def _spearman_loss(pred, target, eps: float = 1e-6):
        pred = _to_2d(pred).float()
        target = _to_2d(target).float()
        _, idxp = torch.sort(pred, dim=-1)
        rp = torch.zeros_like(pred).scatter(-1, idxp, torch.arange(pred.size(-1), device=pred.device, dtype=pred.dtype).expand_as(pred))
        _, idxt = torch.sort(target, dim=-1)
        rt = torch.zeros_like(target).scatter(-1, idxt, torch.arange(target.size(-1), device=target.device, dtype=target.dtype).expand_as(target))
        rp -= rp.mean(-1, keepdim=True); rt -= rt.mean(-1, keepdim=True)
        num = (rp * rt).sum(-1)
        den = torch.sqrt((rp * rp).sum(-1) * (rt * rt).sum(-1) + eps)
        rho = torch.clamp(num / (den + eps), -1, 1)
        return (1 - rho).mean()

    # ------- 训练一个 epoch -------
    def train_one_epoch(self, epoch: int, tb_writer=None):
        self.model.train()
        self.head.train()

        # Ep<2 冻结 head，Ep>=2 再解冻
        self._set_head_requires_grad(epoch >= 2)

        meters = {"total": 0.0, "v_mse": 0.0, "cal": 0.0, "rank": 0.0,
                  "fuse_mse": 0.0, "fuse_spr": 0.0}
        n = 0

        # 采样占比统计
        snr_buckets = {"low": 0, "mid": 0, "high": 0}

        enable_cal, _ = self._phase_flags(epoch)

        pbar = tqdm(self.train_loader, desc=f"Train Ep{epoch:03d}")

        for it, batch in enumerate(pbar):
            y0 = batch["y"].to(self.device)        # (B, N)
            z_dna = batch["z_dna"].to(self.device) # (B, D)
            sids = batch.get("sid", None)

            if it == 0:
                print(f"[DEBUG] First batch shapes: y0={y0.shape}, z_dna={z_dna.shape}")

            with torch.no_grad():
                y_prior, valid_k, wmax, H = self.ret.query(z_dna, sids=sids)

            y_prior_proj = self.proj_y(y_prior)
            cond_vec = torch.cat([z_dna, y_prior_proj], dim=-1)

            B = y0.size(0)

            # ---- 三段概率采样 ----
            if epoch < 2:
                p_low, p_mid, p_high = 0.15, 0.15, 0.70  # Ep0-1: 高段>70%
            else:
                p_low, p_mid, p_high = 0.30, 0.30, 0.40  # Ep2+: 高段40%

            if self.last_gateA_worst_bucket == 'low':
                p_low, p_mid, p_high = 0.50, 0.35, 0.15
            elif self.last_gateA_worst_bucket == 'high':
                p_low, p_mid, p_high = 0.25, 0.35, 0.40

            r = torch.rand((), device=self.device).item()
            if r < p_low:
                t = self._sample_in_range(B, s_low=-2.0, s_high=1.0)
                snr_buckets["low"] += B
            elif r < p_low + p_mid:
                t = self._sample_in_range(B, s_low=1.0,  s_high=4.5)
                snr_buckets["mid"] += B
            else:
                t = self._sample_in_range(B, s_low=5.2,  s_high=6.6)
                snr_buckets["high"] += B

            alpha = self.alpha_t[t].view(-1, 1)
            sigma = self.sigma_t[t].view(-1, 1)
            snr = (alpha * alpha) / (sigma * sigma + 1e-12)

            if it % 50 == 0:
                logsnr_batch = torch.log((alpha * alpha) / (sigma * sigma + 1e-12) + 1e-12).view(-1)
                print(f"[DBG] logSNR mean={logsnr_batch.mean():.3f} max={logsnr_batch.max():.3f}")

            noise = torch.randn_like(y0)
            xt = alpha * y0 + sigma * noise

            with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
                v_true = v_from_eps_x0(noise, y0, alpha, sigma)
                v_pred = self.model(xt, t, cond_vec)

                # 融合头（Ep<2 仅前向，不训练）
                if epoch < 2:
                    with torch.no_grad():
                        k_corr, w_gate, u_coef, B_basis = self.head(cond_vec)
                else:
                    k_corr, w_gate, u_coef, B_basis = self.head(cond_vec)

                # 校正头安全夹取
                k_corr = k_corr.clamp(0.8, 1.25)  # 防止过度校正

                # 对低秩偏置在高 SNR 处衰减（避免把先验形状"刻"进 eps）
                logsnr = torch.log((alpha*alpha)/(sigma*sigma + 1e-12) + 1e-12).view(-1, 1)
                head_decay = torch.sigmoid((self.high_center - logsnr) / self.high_tau)  # 高SNR→小
                u_coef = u_coef * head_decay  # u_coef 的有效幅度按 head_decay 缩小

                eps_hat = eps_from_v_xt(v_pred, xt, alpha, sigma)
                eps_hat_corr = k_corr.view(-1, 1) * eps_hat + u_coef @ B_basis.T

                v_corr = v_from_eps_x0(eps_hat_corr, y0, alpha, sigma)
                x0_hat = x0_from_v_xt(v_corr, xt, alpha, sigma)
                y0_mix = w_gate.view(-1, 1) * x0_hat + (1. - w_gate.view(-1, 1)) * y_prior

                # 主损失（v-pred）
                loss_dict = self.criterion(
                    v_pred=v_pred,
                    v_true=v_true.detach(),
                    snr=snr.view(-1),
                    xt=xt.float(),
                    eps_true=noise.detach().float(),
                    alpha=alpha.float(),
                    sigma=sigma.float(),
                    y0_true=y0.float(),
                    enable_high_cal=enable_cal,
                    epoch=epoch,
                )

                # 融合损失 / 正则
                loss_fuse = v_pred.new_tensor(0.0)
                loss_reg  = v_pred.new_tensor(0.0)

                if epoch >= 2:
                    fuse_mse = F.mse_loss(y0_mix, y0)
                    fuse_spr = self._spearman_loss(y0_mix, y0)
                    loss_fuse = fuse_mse + 0.1 * fuse_spr

                    # 置信度（熵归一）
                    k_eff = valid_k.clamp(min=1).float()
                    H_norm = (H / k_eff.log().clamp_min(1e-6)).clamp(0, 1)
                    prior_conf = (valid_k.float() / self.ret.k) * wmax * (1.0 - H_norm)
                    prior_conf = prior_conf.clamp(0, 1).detach()

                    # 正则：k→1、U 小、先验高可信时 w 小
                    loss_reg = ((k_corr - 1) ** 2).mean() \
                               + self.lambda_reg * (u_coef ** 2).mean() \
                               + 0.1 * (prior_conf * w_gate).mean()

                # 使用原始loss进行反向传播，但记录缩放后的loss
                loss_raw = loss_dict["total_raw"] + self.lambda_fuse * loss_fuse + loss_reg
                loss_scaled = loss_dict["total"] + self.lambda_fuse * loss_fuse * self.criterion.loss_scale + loss_reg * self.criterion.loss_scale

            # 诊断
            if it == 0:
                with torch.no_grad():
                    eps_hat_base = eps_from_v_xt(v_pred, xt, alpha, sigma)
                    k_dbg = _k_scale(eps_hat_base, noise)
                    cos_dbg = _cos_sim(eps_hat_base, noise)
                    print(f"[DEBUG] v_mse={loss_dict['v_mse']:.4f}  k_mean={k_dbg.mean().item():.3f}  cos_mean={cos_dbg.mean().item():.3f}")
                    logsnr = torch.log((alpha*alpha)/(sigma*sigma+1e-12) + 1e-12).view(-1)
                    gate = torch.sigmoid((logsnr - self.high_center) / self.high_tau)
                    print(f"[DBG] gate_high mean={gate.mean():.3f} min={gate.min():.3f} max={gate.max():.3f}")

            # 反传（使用原始loss）
            self.optim.zero_grad(set_to_none=True)
            if self.scaler.is_enabled():
                self.scaler.scale(loss_raw).backward()
                self.scaler.unscale_(self.optim)
                clip = float(getattr(self.cfg.train, "max_grad_norm", getattr(self.cfg.train, "grad_clip", 0.5)))
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                torch.nn.utils.clip_grad_norm_(self.head.parameters(), clip)
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                loss_raw.backward()
                clip = float(getattr(self.cfg.train, "max_grad_norm", getattr(self.cfg.train, "grad_clip", 0.5)))
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                torch.nn.utils.clip_grad_norm_(self.head.parameters(), clip)
                self.optim.step()

            # 记录缩放后的loss用于可视化
            meters["total"] += loss_scaled.item() * B
            meters["v_mse"] += loss_dict["v_mse"].item() * B
            meters["cal"]   += loss_dict["cal"].item() * B
            meters["rank"]  += loss_dict["rank"].item() * B
            if epoch >= 2:
                meters["fuse_mse"] += fuse_mse.item() * B
                meters["fuse_spr"] += fuse_spr.item() * B
            else:
                meters["fuse_mse"] += 0.0 * B
                meters["fuse_spr"] += 0.0 * B
            n += B

            pbar.set_postfix(loss=f"{meters['total']/n:.4f}")
            if tb_writer and it % 20 == 0:
                step = epoch * len(self.train_loader) + it
                tb_writer.add_scalar("train/loss_step", loss_scaled.item(), step)

        avg = {k: v / n for k, v in meters.items()}
        
        # 打印采样占比
        total_samples = sum(snr_buckets.values())
        if total_samples > 0:
            pct_low = snr_buckets["low"] / total_samples * 100
            pct_mid = snr_buckets["mid"] / total_samples * 100
            pct_high = snr_buckets["high"] / total_samples * 100
            print(f"[Sampling] Ep{epoch}: low={pct_low:.1f}% mid={pct_mid:.1f}% high={pct_high:.1f}%")
        
        if tb_writer:
            tb_writer.add_scalar("train/total_epoch", avg["total"], epoch)
            tb_writer.add_scalar("train/v_mse_epoch", avg["v_mse"], epoch)
            tb_writer.add_scalar("train/cal_epoch", avg["cal"], epoch)
            tb_writer.add_scalar("train/rank_epoch", avg["rank"], epoch)
            tb_writer.add_scalar("train/fuse_mse_epoch", avg["fuse_mse"], epoch)
            tb_writer.add_scalar("train/fuse_spr_epoch", avg["fuse_spr"], epoch)
        return avg

    # ------- 验证 -------
    @torch.no_grad()
    def evaluate(self, tb_writer=None, epoch=0):
        self.model.eval()
        self.head.eval()

        meters = {"v_mse": 0.0, "y0_mix_mse": 0.0, "y0_mix_spr": 0.0}
        n = 0

        for batch in self.val_loader:
            y0 = batch["y"].to(self.device)
            z = batch["z_dna"].to(self.device)
            sids = batch.get("sid", None)

            y_prior, valid_k, wmax, H = self.ret.query(z, sids=sids)
            y_prior_proj = self.proj_y(y_prior)
            cond_vec = torch.cat([z, y_prior_proj], dim=-1)

            B = y0.size(0)
            t = self._sample_t_logsnr(B, force_high=False)
            alpha = self.alpha_t[t].view(-1, 1)
            sigma = self.sigma_t[t].view(-1, 1)
            snr = (alpha * alpha) / (sigma * sigma + 1e-12)
            noise = torch.randn_like(y0)
            xt = alpha * y0 + sigma * noise

            with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
                v_true = v_from_eps_x0(noise, y0, alpha, sigma)
                v_pred = self.model(xt, t, cond_vec)

                # 与训练完全一致的权重（使用当前epoch的权重策略）
                gamma = float(self.criterion.gamma)
                if epoch < 2:
                    w_main = torch.ones_like(snr)
                else:
                    w_main = snr / (snr + gamma)
                    w_main = w_main / (w_main.mean() + 1e-8)

                mse_v = ((v_pred - v_true) ** 2).mean(dim=tuple(range(1, v_pred.ndim)))
                loss_main = (w_main * mse_v).mean()

                # 融合输出
                k_corr, w_gate, u_coef, B_basis = self.head(cond_vec)
                eps_hat = eps_from_v_xt(v_pred, xt, alpha, sigma)
                eps_hat_corr = k_corr.view(-1, 1) * eps_hat + u_coef @ B_basis.T

                v_corr = v_from_eps_x0(eps_hat_corr, y0, alpha, sigma)
                x0_hat = x0_from_v_xt(v_corr, xt, alpha, sigma)
                y0_mix = w_gate.view(-1, 1) * x0_hat + (1. - w_gate.view(-1, 1)) * y_prior

                mix_mse = F.mse_loss(y0_mix, y0)
                mix_spr = self._spearman_loss(y0_mix, y0)

            meters["v_mse"] += loss_main.item() * B
            meters["y0_mix_mse"] += mix_mse.item() * B
            meters["y0_mix_spr"] += mix_spr.item() * B
            n += B

        v_mse = meters["v_mse"] / n
        mix_mse = meters["y0_mix_mse"] / n
        mix_spr = meters["y0_mix_spr"] / n

        if tb_writer and epoch is not None:
            tb_writer.add_scalar("val/v_mse", v_mse, epoch)
            tb_writer.add_scalar("val/y0_mix_mse", mix_mse, epoch)
            tb_writer.add_scalar("val/y0_mix_spr", mix_spr, epoch)
        return v_mse

    # ------- Gate A 快速评估 -------
    @torch.no_grad()
    def eval_gateA_once(self, max_batches: int = 50, num_bins: int = 12, use_autocast: bool = True):
        device = self.device
        scheduler = self.gateA_scheduler
        use_head = True  # 默认启用 head；如需按 epoch 控制可自行改写

        # 构造 val 流：{"x0","cond"}
        def _val_iter():
            for batch in self.val_loader:
                y0 = batch["y"].to(device)
                z  = batch["z_dna"].to(device)
                sids = batch.get("sid", None)
                y_prior, _, _, _ = self.ret.query(z, sids=sids)
                y_prior_proj = self.proj_y(y_prior)
                cond_vec = torch.cat([z, y_prior_proj], dim=-1)
                yield {"x0": y0, "cond": cond_vec}

        class _Wrapper:
            def __iter__(self): return _val_iter()

        # v̂ → ε̂ （含校正头）
        class _V2Eps(nn.Module):
            def __init__(self, base, head, alpha_t, sigma_t, use_head):
                super().__init__()
                self.base = base
                self.head = head
                self.alpha_t = alpha_t
                self.sigma_t = sigma_t
                self.use_head = use_head
            def forward(self, xt, t, cond):
                v_hat = self.base(xt, t, cond)
                a = self.alpha_t[t].view(-1, *([1] * (xt.ndim - 1)))
                s = self.sigma_t[t].view(-1, *([1] * (xt.ndim - 1)))
                eps_hat = (s * xt + a * v_hat) / (a.pow(2) + s.pow(2) + 1e-12)
                if self.use_head:
                    k, w, u, B = self.head(cond)
                    eps_hat = k.view(-1,1) * eps_hat + u @ B.T
                return eps_hat

        proxy = _V2Eps(self.model, self.head, self.alpha_t, self.sigma_t, use_head).to(device)

        from gutclip.evaluate.eval_gateA import evaluate_gateA
        verdict, buckets, raw, suggestions = evaluate_gateA(
            model=proxy,
            val_loader=_Wrapper(),
            scheduler=scheduler,
            device=device,
            num_bins=num_bins,
            max_batches=max_batches,
            use_autocast=use_autocast,
        )
        return verdict, buckets, raw, suggestions

    # ------- Gate B 残差校准与采样稳定性评估 -------
    @torch.no_grad()
    def eval_gateB_once(self, max_batches: int = 20, num_ddim_steps: int = 50, use_autocast: bool = True):
        device = self.device
        scheduler = self.gateA_scheduler  # 使用相同的调度器
        use_head = True

        # 构造 val 流：{"x0","xt","eps","t","alpha","sigma","cond_vec"}
        def _val_iter():
            for batch in self.val_loader:
                y0 = batch["y"].to(device)
                z  = batch["z_dna"].to(device)
                sids = batch.get("sid", None)
                y_prior, _, _, _ = self.ret.query(z, sids=sids)
                y_prior_proj = self.proj_y(y_prior)
                cond_vec = torch.cat([z, y_prior_proj], dim=-1)
                
                # 生成噪声和加噪样本
                B = y0.shape[0]
                t = torch.randint(0, self.T, (B,), device=device)
                alpha_t = self.alpha_t[t].view(-1, *([1] * (y0.ndim - 1)))
                sigma_t = self.sigma_t[t].view(-1, *([1] * (y0.ndim - 1)))
                eps = torch.randn_like(y0)
                xt = alpha_t * y0 + sigma_t * eps
                
                yield {
                    "x0": y0,
                    "xt": xt,
                    "eps": eps,
                    "t": t,
                    "alpha": alpha_t.view(-1),
                    "sigma": sigma_t.view(-1),
                    "cond_vec": cond_vec
                }

        class _Wrapper:
            def __iter__(self): return _val_iter()

        # v̂ → ε̂ （含校正头）
        class _V2Eps(nn.Module):
            def __init__(self, base, head, alpha_t, sigma_t, use_head):
                super().__init__()
                self.base = base
                self.head = head
                self.alpha_t = alpha_t
                self.sigma_t = sigma_t
                self.use_head = use_head
            def forward(self, xt, t, cond):
                v_hat = self.base(xt, t, cond)
                a = self.alpha_t[t].view(-1, *([1] * (xt.ndim - 1)))
                s = self.sigma_t[t].view(-1, *([1] * (xt.ndim - 1)))
                eps_hat = (s * xt + a * v_hat) / (a.pow(2) + s.pow(2) + 1e-12)
                if self.use_head:
                    k, w, u, B = self.head(cond)
                    eps_hat = k.view(-1,1) * eps_hat + u @ B.T
                return eps_hat

        proxy = _V2Eps(self.model, self.head, self.alpha_t, self.sigma_t, use_head).to(device)

        from gutclip.evaluate.eval_gateB import evaluate_gateB
        verdict, raw_metrics, suggestions = evaluate_gateB(
            model=proxy,
            val_loader=_Wrapper(),
            scheduler=scheduler,
            device=device,
            max_batches=max_batches,
            use_autocast=use_autocast,
            num_ddim_steps=num_ddim_steps,
        )
        return verdict, raw_metrics, suggestions

    # ------- 保存/加载 -------
    def save_ckpt(self, tag: str, epoch: Optional[int], metrics: Optional[dict],
                  dir_: str = "checkpoints/diffusion", remove_old: bool = False) -> Path:
        Path(dir_).mkdir(exist_ok=True, parents=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        run = self.cfg.run_name if hasattr(self.cfg, "run_name") else "diffusion"
        metric_suffix = ""
        if metrics and "v_mse" in metrics:
            metric_suffix += f"_vmse{metrics['v_mse']:.3f}"
        fpath = Path(dir_) / f"{run}_{tag}_{ts}{metric_suffix}.pt"

        if remove_old:
            for old in Path(dir_).glob(f"{run}_{tag}_*.pt"):
                if old != fpath:
                    old.unlink(missing_ok=True)
                    print(f"[INFO] 删除旧文件: {old.name}")

        ckpt = {
            "epoch": epoch,
            "metrics": metrics,
            "model": self.model.state_dict(),
            "head": self.head.state_dict(),
            "optimizer": self.optim.state_dict(),
            "cfg": OmegaConf.to_container(self.cfg, resolve=True),
        }
        torch.save(ckpt, fpath)
        return fpath

    def load(self, path: Union[str, Path]):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        if "head" in ckpt:
            self.head.load_state_dict(ckpt["head"])
        opt_state = ckpt.get("optimizer")
        if opt_state is not None:
            try:
                self.optim.load_state_dict(opt_state)
            except Exception as e:
                print(f"[WARN] optimizer state load failed: {e}")
        return ckpt