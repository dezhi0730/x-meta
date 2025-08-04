"""
关卡A：噪声匹配评估 (Gate A: Noise Matching Evaluation)

这是扩散模型质量检测的第一关，专注于验证噪声匹配是否达标。

核心概念：
- 扩散模型的本质任务是：给定 (x_t, t, c) 正确预测噪声 ε
- 如果在主要噪声强度区间上 ε 的方向与尺度都对，用它反推的 x̂₀ 也会对
- 关卡A把问题分桶到不同 SNR，能暴露"某些区间没学会"的盲区

失败意味着：
- 高 SNR 桶差：强噪声下学不会，常见于 sigma_min 太小或目标参数化不合适
- 低 SNR 桶好、x₀ 却差：数值标定/反推公式有问题，或 ε 方向虽对但幅度偏
- 曲线波动大：学习率过高、正则不足、条件注入不稳

通过后你可以做什么：
- 确认"去噪器"在物理上可用，进入更高层的校准与采样稳定性检测（关卡B）
"""

import os
import json
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ---------------------------- Utilities ----------------------------

def _flatten(x: torch.Tensor) -> torch.Tensor:
    """Flatten each sample into a vector: [B, ...] -> [B, D]."""
    return x.reshape(x.size(0), -1)


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ------------------------- Per-sample metrics -------------------------

@torch.no_grad()
def _noise_metrics_per_sample(
    x0: torch.Tensor,
    xt: torch.Tensor,
    eps_true: torch.Tensor,
    eps_hat: torch.Tensor,
    t: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    eps_num: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """
    Compute sample-wise metrics used by Gate A.
    Returns tensors of shape [B] for each metric.
    """
    # Ensure broadcast shapes
    B = x0.shape[0]
    t = t.reshape(B)
    # SNR = alpha^2 / sigma^2
    snr = (alpha.pow(2) / (sigma.pow(2) + eps_num)).reshape(B)

    # NMSE(eps) = ||eps_hat - eps||^2 / ||eps||^2
    num = _flatten(eps_hat - eps_true).pow(2).sum(dim=1)
    den = _flatten(eps_true).pow(2).sum(dim=1) + eps_num
    nmse_eps = num / den

    # Cosine similarity of eps
    cossim_eps = F.cosine_similarity(_flatten(eps_hat), _flatten(eps_true), dim=1)

    # x0_hat
    x0_hat = (xt - sigma * eps_hat) / (alpha + eps_num)
    mse_x0 = _flatten(x0_hat - x0).pow(2).mean(dim=1)

    # Diagnostics
    absmse_eps = _flatten(eps_hat - eps_true).pow(2).mean(dim=1)
    # Best scale k* minimizing ||eps_hat - k eps||^2
    dot = (_flatten(eps_hat) * _flatten(eps_true)).sum(dim=1)
    k_scale = dot / (den + eps_num)

    return {
        "t": t,
        "snr": snr,
        "nmse_eps": nmse_eps,
        "cossim_eps": cossim_eps,
        "mse_x0": mse_x0,
        "absmse_eps": absmse_eps,
        "k_scale": k_scale,
    }


# ------------------------- Binning & aggregation -------------------------

def _make_bins_by_snr(snr_all: np.ndarray, num_bins: int = 12) -> np.ndarray:
    """Return monotonically increasing bin edges using quantiles. Ensures uniqueness."""
    qs = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(snr_all, qs)
    edges = np.unique(edges)
    if edges.size < 2:
        # Fallback to min/max with two bins
        mn, mx = float(snr_all.min()), float(snr_all.max())
        edges = np.array([mn, mx], dtype=np.float64)
    return edges


def _mean_ci(values: np.ndarray) -> Tuple[float, float, int]:
    """Return (mean, 95% CI half-width, n)."""
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    if n == 0:
        return float("nan"), float("nan"), 0
    m = float(v.mean())
    # Use population std with ddof=1 if n>1
    if n > 1:
        se = v.std(ddof=1) / np.sqrt(n)
    else:
        se = 0.0
    ci = 1.96 * float(se)
    return m, ci, n


def _aggregate_by_bins(out: Dict[str, np.ndarray], edges: np.ndarray) -> List[Dict[str, Any]]:
    """Aggregate metrics in each SNR bin. Returns list of bucket dicts."""
    snr = out["snr"]
    buckets: List[Dict[str, Any]] = []
    centers = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i < len(edges) - 2:
            mask = (snr >= lo) & (snr < hi)
        else:
            mask = (snr >= lo) & (snr <= hi)
        idx = np.where(mask)[0]
        center = 0.5 * (lo + hi)
        centers.append(center)

        def agg(key: str) -> Dict[str, float]:
            mean, ci, n = _mean_ci(out[key][idx])
            return {"mean": mean, "ci95": ci, "n": int(n)}

        buckets.append({
            "snr_lo": float(lo),
            "snr_hi": float(hi),
            "snr_center": float(center),
            "nmse_eps": agg("nmse_eps"),
            "cossim_eps": agg("cossim_eps"),
            "mse_x0": agg("mse_x0"),
            # diagnostics
            "absmse_eps": agg("absmse_eps"),
            "k_scale": agg("k_scale"),
        })
    return buckets


# ------------------------- Thresholds & judgment -------------------------

# Tunable thresholds for PASS criteria
NMSE_MAIN_MAX = 0.10          # nmse upper bound in main SNR region
COS_MAIN_MIN  = 0.98          # cosine similarity lower bound in main SNR region
NMSE_TAIL_MAX = 0.20          # nmse upper bound in head/tail
COS_TAIL_MIN  = 0.95          # cosine similarity lower bound in head/tail

MAIN_COVER_RATIO = 0.60       # main region covers middle 60% volume by sample count
MAIN_PASS_RATIO  = 0.80       # at least 80% buckets in main region meet the threshold


def _judge_gateA(buckets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return pass flag and details for Gate A."""
    if len(buckets) == 0:
        return {"pass": False, "details": {"reason": "no_buckets"}}

    # Sort by snr_center
    order = np.argsort([b["snr_center"] for b in buckets]).astype(int)
    ns = np.array([b["nmse_eps"]["n"] for b in buckets], dtype=np.float64)[order]
    total = ns.sum()
    if total <= 0:
        return {"pass": False, "details": {"reason": "no_samples"}}

    cum = np.cumsum(ns) / total
    # Main region ~ middle 60% => between 0.2 and 0.8
    main_mask = (cum >= 0.2) & (cum <= 0.8)

    def series(key: str) -> np.ndarray:
        return np.array([b[key]["mean"] for b in buckets], dtype=np.float64)[order]

    nmse = series("nmse_eps")
    cosv = series("cossim_eps")

    if np.any(main_mask):
        pass_main_nmse = (nmse[main_mask] < NMSE_MAIN_MAX).mean() >= MAIN_PASS_RATIO
        pass_main_cos  = (cosv[main_mask] > COS_MAIN_MIN).mean()  >= MAIN_PASS_RATIO
        nmse_main_mean = float(np.nanmean(nmse[main_mask]))
        cos_main_mean  = float(np.nanmean(cosv[main_mask]))
    else:
        pass_main_nmse = False
        pass_main_cos  = False
        nmse_main_mean = float("nan")
        cos_main_mean  = float("nan")

    tail_mask = ~main_mask
    if np.any(tail_mask):
        nmse_tail_max = float(np.nanmax(nmse[tail_mask]))
        cos_tail_min  = float(np.nanmin(cosv[tail_mask]))
        pass_tail_nmse = nmse_tail_max < NMSE_TAIL_MAX
        pass_tail_cos  = cos_tail_min  > COS_TAIL_MIN
    else:
        nmse_tail_max = float("nan")
        cos_tail_min  = float("nan")
        pass_tail_nmse = True
        pass_tail_cos  = True

    pass_flag = bool(pass_main_nmse and pass_main_cos and pass_tail_nmse and pass_tail_cos)

    details = {
        "ok_main_nmse": bool(pass_main_nmse),
        "ok_main_cos":  bool(pass_main_cos),
        "ok_tail_nmse": bool(pass_tail_nmse),
        "ok_tail_cos":  bool(pass_tail_cos),
        "nmse_main_mean": nmse_main_mean,
        "cos_main_mean":  cos_main_mean,
        "nmse_tail_max":  nmse_tail_max,
        "cos_tail_min":   cos_tail_min,
        "main_bucket_ratio_required": MAIN_PASS_RATIO,
        "thresholds": {
            "NMSE_MAIN_MAX": NMSE_MAIN_MAX,
            "COS_MAIN_MIN":  COS_MAIN_MIN,
            "NMSE_TAIL_MAX": NMSE_TAIL_MAX,
            "COS_TAIL_MIN":  COS_TAIL_MIN,
        },
    }
    return {"pass": pass_flag, "details": details}


def log_tail_worst(buckets):
    import numpy as np
    if not buckets:
        print("[GateA] no buckets")
        return
    snr = np.array([b["snr_center"] for b in buckets], dtype=float)
    nmse = np.array([b["nmse_eps"]["mean"] for b in buckets], dtype=float)
    coss = np.array([b["cossim_eps"]["mean"] for b in buckets], dtype=float)
    k    = np.array([b["k_scale"]["mean"] for b in buckets], dtype=float)

    i_nmse = int(np.nanargmax(nmse))
    i_coss = int(np.nanargmin(coss))

    def _fmt(i):
        return f"snr={snr[i]:.4g}  nmse={nmse[i]:.3f}  cos={coss[i]:.3f}  k={k[i]:.3f}"

    print("[GateA] worst-NMSE :", _fmt(i_nmse))
    print("[GateA] worst-Cos  :", _fmt(i_coss))


def get_gateA_diagnostic_suggestions(verdict: Dict[str, Any], buckets: List[Dict[str, Any]]) -> List[str]:
    """
    根据关卡A结果生成诊断建议
    
    Args:
        verdict: _judge_gateA的返回结果
        buckets: 分桶统计结果
        
    Returns:
        诊断建议列表
    """
    suggestions = []
    details = verdict.get("details", {})
    
    if not verdict.get("pass", False):
        # 主区域问题
        if not details.get("ok_main_nmse", True):
            suggestions.append("⚠️  主区域NMSE过高 - 可能sigma_min太小或目标参数化不合适")
        if not details.get("ok_main_cos", True):
            suggestions.append("⚠️  主区域余弦相似度过低 - 噪声方向预测不准确")
        
        # 尾部问题
        if not details.get("ok_tail_nmse", True):
            suggestions.append("⚠️  尾部NMSE过高 - 强噪声下学习能力不足")
        if not details.get("ok_tail_cos", True):
            suggestions.append("⚠️  尾部余弦相似度过低 - 极值噪声处理能力差")
    
    # 分析各桶表现
    if buckets:
        snr_centers = [b["snr_center"] for b in buckets]
        nmse_values = [b["nmse_eps"]["mean"] for b in buckets]
        cos_values = [b["cossim_eps"]["mean"] for b in buckets]
        
        # 检查高SNR桶
        high_snr_indices = [i for i, snr in enumerate(snr_centers) if snr > 5.0]
        if high_snr_indices:
            high_nmse = [nmse_values[i] for i in high_snr_indices]
            if max(high_nmse) > 0.3:
                suggestions.append("⚠️  高SNR桶NMSE过高 - 建议增加高噪声训练数据")
        
        # 检查低SNR桶
        low_snr_indices = [i for i, snr in enumerate(snr_centers) if snr < 0.5]
        if low_snr_indices:
            low_cos = [cos_values[i] for i in low_snr_indices]
            if min(low_cos) < 0.9:
                suggestions.append("⚠️  低SNR桶余弦相似度低 - 检查数值精度和去噪公式")
        
        # 检查波动性
        nmse_std = np.std(nmse_values)
        if nmse_std > 0.1:
            suggestions.append("⚠️  NMSE波动较大 - 可能学习率过高或正则不足")
    
    if not suggestions:
        suggestions.append("✅ 关卡A通过！噪声匹配表现良好")
    
    return suggestions


# ------------------------- Main evaluation entry -------------------------

@torch.no_grad()
def evaluate_gateA(
    model,
    val_loader,
    scheduler,
    device: str = "cuda",
    num_bins: int = 12,
    max_batches: Optional[int] = None,
    use_autocast: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, np.ndarray], List[str]]:
    """
    Run Gate A evaluation.
    Returns:
        verdict: {"pass": bool, "details": {...}}
        buckets: list of dicts with bucketed statistics
        raw: dict of raw per-sample arrays for t, snr, and metrics
        suggestions: list of diagnostic suggestions
    """
    model.eval()

    per_batch: List[Dict[str, torch.Tensor]] = []
    n_batches = 0

    # Optional autocast (AMP) for forward speed
    autocast_ctx = torch.amp.autocast if use_autocast else _nullcontext
    with autocast_ctx('cuda' if use_autocast else None):
        for batch in val_loader:
            n_batches += 1
            if (max_batches is not None) and (n_batches > max_batches):
                break

            x0   = batch["x0"].to(device, non_blocking=True)
            cond = batch["cond"].to(device, non_blocking=True)
            B = x0.size(0)

            t = scheduler.sample_t(B, device=device)                # [B]
            alpha, sigma = scheduler.alpha_sigma(t)                 # broadcastable

            # Ensure alpha/sigma broadcast shapes match x0
            while alpha.ndim < x0.ndim:
                alpha = alpha.view(alpha.shape + (1,))
            while sigma.ndim < x0.ndim:
                sigma = sigma.view(sigma.shape + (1,))

            eps_true = torch.randn_like(x0)
            xt = alpha * x0 + sigma * eps_true

            eps_hat = model(xt, t, cond)

            stats = _noise_metrics_per_sample(
                x0=x0, xt=xt, eps_true=eps_true, eps_hat=eps_hat,
                t=t, alpha=alpha, sigma=sigma
            )
            per_batch.append(stats)

    # Concatenate per-sample metrics
    def cat_torch(key: str) -> torch.Tensor:
        return torch.cat([d[key] for d in per_batch], dim=0)

    # Move to numpy
    to_np = lambda x: cat_torch(x).detach().cpu().numpy()

    raw = {
        "t":          to_np("t"),
        "snr":        to_np("snr"),
        "nmse_eps":   to_np("nmse_eps"),
        "cossim_eps": to_np("cossim_eps"),
        "mse_x0":     to_np("mse_x0"),
        "absmse_eps": to_np("absmse_eps"),
        "k_scale":    to_np("k_scale"),
    }

    # Bucketize
    edges = _make_bins_by_snr(raw["snr"], num_bins=num_bins)
    buckets = _aggregate_by_bins(raw, edges)

    # Verdict
    verdict = _judge_gateA(buckets)
    
    # Diagnostic suggestions
    suggestions = get_gateA_diagnostic_suggestions(verdict, buckets)

    return verdict, buckets, raw, suggestions


# ------------------------- Plotting -------------------------

def plot_gateA_curves(
    buckets: List[Dict[str, Any]],
    out_png: Optional[str] = None,
    title_prefix: str = "Gate A",
) -> None:
    """Plot three separate figures for nmse_eps, cossim_eps, mse_x0 vs SNR center."""
    if len(buckets) == 0:
        return

    snr_center = np.array([b["snr_center"] for b in buckets], dtype=np.float64)

    def series(key: str) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.array([b[key]["mean"] for b in buckets], dtype=np.float64)
        ci95 = np.array([b[key]["ci95"] for b in buckets], dtype=np.float64)
        return mean, ci95

    # 1) NMSE(ε)
    nmse_mean, nmse_ci = series("nmse_eps")
    plt.figure()
    plt.title(f"{title_prefix}: NMSE(eps) vs SNR")
    plt.xlabel("SNR (bin center)")
    plt.ylabel("NMSE(eps)")
    plt.plot(snr_center, nmse_mean, marker="o")
    plt.fill_between(snr_center, nmse_mean - nmse_ci, nmse_mean + nmse_ci, alpha=0.2)
    plt.grid(True)
    if out_png:
        base, ext = os.path.splitext(out_png)
        path = f"{base}_nmse{ext or '.png'}"
        plt.savefig(path, bbox_inches="tight", dpi=200)
    else:
        plt.show()

    # 2) CosSim(ε)
    cos_mean, cos_ci = series("cossim_eps")
    plt.figure()
    plt.title(f"{title_prefix}: CosSim(eps) vs SNR")
    plt.xlabel("SNR (bin center)")
    plt.ylabel("Cosine similarity")
    plt.plot(snr_center, cos_mean, marker="o")
    plt.fill_between(snr_center, cos_mean - cos_ci, cos_mean + cos_ci, alpha=0.2)
    plt.grid(True)
    if out_png:
        base, ext = os.path.splitext(out_png)
        path = f"{base}_cossim{ext or '.png'}"
        plt.savefig(path, bbox_inches="tight", dpi=200)
    else:
        plt.show()

    # 3) MSE(x0_hat)
    x0_mean, x0_ci = series("mse_x0")
    plt.figure()
    plt.title(f"{title_prefix}: MSE(x0_hat) vs SNR")
    plt.xlabel("SNR (bin center)")
    plt.ylabel("MSE(x0_hat)")
    plt.plot(snr_center, x0_mean, marker="o")
    plt.fill_between(snr_center, x0_mean - x0_ci, x0_mean + x0_ci, alpha=0.2)
    plt.grid(True)
    if out_png:
        base, ext = os.path.splitext(out_png)
        path = f"{base}_mse_x0{ext or '.png'}"
        plt.savefig(path, bbox_inches="tight", dpi=200)
    else:
        plt.show()


# ------------------------- Context manager -------------------------

class _nullcontext:
    """Minimal null context manager for optional autocast-like usage."""
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc, tb):
        return False
