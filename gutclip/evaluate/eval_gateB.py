"""
关卡B：残差校准与采样稳定性评估 (Gate B: Residual Calibration & Sampling Stability)

这是扩散模型质量检测的第二关，专注于验证：
1. 标准化残差 r = (ε̂ - ε) / σ_t 的均值和方差是否正确
2. DDIM轨迹是否数值稳定（范数不爆不塌）

核心概念：
- 不仅要平均正确，还要方差标定正确
- 采样是一个多步反馈过程，系统性偏差会在多步传播中放大
- 条件指导/值函数会进一步放大不稳定性

失败意味着：
- 均值偏移：模型系统性高估/低估噪声，采样时会"漂移"
- 方差>1 或厚尾：偶发大误差，会在多步过程中积累成崩溃
- DDIM 轨迹发散：通常是调度范围、FiLM 放大、学习率/权重衰减设置不当

通过后你可以做什么：
- 说明模型在数值上可稳定采样，可以开始问：它是否真的用到了条件，能做任务？（进入关卡 C）
"""

import os
import json
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm


# ---------------------------- Utilities ----------------------------

def _flatten(x: torch.Tensor) -> torch.Tensor:
    """Flatten each sample into a vector: [B, ...] -> [B, D]."""
    return x.reshape(x.size(0), -1)


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ------------------------- Residual Calibration Metrics -------------------------

@torch.no_grad()
def _compute_standardized_residuals(
    eps_true: torch.Tensor,
    eps_hat: torch.Tensor,
    sigma: torch.Tensor,
    eps_num: float = 1e-12,
) -> torch.Tensor:
    """
    Compute standardized residuals: r = (ε̂ - ε) / σ_t
    
    Args:
        eps_true: True noise [B, ...]
        eps_hat: Predicted noise [B, ...]
        sigma: Noise schedule σ_t [B] or [B, 1, ...]
        eps_num: Small constant for numerical stability
    
    Returns:
        Standardized residuals [B, ...]
    """
    # Ensure sigma has proper shape for broadcasting
    if sigma.dim() == 1:
        sigma = sigma.view(-1, 1)
    
    # Compute residuals
    residuals = eps_hat - eps_true
    
    # Standardize by sigma
    sigma_flat = sigma.expand_as(residuals)
    standardized_residuals = residuals / (sigma_flat + eps_num)
    
    return standardized_residuals


@torch.no_grad()
def _residual_calibration_metrics(
    eps_true: torch.Tensor,
    eps_hat: torch.Tensor,
    sigma: torch.Tensor,
    t: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute residual calibration metrics per sample.
    
    Returns:
        Dictionary with metrics for each sample [B]
    """
    B = eps_true.shape[0]
    t = t.reshape(B)
    
    # Standardized residuals
    r = _compute_standardized_residuals(eps_true, eps_hat, sigma)
    
    # Flatten for statistics
    r_flat = _flatten(r)
    
    # Mean of standardized residuals (should be ≈ 0)
    mean_r = r_flat.mean(dim=1)
    
    # Variance of standardized residuals (should be ≈ 1)
    var_r = r_flat.var(dim=1)
    
    # Skewness (should be ≈ 0 for normal distribution)
    skew_r = torch.zeros(B, device=r.device)
    for i in range(B):
        if r_flat[i].numel() > 0:
            skew_r[i] = torch.mean(((r_flat[i] - mean_r[i]) / (torch.sqrt(var_r[i]) + 1e-8)) ** 3)
    
    # Kurtosis (should be ≈ 3 for normal distribution)
    kurt_r = torch.zeros(B, device=r.device)
    for i in range(B):
        if r_flat[i].numel() > 0:
            kurt_r[i] = torch.mean(((r_flat[i] - mean_r[i]) / (torch.sqrt(var_r[i]) + 1e-8)) ** 4)
    
    # Maximum absolute residual (for outlier detection)
    max_abs_r = r_flat.abs().max(dim=1)[0]
    
    return {
        "t": t,
        "mean_r": mean_r,
        "var_r": var_r,
        "skew_r": skew_r,
        "kurt_r": kurt_r,
        "max_abs_r": max_abs_r,
    }


# ------------------------- DDIM Trajectory Stability -------------------------

@torch.no_grad()
def _ddim_trajectory_stability(
    model,
    scheduler,
    x0: torch.Tensor,
    cond_vec: torch.Tensor,
    device: str = "cuda",
    num_steps: int = 50,
    use_autocast: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Test DDIM trajectory stability by running reverse process on ground truth x0.
    
    Args:
        model: Diffusion model
        scheduler: DDIM scheduler
        x0: Ground truth samples [B, ...]
        cond_vec: Conditioning vectors [B, ...]
        device: Device to run on
        num_steps: Number of DDIM steps
        use_autocast: Whether to use automatic mixed precision
    
    Returns:
        Dictionary with trajectory metrics
    """
    B = x0.shape[0]
    
    # Set up scheduler
    scheduler.set_timesteps(num_steps, device=device)
    timesteps = scheduler.timesteps
    
    # Initialize trajectory tracking
    trajectory_norms = []
    trajectory_changes = []
    noise_pred_norms = []
    
    # Start from x0 and go forward
    xt = x0.clone()
    
    autocast_context = torch.cuda.amp.autocast() if use_autocast else _nullcontext()
    
    for i, t in enumerate(timesteps):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        
        with autocast_context:
            # Predict noise
            noise_pred = model(xt, t_tensor, cond_vec)
        
        # Record norms
        noise_pred_norm = torch.norm(_flatten(noise_pred), dim=1)
        noise_pred_norms.append(noise_pred_norm)
        
        # Step forward - HuggingFace schedulers return dict with "prev_sample"
        step_output = scheduler.step(noise_pred, t, xt)
        xt_prev = step_output["prev_sample"]
        
        # Record trajectory metrics
        xt_norm = torch.norm(_flatten(xt), dim=1)
        trajectory_norms.append(xt_norm)
        
        if i > 0:
            change = torch.norm(_flatten(xt - xt_prev), dim=1)
            trajectory_changes.append(change)
    
    return {
        "trajectory_norms": torch.stack(trajectory_norms, dim=1),  # [B, num_steps]
        "trajectory_changes": torch.stack(trajectory_changes, dim=1),  # [B, num_steps-1]
        "noise_pred_norms": torch.stack(noise_pred_norms, dim=1),  # [B, num_steps]
    }


# ------------------------- Statistical Tests -------------------------

def _test_residual_calibration(
    mean_r: np.ndarray,
    var_r: np.ndarray,
    skew_r: np.ndarray,
    kurt_r: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Perform statistical tests for residual calibration.
    
    Args:
        mean_r: Mean of standardized residuals
        var_r: Variance of standardized residuals
        skew_r: Skewness of standardized residuals
        kurt_r: Kurtosis of standardized residuals
        alpha: Significance level
    
    Returns:
        Dictionary with test results
    """
    results = {}
    
    # Test 1: Mean should be ≈ 0
    mean_test = stats.ttest_1samp(mean_r, 0.0)
    results["mean_test"] = {
        "statistic": mean_test.statistic,
        "pvalue": mean_test.pvalue,
        "significant": mean_test.pvalue < alpha,
        "mean_abs": np.abs(mean_r).mean(),
        "mean_std": mean_r.std(),
    }
    
    # Test 2: Variance should be ≈ 1
    var_test = stats.ttest_1samp(var_r, 1.0)
    results["var_test"] = {
        "statistic": var_test.statistic,
        "pvalue": var_test.pvalue,
        "significant": var_test.pvalue < alpha,
        "var_mean": var_r.mean(),
        "var_std": var_r.std(),
    }
    
    # Test 3: Skewness should be ≈ 0 (for normal distribution)
    skew_test = stats.ttest_1samp(skew_r, 0.0)
    results["skew_test"] = {
        "statistic": skew_test.statistic,
        "pvalue": skew_test.pvalue,
        "significant": skew_test.pvalue < alpha,
        "skew_mean": skew_r.mean(),
        "skew_std": skew_r.std(),
    }
    
    # Test 4: Kurtosis should be ≈ 3 (for normal distribution)
    kurt_test = stats.ttest_1samp(kurt_r, 3.0)
    results["kurt_test"] = {
        "statistic": kurt_test.statistic,
        "pvalue": kurt_test.pvalue,
        "significant": kurt_test.pvalue < alpha,
        "kurt_mean": kurt_r.mean(),
        "kurt_std": kurt_r.std(),
    }
    
    return results


def _test_trajectory_stability(
    trajectory_norms: np.ndarray,
    trajectory_changes: np.ndarray,
    noise_pred_norms: np.ndarray,
) -> Dict[str, Any]:
    """
    Test DDIM trajectory stability.
    
    Args:
        trajectory_norms: Norms of trajectory points [B, num_steps]
        trajectory_changes: Changes between steps [B, num_steps-1]
        noise_pred_norms: Norms of predicted noise [B, num_steps]
    
    Returns:
        Dictionary with stability metrics
    """
    results = {}
    
    # Check for explosion (norms growing too fast)
    norm_growth = trajectory_norms[:, 1:] / (trajectory_norms[:, :-1] + 1e-8)
    explosion_threshold = 10.0  # Norm should not grow by more than 10x
    explosion_ratio = (norm_growth > explosion_threshold).mean()
    results["explosion_ratio"] = explosion_ratio
    
    # Check for collapse (norms shrinking too fast)
    collapse_threshold = 0.1  # Norm should not shrink by more than 90%
    collapse_ratio = (norm_growth < collapse_threshold).mean()
    results["collapse_ratio"] = collapse_ratio
    
    # Check for excessive changes between steps
    change_threshold = 5.0  # Relative change should not exceed 5x
    excessive_changes = (trajectory_changes > change_threshold).mean()
    results["excessive_changes"] = excessive_changes
    
    # Check noise prediction stability
    noise_norm_mean = noise_pred_norms.mean()
    noise_norm_std = noise_pred_norms.std()
    results["noise_norm_stats"] = {
        "mean": noise_norm_mean,
        "std": noise_norm_std,
        "cv": noise_norm_std / (noise_norm_mean + 1e-8),  # Coefficient of variation
    }
    
    return results


# ------------------------- Judgment Logic -------------------------

def _judge_gateB(
    residual_tests: Dict[str, Any],
    stability_tests: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Judge whether Gate B is passed based on test results.
    
    Args:
        residual_tests: Results from residual calibration tests
        stability_tests: Results from trajectory stability tests
    
    Returns:
        Dictionary with judgment results
    """
    verdict = {
        "passed": True,
        "issues": [],
        "scores": {},
    }
    
    # Residual calibration checks
    mean_issue = residual_tests["mean_test"]["mean_abs"] > 0.1
    var_issue = abs(residual_tests["var_test"]["var_mean"] - 1.0) > 0.2
    skew_issue = abs(residual_tests["skew_test"]["skew_mean"]) > 0.5
    kurt_issue = abs(residual_tests["kurt_test"]["kurt_mean"] - 3.0) > 1.0
    
    if mean_issue:
        verdict["passed"] = False
        verdict["issues"].append("Residual mean significantly different from 0")
    
    if var_issue:
        verdict["passed"] = False
        verdict["issues"].append("Residual variance significantly different from 1")
    
    if skew_issue:
        verdict["issues"].append("Residual distribution is skewed")
    
    if kurt_issue:
        verdict["issues"].append("Residual distribution has non-normal kurtosis")
    
    # Trajectory stability checks
    explosion_issue = stability_tests["explosion_ratio"] > 0.01
    collapse_issue = stability_tests["collapse_ratio"] > 0.01
    change_issue = stability_tests["excessive_changes"] > 0.05
    
    if explosion_issue:
        verdict["passed"] = False
        verdict["issues"].append("Trajectory norms exploding")
    
    if collapse_issue:
        verdict["passed"] = False
        verdict["issues"].append("Trajectory norms collapsing")
    
    if change_issue:
        verdict["issues"].append("Excessive changes between trajectory steps")
    
    # Assign scores
    verdict["scores"] = {
        "residual_calibration": 1.0 - (mean_issue + var_issue) * 0.5,
        "trajectory_stability": 1.0 - (explosion_issue + collapse_issue) * 0.5,
        "overall": 1.0 - len(verdict["issues"]) * 0.1,
    }
    
    return verdict


# ------------------------- Main Evaluation Function -------------------------

@torch.no_grad()
def evaluate_gateB(
    model,
    val_loader,
    scheduler,
    device: str = "cuda",
    max_batches: Optional[int] = None,
    use_autocast: bool = False,
    num_ddim_steps: int = 50,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], List[str]]:
    """
    Evaluate Gate B: Residual Calibration & Sampling Stability.
    
    Args:
        model: Trained diffusion model
        val_loader: Validation data loader
        scheduler: DDIM scheduler
        device: Device to run on
        max_batches: Maximum number of batches to evaluate
        use_autocast: Whether to use automatic mixed precision
        num_ddim_steps: Number of DDIM steps for trajectory test
    
    Returns:
        Tuple of (verdict, raw_metrics, suggestions)
    """
    model.eval()
    
    # Collect metrics
    all_metrics = []
    all_trajectory_metrics = []
    
    autocast_context = torch.cuda.amp.autocast() if use_autocast else _nullcontext()
    
    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating Gate B")):
        if max_batches and batch_idx >= max_batches:
            break
        
        # Extract data
        x0 = batch["x0"].to(device)
        xt = batch["xt"].to(device)
        eps_true = batch["eps"].to(device)
        t = batch["t"].to(device)
        alpha = batch["alpha"].to(device)
        sigma = batch["sigma"].to(device)
        cond_vec = batch["cond_vec"].to(device)
        
        with autocast_context:
            # Predict noise
            eps_hat = model(xt, t, cond_vec)
        
        # Compute residual calibration metrics
        residual_metrics = _residual_calibration_metrics(
            eps_true, eps_hat, sigma, t
        )
        all_metrics.append(residual_metrics)
        
        # Test DDIM trajectory stability (on a subset to save time)
        if batch_idx < 5:  # Only test on first 5 batches
            trajectory_metrics = _ddim_trajectory_stability(
                model, scheduler, x0, cond_vec, device, num_ddim_steps, use_autocast
            )
            all_trajectory_metrics.append(trajectory_metrics)
    
    # Aggregate metrics
    def cat_torch(key: str) -> torch.Tensor:
        return torch.cat([m[key] for m in all_metrics], dim=0)
    
    raw_metrics = {
        "mean_r": cat_torch("mean_r").cpu().numpy(),
        "var_r": cat_torch("var_r").cpu().numpy(),
        "skew_r": cat_torch("skew_r").cpu().numpy(),
        "kurt_r": cat_torch("kurt_r").cpu().numpy(),
        "max_abs_r": cat_torch("max_abs_r").cpu().numpy(),
    }
    
    # Aggregate trajectory metrics
    if all_trajectory_metrics:
        trajectory_norms = torch.cat([m["trajectory_norms"] for m in all_trajectory_metrics], dim=0).cpu().numpy()
        trajectory_changes = torch.cat([m["trajectory_changes"] for m in all_trajectory_metrics], dim=0).cpu().numpy()
        noise_pred_norms = torch.cat([m["noise_pred_norms"] for m in all_trajectory_metrics], dim=0).cpu().numpy()
    else:
        trajectory_norms = np.array([])
        trajectory_changes = np.array([])
        noise_pred_norms = np.array([])
    
    # Perform statistical tests
    residual_tests = _test_residual_calibration(
        raw_metrics["mean_r"],
        raw_metrics["var_r"],
        raw_metrics["skew_r"],
        raw_metrics["kurt_r"],
    )
    
    stability_tests = _test_trajectory_stability(
        trajectory_norms,
        trajectory_changes,
        noise_pred_norms,
    )
    
    # Judge results
    verdict = _judge_gateB(residual_tests, stability_tests)
    
    # Generate suggestions
    suggestions = get_gateB_diagnostic_suggestions(verdict, residual_tests, stability_tests)
    
    return verdict, raw_metrics, suggestions


def get_gateB_diagnostic_suggestions(
    verdict: Dict[str, Any],
    residual_tests: Dict[str, Any],
    stability_tests: Dict[str, Any],
) -> List[str]:
    """
    Generate diagnostic suggestions based on Gate B results.
    """
    suggestions = []
    
    if not verdict["passed"]:
        suggestions.append("❌ Gate B FAILED - Model needs calibration before proceeding")
    
    # Residual calibration suggestions
    mean_abs = residual_tests["mean_test"]["mean_abs"]
    var_mean = residual_tests["var_test"]["var_mean"]
    
    if mean_abs > 0.1:
        suggestions.append(f"⚠️  Residual mean too high ({mean_abs:.3f}) - check model bias")
    
    if abs(var_mean - 1.0) > 0.2:
        suggestions.append(f"⚠️  Residual variance off ({var_mean:.3f}) - check noise schedule")
    
    # Trajectory stability suggestions
    explosion_ratio = stability_tests["explosion_ratio"]
    collapse_ratio = stability_tests["collapse_ratio"]
    
    if explosion_ratio > 0.01:
        suggestions.append(f"⚠️  Trajectory explosion detected ({explosion_ratio:.1%}) - check learning rate")
    
    if collapse_ratio > 0.01:
        suggestions.append(f"⚠️  Trajectory collapse detected ({collapse_ratio:.1%}) - check scheduler")
    
    if verdict["passed"]:
        suggestions.append("✅ Gate B PASSED - Model is numerically stable for sampling")
        suggestions.append("🎯 Ready to test conditional generation (Gate C)")
    
    return suggestions


# ------------------------- Plotting -------------------------

def plot_gateB_results(
    raw_metrics: Dict[str, np.ndarray],
    out_png: Optional[str] = None,
    title_prefix: str = "Gate B",
) -> None:
    """
    Plot Gate B evaluation results.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"{title_prefix}: Residual Calibration & Sampling Stability", fontsize=16)
    
    # Residual distribution
    axes[0, 0].hist(raw_metrics["mean_r"], bins=50, alpha=0.7, density=True)
    axes[0, 0].axvline(0, color='red', linestyle='--', label='Target: 0')
    axes[0, 0].set_title("Residual Mean Distribution")
    axes[0, 0].set_xlabel("Mean of Standardized Residuals")
    axes[0, 0].legend()
    
    # Residual variance
    axes[0, 1].hist(raw_metrics["var_r"], bins=50, alpha=0.7, density=True)
    axes[0, 1].axvline(1, color='red', linestyle='--', label='Target: 1')
    axes[0, 1].set_title("Residual Variance Distribution")
    axes[0, 1].set_xlabel("Variance of Standardized Residuals")
    axes[0, 1].legend()
    
    # Residual skewness
    axes[0, 2].hist(raw_metrics["skew_r"], bins=50, alpha=0.7, density=True)
    axes[0, 2].axvline(0, color='red', linestyle='--', label='Target: 0')
    axes[0, 2].set_title("Residual Skewness Distribution")
    axes[0, 2].set_xlabel("Skewness of Standardized Residuals")
    axes[0, 2].legend()
    
    # Residual kurtosis
    axes[1, 0].hist(raw_metrics["kurt_r"], bins=50, alpha=0.7, density=True)
    axes[1, 0].axvline(3, color='red', linestyle='--', label='Target: 3')
    axes[1, 0].set_title("Residual Kurtosis Distribution")
    axes[1, 0].set_xlabel("Kurtosis of Standardized Residuals")
    axes[1, 0].legend()
    
    # Max absolute residual
    axes[1, 1].hist(raw_metrics["max_abs_r"], bins=50, alpha=0.7, density=True)
    axes[1, 1].set_title("Max Absolute Residual Distribution")
    axes[1, 1].set_xlabel("Max |Standardized Residual|")
    
    # Q-Q plot for normality test
    from scipy import stats
    standardized_residuals = (raw_metrics["mean_r"] - raw_metrics["mean_r"].mean()) / raw_metrics["mean_r"].std()
    stats.probplot(standardized_residuals, dist="norm", plot=axes[1, 2])
    axes[1, 2].set_title("Q-Q Plot (Normality Test)")
    
    plt.tight_layout()
    
    if out_png:
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        print(f"Gate B plots saved to {out_png}")
    
    plt.show()


# ------------------------- Context Manager -------------------------

class _nullcontext:
    def __enter__(self):
        return None
    
    def __exit__(self, exc_type, exc, tb):
        return False


# ------------------------- Main Execution -------------------------

if __name__ == "__main__":
    # Example usage
    print("Gate B: Residual Calibration & Sampling Stability")
    print("This script should be imported and used with your model and data loader.") 