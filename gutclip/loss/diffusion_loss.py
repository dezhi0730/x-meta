import torch
import torch.nn.functional as F

def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    """统一到 (B, N)。高维压平成特征维。"""
    if x.dim() == 0:
        return x.reshape(1, 1)
    if x.dim() == 1:
        return x.unsqueeze(0)
    if x.dim() == 2:
        return x
    B = x.size(0)
    return x.reshape(B, -1)

def _spearman_loss(pred: torch.Tensor,
                   target: torch.Tensor,
                   eps: float = 1e-6) -> torch.Tensor:
    """
    平均样本的 1-ρ，ρ 在最后一维上计算秩相关。
    采用几何形式，避免“无偏/有偏”因子不一致导致 |ρ|>1。
    """
    pred   = _ensure_2d(pred).float()
    target = _ensure_2d(target).float()

    def _rank(x: torch.Tensor) -> torch.Tensor:
        # x: (B, N)
        _, idx = torch.sort(x, dim=-1)
        rank = torch.zeros_like(x, dtype=torch.float32)
        arange = torch.arange(x.size(-1), device=x.device, dtype=torch.float32)
        rank.scatter_(dim=-1, index=idx, src=arange.expand_as(x))
        return rank

    rp = _rank(pred)
    rt = _rank(target)

    rp_c = rp - rp.mean(dim=-1, keepdim=True)
    rt_c = rt - rt.mean(dim=-1, keepdim=True)

    num = (rp_c * rt_c).sum(dim=-1)                    # (B,)
    sx  = (rp_c * rp_c).sum(dim=-1) + eps
    sy  = (rt_c * rt_c).sum(dim=-1) + eps
    rho = num / torch.sqrt(sx * sy)

    # 数值稳健：限制到 [-1, 1]
    rho = torch.clamp(rho, -1.0, 1.0)
    return 1.0 - rho.mean()

class DiffusionLoss:
    """
    L = MSE(ε̂, ε) + λ_rank * (1 - ρ(lhs, y_true))
    推荐 lhs = y0_pred（从 ε̂ 反推的去噪目标）；若未提供则回退到 y_prior。
    """
    def __init__(self, lambda_rank: float = 0.0):
        self.lambda_rank = float(lambda_rank)

    def __call__(self,
                 eps_pred:  torch.Tensor,   # (B,N)
                 eps_true:  torch.Tensor,   # (B,N)
                 y_prior:   torch.Tensor,   # (B,N)
                 y_true:    torch.Tensor,   # (B,N)
                 y0_pred:   torch.Tensor = None) -> dict:
        mse = F.mse_loss(eps_pred, eps_true)

        rank = eps_pred.new_tensor(0.0)
        if self.lambda_rank > 0:
            lhs  = y0_pred if y0_pred is not None else y_prior
            rank = _spearman_loss(lhs, y_true)
        total = mse + self.lambda_rank * rank
        return {"total": total, "mse": mse, "rank": rank}