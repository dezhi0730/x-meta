import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Tuple


# ------------------------------------------------------------------
# 0. 辅助函数
# ------------------------------------------------------------------
def _world_gather(x: torch.Tensor) -> torch.Tensor:
    """
    多卡场景下：把各卡 batch 对齐到全局最大，再 concat。
    单卡 / 未初始化 dist 时直接返回 x。
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return x

    world = dist.get_world_size()
    local_bsz = x.size(0)

    # 全局最大 batch
    max_bsz = torch.tensor([local_bsz], device=x.device)
    dist.all_reduce(max_bsz, dist.ReduceOp.MAX)
    max_bsz = int(max_bsz.item())

    # pad 到相同长度
    if local_bsz < max_bsz:
        pad_shape = (max_bsz - local_bsz, *x.shape[1:])
        pad = torch.zeros(pad_shape, device=x.device, dtype=x.dtype)
        x = torch.cat([x, pad], 0)

    # all_gather
    bufs = [torch.zeros_like(x) for _ in range(world)]
    dist.all_gather(bufs, x)

    # 去除 pad
    return torch.cat([b[:local_bsz] for b in bufs], 0)


def _safe_zero_like(*refs: torch.Tensor) -> torch.Tensor:
    """
    生成与 refs 计算图连通的 “0” 标量，用于跳过有问题的 batch
    仍可向后传播，避免 GradScaler 断言。
    """
    z = torch.zeros((), device=refs[0].device, dtype=refs[0].dtype)
    for r in refs:
        z = z + r.sum() * 0.0
    return z


def _variance_reg(z: torch.Tensor,
                  gamma: float,
                  weight: float,
                  eps: float = 1e-4) -> torch.Tensor:
    """
    VicReg-style variance regularization
    """
    std = z.std(0, unbiased=False)                     # (D,)
    valid = (std > eps) & torch.isfinite(std)
    if not valid.any():
        return _safe_zero_like(z)
    return weight * F.relu(gamma - std[valid]).mean()


# ------------------------------------------------------------------
# 1. 主损失类
# ------------------------------------------------------------------
class SparseCLIPLoss(nn.Module):
    """
    loss = ½·[InfoNCE(tree→dna) + InfoNCE(dna→tree)]  +  VarReg
    可学习参数：zero_weight ∈ (0,1) 控制对 zero‑ratio=1 样本的权重衰减。
    """
    def __init__(self,
                 *,
                 local_loss : bool = True,   # 单卡或者想省显存时 True
                 init_zero_w: float = 0.5,   # zero_weight 初值
                 var_gamma  : float = 1.0,
                 var_weight : float = 1e-2,
                 normalize  : bool = True):
        super().__init__()
        self.local_loss  = local_loss
        self.var_gamma   = var_gamma
        self.var_weight  = var_weight
        self.normalize   = normalize

        # ---- 可学习 zero_weight ----
        # raw 参数 ∈ ℝ, 经 sigmoid → (0,1)
        raw_init = -math.log(1 / init_zero_w - 1)      # sigmoid⁻¹
        self._raw_zero_w = nn.Parameter(torch.tensor([raw_init], dtype=torch.float32))

    # 公开属性：当前 zero_weight (张量)
    @property
    def zero_weight(self) -> torch.Tensor:
        return torch.sigmoid(self._raw_zero_w)          # (0,1)

    # --------------------------------------------------------------
    def forward(self, out: dict) -> torch.Tensor:
        """
        out 需包含：
            tree_emb   (B,D)
            dna_emb    (B,D)
            logit_scale (scalar tensor)
            zero_ratio  (B,)    # 0~1, 若不存在视为 0
        """
        z_t = out["tree_emb"]          # (B,D)
        z_d = out["dna_emb"]           # (B,D)
        # logit_scale = out["logit_scale"].clamp_(0, math.log(100.0))
        LOGIT_MAX = math.log(20.0)         
        logit_scale_raw = out["logit_scale"]  
        logit_scale = logit_scale_raw.clamp(0.0, LOGIT_MAX)


        assert torch.isfinite(z_t).all(), "[NaN] tree_emb (input to loss)"
        assert torch.isfinite(z_d).all(), "[NaN] dna_emb  (input to loss)"

        if self.normalize:
            z_t = F.normalize(z_t, dim=-1, eps=1e-6)
            z_d = F.normalize(z_d, dim=-1, eps=1e-6)

            # ---------- B. normalize 后 ----------
            assert torch.isfinite(z_t).all(), "[NaN] tree_emb after normalize"
            assert torch.isfinite(z_d).all(), "[NaN] dna_emb  after normalize"


        # 遇到 NaN/Inf 直接跳过 batch
        if (not torch.isfinite(z_t).all()) or (not torch.isfinite(z_d).all()):
            print("[WARN] embeddings contain NaN/Inf — skipping loss")
            return _safe_zero_like(z_t, z_d, logit_scale)

        B = z_t.size(0)
        zr = torch.as_tensor(out.get("zero_ratio", 0.0),
                             device=z_t.device).flatten()
        if zr.numel() == 1:            # 标量广播
            zr = zr.repeat(B)

        # gather（若 local_loss=False）
        if self.local_loss:
            z_d_g, zr_g = z_d, zr
        else:
            z_d_g = _world_gather(z_d)
            zr_g  = _world_gather(zr.unsqueeze(1)).flatten()

        row_mask = zr   < 1.0
        col_mask = zr_g < 1.0
        if not row_mask.any():         # 全是空样本
            return _safe_zero_like(z_t)

        # -------- ① 对称 InfoNCE --------
        sim = z_t @ z_d_g.T
        assert torch.isfinite(sim).all(), "[NaN] sim = z_t @ z_dᵀ"   # ← C.可选第三断言
        sim = sim[row_mask][:, col_mask]                    # (K,K)
        sim = torch.nan_to_num(sim, nan=0.0, posinf=5.0, neginf=-5.0)

        logits = sim * logit_scale.exp()
        K = logits.size(0)
        labels = torch.arange(K, device=logits.device)

        # learnable weight
        w = 1 - zr[row_mask] * (1 - self.zero_weight)       # (K,)
        loss_i2t = (F.cross_entropy(logits,
                                     labels,
                                     reduction='none') * w).mean()
        loss_t2i = (F.cross_entropy(logits.t(),
                                     labels,
                                     reduction='none') * w).mean()
        clip_loss = 0.5 * (loss_i2t + loss_t2i)

        # -------- ② 方差正则 --------
        var_loss = _variance_reg(z_t, self.var_gamma, self.var_weight) + \
                   _variance_reg(z_d, self.var_gamma, self.var_weight)

        self.last_clip_loss = clip_loss          # ★ 新增
        self.last_var_loss  = var_loss

        return clip_loss + var_loss


# ------------------------------------------------------------------
# 2. 工具：创建优化器时把 loss 参数加入
# ------------------------------------------------------------------
def add_loss_params_to_optimizer(model: nn.Module,
                                 criterion: nn.Module,
                                 base_lr: float,
                                 weight_decay: float = 0.0,
                                 logit_lr_factor: float = 0.05):
    other_params   = []
    logit_params   = []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if n.endswith("logit_scale"):
            logit_params.append(p)
        else:
            other_params.append(p)

    loss_params = [p for p in criterion.parameters() if p.requires_grad]

    param_groups = [
        {"params": other_params,        "lr": base_lr},
        {"params": logit_params,        "lr": base_lr * logit_lr_factor},
        {"params": loss_params,         "lr": base_lr},
    ]
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)