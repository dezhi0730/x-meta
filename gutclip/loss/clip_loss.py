import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# ----------------------- 工具 -----------------------
def _world_gather(t: torch.Tensor) -> torch.Tensor:
    """DDP 下把各卡 tensor 拼在一起；单卡时直接返回"""
    if dist.is_initialized() and dist.get_world_size() > 1:
        bufs = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(bufs, t)
        t = torch.cat(bufs, 0)
    return t


def _variance_reg(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """CLIP 官方 VIT-B/32 的做法——压低协方差 collapse"""
    std = z.std(dim=0) + 1e-4
    return F.relu(gamma - std).mean()


# ======================= 主损失 =======================
class CLIPLoss(nn.Module):
    """
    Ⅱ. 对称 InfoNCE   +   Ⅲ. feature variance 正则
    tips:
      • 支持 out_dict 里手动传入 'manual_sim'（例如加 margin 时）
      • τ 由模型传进来；内部再做一次 clip，确保 exp(τ) 不炸
    """
    def __init__(self,
                 local_loss: bool = False,
                 gamma: float = 1.0,
                 var_weight: float = 0.05):            # ← 记得把权重降一点
        super().__init__()
        self.local_loss = local_loss
        self.gamma      = gamma
        self.var_w      = var_weight

    # ------------------- 前向 -------------------
    def forward(self, out: dict):
        z_tree, z_dna = out["tree_emb"], out["dna_emb"]
        tau           = out["logit_scale"].clamp(0, 4.6052)   # ln(100)

        # ① 相似度矩阵
        sim = out.get("manual_sim")         # 给 MarginCLIP 用
        if sim is None:
            if self.local_loss:             # 单机 / 本地负样本
                sim = z_tree @ z_dna.T
                labels = torch.arange(sim.size(0), device=sim.device)
            else:                           # 多机：all-gather 全局负样本
                sim   = z_tree @ _world_gather(z_dna).T
                rank  = dist.get_rank() if dist.is_initialized() else 0
                bs    = z_tree.size(0)
                labels = torch.arange(bs, device=sim.device) + rank * bs
        else:
            labels = torch.arange(sim.size(0), device=sim.device)   # 已经是 (B,B)

        # ② InfoNCE
        logits = sim * tau.exp()
        loss_i2t = F.cross_entropy(logits,     labels)
        loss_t2i = F.cross_entropy(logits.T,   labels)
        clip_loss = 0.5 * (loss_i2t + loss_t2i)

        # ③ variance regularization
        var_loss = self.var_w * (_variance_reg(z_tree, self.gamma) +
                                 _variance_reg(z_dna , self.gamma))

        return clip_loss + var_loss


# ======================= 带 Margin 的包装器 =======================
class MarginCLIPLoss(nn.Module):
    """
    additive margin：对角线 sim 减 m，再丢给上面的 CLIPLoss
    """
    def __init__(self,
                 margin: float = 0.1,
                 local_loss: bool = False,
                 gamma: float = 1.0):
        super().__init__()
        self.margin   = margin
        self.clip_loss = CLIPLoss(local_loss=local_loss, gamma=gamma)

    def forward(self, out: dict):
        sim = out["tree_emb"] @ out["dna_emb"].T
        sim = sim - torch.eye(sim.size(0), device=sim.device) * self.margin
        out = {**out, "manual_sim": sim}      # 塞回去
        return self.clip_loss(out)