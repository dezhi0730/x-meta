# gutclip/loss/clip_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

__all__ = ["CLIPLoss"]

class CLIPLoss(nn.Module):
    """
    InfoNCE / CLIP style loss with learnable logit_scale.
    model.forward() 需返回 dict:
        {'tree_emb': (B, D), 'dna_emb': (B, D), 'logit_scale': scalar tensor}
    """
    def __init__(self, local_loss=False, gamma=1.0):
        super().__init__()
        self.local_loss = local_loss        # True: 仅同卡配对；False: 全局配对
        self.gamma = gamma  # 方差阈值，提高到 1.0

    @staticmethod
    def _gather_with_grad(tensor):
        world = dist.get_world_size()
        if world == 1:
            return tensor
        tensors = [torch.zeros_like(tensor) for _ in range(world)]
        dist.all_gather(tensors, tensor)
        return torch.cat(tensors, 0)

    def _variance_loss(self, z):
        """计算方差正则化损失
        Args:
            z: (B, D) 已 L2-norm 的特征
        Returns:
            float: 方差惩罚项
        """
        std = z.std(dim=0) + 1e-4          # (D,)
        penalty = F.relu(self.gamma - std).mean()
        return penalty

    def forward(self, output):
        """
        Args:
            output: dict containing:
                - tree_emb: (B, D) 树结构特征
                - dna_emb: (B, D) DNA特征
                - logit_scale: raw parameter (not exp'd)
        """
        # 1. 特征已经归一化，直接使用
        tree_features = output["tree_emb"]
        dna_features = output["dna_emb"]
        
        # 2. 计算相似度矩阵
        logits = torch.matmul(tree_features, dna_features.t())  # (N, N)
        
        # 3. 应用 logit_scale（只 exp 一次）
        logit_scale = output["logit_scale"].clamp(-5, 5)  # 限制 log 温度范围
        logits = logits * logit_scale.exp()  # 只 exp 一次
        
        # 4. 生成标签
        N = tree_features.size(0)
        if self.local_loss:
            labels = torch.arange(N, device=tree_features.device)
        else:
            # 确保标签在分布式训练中是连续的
            if dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                labels = torch.arange(N, device=tree_features.device) + rank * N
            else:
                labels = torch.arange(N, device=tree_features.device)
        
        # 5. 计算损失
        loss = (F.cross_entropy(logits, labels) + 
                F.cross_entropy(logits.t(), labels)) / 2

        # 6. 打印详细的调试信息
        with torch.no_grad():
            # 计算归一化后的相似度
            sim = tree_features @ dna_features.t()   # 这两已 L2-norm
            diag = sim.diag().mean().item()
            off = (sim.sum() - sim.diag().sum()) / (sim.numel() - sim.size(0))
            
            # 计算特征统计量
            tree_std = tree_features.std(dim=0).mean().item()
            dna_std = dna_features.std(dim=0).mean().item()
            
            # 计算logits统计量
            logits_std = logits.std().item()
            logits_mean = logits.mean().item()
            
            print(f"[Debug] tree_std={tree_std:.4f} dna_std={dna_std:.4f}")
            print(f"[Debug] logits_mean={logits_mean:.4f} logits_std={logits_std:.4f}")
            print(f"[Cos] diag={diag:.4f} off={off:.4f} Δ={(diag-off):.4f}")
        
        # 7. 计算方差正则化损失
        var_loss = 0.2 * (  # 权重从 0.05 提高到 0.2
            self._variance_loss(tree_features) +
            self._variance_loss(dna_features)
        )

        # 8. 总损失
        loss += var_loss

        # 9. 打印调试信息
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[Debug] var_loss={var_loss.item():.4f}, logit_scale={logit_scale.item():.4f}")

        return loss