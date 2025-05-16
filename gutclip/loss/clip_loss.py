# gutclip/loss/clip_loss.py
import torch
import torch.nn.functional as F
import torch.distributed as dist

__all__ = ["CLIPLoss"]

class CLIPLoss(torch.nn.Module):
    """
    InfoNCE / CLIP style loss with learnable logit_scale.
    model.forward() 需返回 dict:
        {'tree_emb': (B, D), 'dna_emb': (B, D), 'logit_scale': scalar tensor}
    """
    def __init__(self, local_loss=False):
        super().__init__()
        self.local_loss = local_loss        # True: 仅同卡配对；False: 全局配对

    @staticmethod
    def _gather_with_grad(tensor):
        world = dist.get_world_size()
        if world == 1:
            return tensor
        tensors = [torch.zeros_like(tensor) for _ in range(world)]
        dist.all_gather(tensors, tensor)
        return torch.cat(tensors, 0)

    def forward(self, output):
        t, d = output["tree_emb"], output["dna_emb"]          # (B, D)
        logit_scale = output["logit_scale"].exp().clamp(1e-4, 1e4)

        if not self.local_loss:                               # 全局配对
            t_all, d_all = self._gather_with_grad(t), self._gather_with_grad(d)
        else:                                                 # 本地
            t_all, d_all = t, d

        logits_per_tree = logit_scale * t @ d_all.t()         # (B, B*world)
        logits_per_dna  = logit_scale * d @ t_all.t()

        bsz = t.shape[0]
        rank = dist.get_rank() if dist.is_initialized() else 0
        labels = torch.arange(bsz, device=t.device) + bsz * rank

        loss_t = F.cross_entropy(logits_per_tree, labels)
        loss_d = F.cross_entropy(logits_per_dna,  labels)
        return (loss_t + loss_d) / 2