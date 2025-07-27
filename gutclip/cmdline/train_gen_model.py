import os, sys, torch, hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import torch.distributed as dist
from pathlib import Path
import wandb
from torch.utils.tensorboard import SummaryWriter

from gutclip.engine.trainer_diffusion import TrainerDiffusion, RetrievalIndex
from gutclip.data import DiffusionDataModule
from gutclip.utils.seed import set_seed

def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
    else:
        rank = 0
    return rank

@hydra.main(config_path="../configs", config_name="train_gen_model", version_base="1.3")
def main(cfg):
    # 可视化确认
    if os.environ.get("DEBUG_CFG") == "1":
        print(OmegaConf.to_yaml(cfg))

    rank = setup_ddp()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed + rank)

    # 自动推断维度并校验
    dm = DiffusionDataModule(cfg)
    dm.setup(stage="fit")
    z_dim = dm.train_ds.z.shape[1]
    y_dim = dm.train_ds.y.shape[1]
    cfg.model.z_dna_dim = getattr(cfg.model, "z_dna_dim", z_dim)
    cfg.model.y_dim     = getattr(cfg.model, "y_dim", y_dim)
    cfg.model.proj_dim  = getattr(cfg.model, "proj_dim", min(256, y_dim))
    cfg.model.cond_dim  = cfg.model.z_dna_dim + cfg.model.proj_dim

    # 日志
    tb = SummaryWriter(f"runs/{cfg.run_name}") if rank == 0 else None
    if rank == 0 and cfg.wandb.mode != "disabled":
        wandb.init(project=cfg.wandb.project,
                   name=cfg.run_name,
                   mode=cfg.wandb.mode,
                   config=OmegaConf.to_container(cfg, resolve=True))

    # Retrieval
    ret = RetrievalIndex(index_file = cfg.retrieval.index,
                         y_file     = cfg.retrieval.y,
                         ids_file   = getattr(cfg.retrieval, "ids", None),
                         gpu        = rank,
                         k          = cfg.retrieval.k)

    # 检查训练/验证集ID重叠
    import numpy as np
    train_ids = set(np.load(cfg.retrieval.index.replace(".faiss", ".ids.npy")).tolist())
    val_ids = set(dm.val_ds.sample_ids)
    intersection = len(train_ids & val_ids)
    print(f"[CHECK] #train_ids: {len(train_ids)}, #val_ids: {len(val_ids)}, intersection: {intersection}")
    if intersection > 0:
        print(f"[WARN] 训练集与验证集有 {intersection} 个重叠样本！")
    else:
        print("[✓] 训练集与验证集无重叠")

    # Trainer
    trainer = TrainerDiffusion(cfg, dataloader=dm.train_dataloader(),
                               val_loader=dm.val_dataloader(), retrieval=ret, device=device)

    best_mse, patience = 1e9, 0
    best_ckpt_path = None  # 记录最优参数路径
    ckpt_dir = Path("checkpoints/diffusion"); ckpt_dir.mkdir(exist_ok=True)

    for ep in range(cfg.train.epochs):
        if dist.is_initialized():
            dm.train_dataloader().sampler.set_epoch(ep)
        train_loss = trainer.train_one_epoch(ep, tb)
        val_mse = trainer.evaluate(tb, ep)

        if rank == 0:
            latest_ckpt_path = trainer.save_ckpt("latest", ep, {"mse": val_mse}, remove_old=True)
            
            if val_mse < best_mse - cfg.train.min_delta:
                # 保存新的最优参数（自动删除旧的）
                best_ckpt_path = trainer.save_ckpt("best", ep, {"mse": val_mse}, remove_old=True)
                best_mse, patience = val_mse, 0
                print(f"[✓] new best mse={val_mse:.6f} -> {best_ckpt_path.name}")
            else:
                patience += 1
                if patience >= cfg.train.patience:
                    print(f"[EarlyStop] no improv >{cfg.train.patience} epochs")
                    break
            if cfg.wandb.mode != "disabled":
                wandb.log({
                    "epoch": ep,
                    "train/total": train_loss["total"],
                    "train/mse":   train_loss["mse"],
                    "train/rank":  train_loss["rank"],
                    "val/mse":     val_mse
                })

    if rank == 0:
        tb and tb.close()
        wandb.finish()
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    sys.exit(main())