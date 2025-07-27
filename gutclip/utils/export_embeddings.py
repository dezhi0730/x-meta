"""
一次性导出 (z_dna, y) 以供检索和扩散训练

用法：
python -m gutclip.utils.export_embeddings \
       --cfg gutclip/configs/default.yaml \
       --ckpt checkpoints/gutclip_exp_best_0720.pt \
       --out datasets/train_embeddings.pt \
       --split train        # train / val / full
"""

import torch, argparse, tqdm, os, json
import pandas as pd
import numpy as np
from gutclip.data import GutDataModule   # 你的数据模块
from gutclip.models import GutCLIPModel               # 带 encode_dna
from omegaconf import OmegaConf                       # cfg loader


def load_cfg(path, opts=None):
    """加载配置文件"""
    cfg = OmegaConf.load(path)
    if opts: 
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(opts))
    return cfg


def load_abundance_matrix(cfg):
    """加载丰度矩阵，返回样本ID到丰度向量的映射"""
    count_path = os.path.join(cfg.data.tree_dir, "count_matrix.tsv")
    otu_path = os.path.join(cfg.data.tree_dir, "otu.csv")
    
    print(f"[INFO] Loading abundance matrix from {count_path}")
    
    # 读取丰度矩阵
    df = pd.read_csv(count_path, sep='\t')
    sample_ids = [sid.replace('.metaphlan.out', '') for sid in df.iloc[:, 0]]
    abundance_matrix = df.iloc[:, 1:].astype(np.float32).to_numpy()
    
    # 读取OTU列表
    otu_list = pd.read_csv(otu_path, header=None).iloc[0].tolist()
    
    print(f"[INFO] Loaded {len(sample_ids)} samples, {len(otu_list)} OTUs")
    
    # 创建样本ID到丰度向量的映射
    sample_to_abundance = {}
    for i, sample_id in enumerate(sample_ids):
        sample_to_abundance[sample_id] = torch.tensor(abundance_matrix[i], dtype=torch.float32)
    
    return sample_to_abundance, otu_list


@torch.no_grad()
def run(cfg_path: str, ckpt: str, out: str,
        split: str = "train", batch_size: int = 256,
        num_workers: int = 4, device: str = "cuda:0"):
    """
    1. 读取 cfg 和权重；冻结模型
    2. 遍历 DataLoader：获得 z_dna 与 y
    3. 保存 {'z': z_all, 'y' : y_all, 'sample_ids': ids} 到 out
    """
    cfg = load_cfg(cfg_path)

    # ---------- 1. 模型 ----------
    model = GutCLIPModel(
        tree_dim   = cfg.tree_dim,
        dna_dim    = cfg.dna_dim,
        output_dict=True              # 让 forward 返回字典
    ).to(device).eval()
    sd = torch.load(ckpt, map_location=device)
    model.load_state_dict(sd, strict=False)
    for p in model.parameters():          # 冻结
        p.requires_grad_(False)

    # ---------- 2. 加载丰度矩阵 ----------
    sample_to_abundance, otu_list = load_abundance_matrix(cfg)
    print(f"[INFO] Loaded abundance matrix with {len(sample_to_abundance)} samples")

    # ---------- 3. DataLoader ----------
    # 修复：GutDataModule 只需要 cfg 参数
    dm = GutDataModule(cfg)
    loader = dm.train_dataloader() if split=="train" else dm.val_dataloader()

    zs, ys, ids = [], [], []

    # ---------- 4. 遍历 ----------
    pbar = tqdm.tqdm(loader, desc=f"export-{split}")
    missing_abundance = 0
    for batch in pbar:
        batch = batch.to(device, non_blocking=True)

        # 4.1 DNA → z_dna
        z_dna = model.encode_dna(
            batch.dna,
            getattr(batch, "dna_pad_mask",  None),
            getattr(batch, "dna_rand_mask", None),
            normalize=True        # 和训练时保持一致
        )
        zs.append(z_dna.cpu())

        # 4.2 真值丰度向量 y - 从count_matrix中提取
        batch_abundances = []
        for sample_id in batch.sample_id:
            if sample_id in sample_to_abundance:
                v = sample_to_abundance.get(sample_id)
                if v is None or torch.all(v == 0):
                    # 完全缺失或全 0  → 直接给全 -1
                    v = torch.full((len(otu_list),), -1.0, dtype=torch.float32)
                else:
                    v = torch.log1p(v)
                    v = (v - v.min()) / (v.max() - v.min() + 1e-6)
                    v = v * 2 - 1
                batch_abundances.append(v)
            else:
                print(f"[WARN] Sample {sample_id} not found in abundance matrix, using -1")
                batch_abundances.append(torch.full((len(otu_list),), -1.0, dtype=torch.float32))
                missing_abundance += 1
        
        # 堆叠batch中的丰度向量
        batch_abundance_tensor = torch.stack(batch_abundances)  # (B, N_OTUs)
        ys.append(batch_abundance_tensor.cpu())

        # 4.3 样本 id（保持顺序，用于 debug）
        ids += [sid for sid in batch.sample_id]

    z_all = torch.cat(zs)          # (B,D)
    y_all = torch.cat(ys).float()  # (B,N)

    if missing_abundance > 0:
        print(f"[WARN] {missing_abundance} samples missing abundance data")

    torch.save({'z': z_all, 'y': y_all, 'sample_ids': ids}, out)
    print(f"[✓] wrote {out}  |  z: {tuple(z_all.shape)}, y: {tuple(y_all.shape)}")
    print(f"[INFO] OTU count: {len(otu_list)}")


# ---------------- CLI ----------------
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--cfg",  required=True)
    pa.add_argument("--ckpt", required=True)
    pa.add_argument("--out",  required=True)
    pa.add_argument("--split", default="train", choices=["train", "val", "full"])
    pa.add_argument("--bsz",   type=int, default=256)
    pa.add_argument("--workers", type=int, default=4)
    pa.add_argument("--device", default="cuda:0")
    args = pa.parse_args()

    run(args.cfg, args.ckpt, args.out,
        split=args.split, batch_size=args.bsz,
        num_workers=args.workers, device=args.device)