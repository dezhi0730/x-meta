#!/usr/bin/env python3
"""
Batch DNA ➜ Tree Retrieval with Cache & Metrics (GutCLIP)
========================================================
支持两种调用：
1. **单样本**  `--dna_pt_single sample123.pt`
2. **批量样本** `--dna_csv samples.csv --dna_pt_dir dna_pts/`
   * `samples.csv` 至少包含一列 `sample_id`
   * DNA 向量文件默认 `<dna_pt_dir>/<sample_id>.pt` (可带 .pt 或不带)

输出
-----
* `topk_matches.csv`  — 每个 DNA 样本的 Top‑k Tree 检索结果
* `metrics.json`      — Top‑1 / Recall@5 / Recall@10 / MRR / Median Rank （若 DNA IDs 与 Tree IDs 完全对齐时才计算）

Tree embedding 首次编码后自动缓存为 `<tree_dataset_dir>/tree_emb.pt`，下次秒级加载。
"""
from __future__ import annotations
import argparse, json, csv
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import pandas as pd
import torch_geometric.data as pyg_data
import torch.nn.functional as F

from gutclip.models import GutCLIPModel
from gutclip.data.tree_dataset import TreeEGNNDataset

EMBED_DIM   = 1280
TREE_DIM    = 256
DNA_DIM     = 768

# ----------------- Metrics -----------------

def retrieval_metrics(sim: torch.Tensor) -> Dict[str, float]:
    ranks = (-sim).argsort(dim=1)
    target = torch.arange(sim.size(0), device=sim.device).unsqueeze(1)
    correct = ranks.eq(target)
    top1 = correct[:, 0].float().mean()
    recall5  = correct[:, :5 ].any(dim=1).float().mean()
    recall10 = correct[:, :10].any(dim=1).float().mean()
    rank_pos = torch.nonzero(correct)[:, 1] + 1
    mrr = (1.0 / rank_pos.float()).mean(); median = rank_pos.median().float()
    return {"top1": top1.item(), "recall@5": recall5.item(), "recall@10": recall10.item(),
            "mrr": mrr.item(), "median_rank": median.item()}

# ----------------- DNA loading -----------------

def load_single_dna(path: Path) -> torch.Tensor:
    t = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.is_tensor(t) or t.dim()!=2 or t.size(1)!=DNA_DIM:
        raise ValueError(f"{path} 不是期望的 Tensor[N,{DNA_DIM}] 格式")
    return t.unsqueeze(0).float()  # (1,N,768)


def load_batch_dna(csv_path: Path, pt_dir: Path) -> Tuple[List[str], List[torch.Tensor]]:
    ids, tensors = [], []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        if 'sample_id' not in reader.fieldnames:
            raise ValueError("CSV 必须包含列 'sample_id'")
        for row in reader:
            sid = row['sample_id']
            cand = [pt_dir/f"{sid}.pt", pt_dir/sid]  # 支持有无 .pt
            for p in cand:
                if p.exists():
                    t = load_single_dna(p)
                    ids.append(sid); tensors.append(t)
                    break
            else:
                raise FileNotFoundError(f"找不到 DNA 向量文件 {sid}.pt")
    return ids, tensors

# ----------------- Tree cache -----------------

def build_or_load_tree_emb(dataset_dir: Path, ckpt: Path, device: str, fp16: bool, batch_size: int) -> Tuple[List[str], torch.Tensor]:
    """构建或加载树编码缓存
    
    Args:
        dataset_dir: 树数据集目录
        ckpt: 模型检查点路径
        device: 计算设备
        fp16: 是否使用半精度
        batch_size: 批处理大小
    
    Returns:
        Tuple[List[str], torch.Tensor]: 样本ID列表和对应的树编码
    """
    # 1. 检查缓存
    cache_path = dataset_dir/"tree_emb.pt"
    if cache_path.exists():
        obj = torch.load(cache_path, map_location="cpu", weights_only=True)
        return obj['ids'], obj['emb'].float()

    # 2. 加载数据集和模型
    dataset = TreeEGNNDataset(str(dataset_dir))
    model = GutCLIPModel(embed_dim=EMBED_DIM, tree_dim=TREE_DIM, dna_dim=DNA_DIM)
    
    # 3. 加载权重
    ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=True)
    state_dict = ckpt_data.get("model", ckpt_data.get("state_dict", ckpt_data))
    clean_state = {k[6:] if k.startswith("model.") else k: v 
                  for k, v in state_dict.items()}
    model.load_state_dict(clean_state)
    model.eval().to(device)
    
    # 4. 批量编码树数据
    tree_embeddings = []
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=fp16):
        for i in range(0, len(dataset), batch_size):
            # 构建批次
            batch = pyg_data.Batch.from_data_list(
                dataset[i:i + batch_size]
            ).to(device)
            
            # 提取需要的字段
            tree_data = {
                "x": batch.x,
                "edge_index": batch.edge_index,
                "pos": batch.pos,
                "batch": batch.batch
            }
            
            # 编码并收集结果
            tree_emb = model.encode_tree(tree_data)
            tree_embeddings.append(tree_emb.cpu())
    
    # 5. 合并结果并保存缓存
    tree_emb = torch.cat(tree_embeddings, dim=0)
    torch.save({
        "ids": dataset.sample_ids,
        "emb": tree_emb.half()  # 使用半精度保存以节省空间
    }, cache_path)
    
    print(f"[INFO] Tree embedding 缓存已保存 → {cache_path}")
    return dataset.sample_ids, tree_emb

# ----------------- Main -----------------

def main():
    p = argparse.ArgumentParser("Batch DNA→Tree retrieval with cache & metrics")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dna_pt_single", type=str, help="单样本 .pt")
    src.add_argument("--dna_csv", type=str, help="包含 sample_id 的 CSV")

    p.add_argument("--dna_pt_dir", type=str, help="批量模式下 .pt 文件所在目录")
    p.add_argument("--tree_dataset_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--out_dir", default="eval_out")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    
    if args.dna_pt_single:
        dna_ids = [Path(args.dna_pt_single).stem]
        dna_tensors = [load_single_dna(Path(args.dna_pt_single))]
    else:
        if not args.dna_pt_dir:
            raise ValueError("批量模式需要 --dna_pt_dir")
        dna_ids, dna_tensors = load_batch_dna(Path(args.dna_csv), Path(args.dna_pt_dir))
    print(f"[INFO] 加载 DNA 样本 {len(dna_ids)} 条")

    # 2. Tree embedding cache
    tree_ids, tree_emb = build_or_load_tree_emb(Path(args.tree_dataset_dir), Path(args.ckpt),
                                                args.device, args.fp16, args.bs)

    # Filter DNA samples to only keep those with matching tree samples
    tree_id2row = {sid: i for i, sid in enumerate(tree_ids)}
    keep_ids, keep_tensors, keep_tree_rows = [], [], []
    for sid, tens in zip(dna_ids, dna_tensors):
        if sid in tree_id2row:
            keep_ids.append(sid)
            keep_tensors.append(tens)
            keep_tree_rows.append(tree_id2row[sid])
        else:
            print(f"[WARN] 跳过 DNA {sid}: 在 Tree 数据中找不到对应样本")

    if not keep_ids:
        raise RuntimeError("所有 DNA 样本都找不到对应 Tree！")

    # Update variables with filtered data
    dna_ids = keep_ids
    dna_tensors = keep_tensors
    tree_emb = tree_emb[torch.tensor(keep_tree_rows)]
    tree_ids = [tree_ids[i] for i in keep_tree_rows]  # Update tree_ids to match filtered embeddings
    print(f"[INFO] 过滤后剩余 {len(dna_ids)} 个匹配样本")

    # 3. Load model (dna encoder)
    ckpt_data = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    state_dict = ckpt_data.get("model", ckpt_data.get("state_dict", ckpt_data))
    
    # 移除前缀
    clean_state = {}
    for k, v in state_dict.items():
        if k.startswith("model."):   # Lightning
            k = k[len("model."):]
        if k.startswith("module."):  # DDP
            k = k[len("module."):]
        clean_state[k] = v
    
    model = GutCLIPModel(embed_dim=EMBED_DIM, tree_dim=TREE_DIM, dna_dim=DNA_DIM)
    model.load_state_dict(clean_state)
    model.eval().to(args.device)
    

    # Print logit_scale for verification
    print(f"\n[诊断] logit_scale = {model.logit_scale.item():.4f}")
    print(f"      exp(logit_scale) = {model.logit_scale.exp().item():.4f}")

    # 4. Encode DNA & compute similarity
    dna_vecs = []
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=args.fp16):
        for t in dna_tensors:
            # Match training pipeline: only project and normalize
            v = model.proj_dna(t.to(args.device))  # (1,1280)
            v = F.normalize(v, dim=-1).cpu()
            dna_vecs.append(v)
    dna_emb = torch.cat(dna_vecs)

    sim = dna_emb @ tree_emb.T  # (N_dna, N_tree)

    # 5. Save top‑k
    rows = []
    for i, sid in enumerate(dna_ids):
        idx = torch.topk(sim[i], k=args.topk).indices.tolist()
        for r, j in enumerate(idx):
            rows.append({"dna_id": sid, "tree_id": tree_ids[j],
                         "rank": r+1, "sim": sim[i,j].item()})
    pd.DataFrame(rows).to_csv(out_dir/"topk_matches.csv", index=False)
    print("[OK] Top‑k 结果写入", out_dir/"topk_matches.csv")

    # 6. Metrics (only if sizes match and ids align)
    if len(dna_ids)==len(tree_ids) and dna_ids==tree_ids:
        # Diagnostic check for ID alignment
        mismatch = [i for i,(d,t) in enumerate(zip(dna_ids, tree_ids)) if d!=t]
        print("\n[诊断] ID 对齐检查:")
        print(f"总样本: {len(dna_ids)}")
        print(f"首 5 对: {list(zip(dna_ids[:5], tree_ids[:5]))}")
        print(f"顺序一致的个数: {len(dna_ids)-len(mismatch)}")
        
        if mismatch:
            print("\n[警告] 发现 ID 错位! 示例错位 idx → (dna_id, tree_id):")
            for i in mismatch[:10]:
                print(f"  {i}: {dna_ids[i]} ≠ {tree_ids[i]}")
            print("\n[错误] ID 未对齐，跳过指标计算")
        else:
            # Check diagonal vs off-diagonal similarities
            diag = sim.diag()
            off = sim[~torch.eye(sim.size(0), dtype=torch.bool)]
            print("\n[诊断] 相似度分布:")
            print(f"  对角线平均: {diag.mean().item():.4f}")
            print(f"  非对角线平均: {off.mean().item():.4f}")
            print(f"  对角线 > 非对角线的比例: {(diag > off.mean()).float().mean().item():.2%}")
            
            # Check with temperature scaling
            gamma = model.logit_scale.item()
            sim2 = gamma * sim
            diag2 = sim2.diag()
            off2 = sim2[~torch.eye(sim2.size(0), dtype=torch.bool)]
            print(f"\n[诊断] 温度缩放后 (γ = {gamma:.2f}):")
            print(f"  对角线平均: {diag2.mean().item():.4f}")
            print(f"  非对角线平均: {off2.mean().item():.4f}")
            print(f"  对角线 > 非对角线的比例: {(diag2 > off2.mean()).float().mean().item():.2%}")
            
            # Use temperature-scaled similarities for metrics
            metrics = retrieval_metrics(sim2)
            with open(out_dir/"metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            print("\nMetrics:")
            for k,v in metrics.items():
                print(f"  {k:<12}= {v:.4f}" if not k.startswith("median") else f"  {k:<12}= {v:.0f}")
    else:
        print("[INFO] ID 对不齐 — 跳过指标计算")


if __name__ == "__main__":
    main()
