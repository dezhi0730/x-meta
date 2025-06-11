#!/usr/bin/env python3
"""
Single‑DNA ➜ Tree Retrieval (GutCLIP)
====================================
只保留 *模式 C* —— 给定单个 DNA 预计算向量 (`.pt`)，
从批量 Tree 数据中找出 Top‑k 最相似。

输入文件格式
-------------
* **dna_pt_single**  : `torch.Tensor[N, 768]` — 多条 reads 的向量 (float / half)
* **tree_pt**        : `torch.save({"sample_ids": [...], "sample_id_to_idx": {...}})` — 样本元数据

>>> Example
```bash
python gutclip/cmdline/dna2tree.py \
    --dna_pt_single sample123.pt \
    --tree_pt datasets/raw/tree/sample_metadata.pt \
    --ckpt /data/home/wudezhi/project/school/x-meta/checkpoints/gutclip_exp_best.pt \
    --topk 10
```
⮕ 生成 `eval_out/topk_matches.csv`
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
import pandas as pd
import torch_geometric.data as pyg_data

from gutclip.models import GutCLIPModel
from gutclip.data.tree_dataset import TreeEGNNDataset

# ============ IO ============

def load_single_dna_pt(path: str) -> torch.Tensor:
    """加载单个DNA的预计算向量
    Args:
        path: DNA向量文件路径
    Returns:
        Tensor[1, N, 768] - 添加batch维度
    """
    t = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.is_tensor(t) or t.dim()!=2 or t.size(1)!=768:
        raise ValueError("dna_pt_single 应为 Tensor[N,768]")
    return t.unsqueeze(0).float()        # → (1,N,768)


def load_tree_dataset(tree_pt_path: str) -> TreeEGNNDataset:
    """加载树数据集
    Args:
        tree_pt_path: 树数据文件路径
    Returns:
        TreeEGNNDataset对象
    """
    dataset_dir = str(Path(tree_pt_path).parent)
    return TreeEGNNDataset(dataset_dir)


def save_topk_csv(sim: torch.Tensor, dataset: TreeEGNNDataset, dna_id: str, k: int, out_csv: Path):
    """保存Top-k检索结果到CSV
    Args:
        sim: 相似度矩阵 [1, M]
        dataset: 树数据集
        dna_id: DNA样本ID
        k: 返回的top-k数量
        out_csv: 输出CSV路径
    """
    idx = torch.topk(sim.squeeze(0), k=k).indices.tolist()
    rows = []
    for r, j in enumerate(idx):
        sample_id = dataset.sample_ids[j]
        rows.append({
            "dna_id": dna_id,
            "tree_id": sample_id,
            "rank": r+1,
            "sim": sim[0,j].item()
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)

# ============ 主流程 ============

def main():
    ap = argparse.ArgumentParser("Single DNA → Tree retrieval (GutCLIP)")
    ap.add_argument("--dna_pt_single", required=True, help="单个DNA的预计算向量文件")
    ap.add_argument("--tree_pt", required=True, help="树数据文件路径")
    ap.add_argument("--ckpt", required=True, help="模型权重文件")
    ap.add_argument("--device", default="cuda", help="计算设备")
    ap.add_argument("--fp16", action="store_true", help="使用半精度")
    ap.add_argument("--topk", type=int, default=5, help="返回的top-k数量")
    ap.add_argument("--out_dir", default="eval_out", help="输出目录")
    ap.add_argument("--bs", type=int, default=32, help="树编码的批处理大小")
    args = ap.parse_args()

    # 创建输出目录
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] 加载数据...")
    # 加载DNA向量
    dna_tensor = load_single_dna_pt(args.dna_pt_single)
    print(f"  DNA向量形状: {dna_tensor.shape}")
    
    # 加载树数据集
    dataset = load_tree_dataset(args.tree_pt)
    print(f"  树数量: {len(dataset)}")
    print(f"  样本ID列表: {dataset.sample_ids[:5]}...")

    # ---------- 1. 读取 checkpoint ----------
    ckpt = torch.load(args.ckpt, map_location="cpu")

    # Lightning / 自定义 save_checkpoint() 一般存储在 'model'
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))

    # ---------- 2. 处理前缀 ----------
    clean_state = {}
    for k, v in state_dict.items():
        if k.startswith("model."):   # Lightning
            k = k[len("model."):]
        if k.startswith("module."):  # DDP
            k = k[len("module."):]
        clean_state[k] = v

    # ---------- 3. 创建模型 (参数要和训练一致) ----------
    model = GutCLIPModel(embed_dim=1280, tree_dim=256, dna_dim=768)

    # ---------- 4. 加载 ----------
    missing, unexpected = model.load_state_dict(clean_state)
    print("[INFO] missing:", missing)       # 通常为空或 少量 BN running stats
    print("[INFO] unexpected:", unexpected) # 应该为空
    model.to(args.device).eval()

    print("[INFO] 编码树数据...")
    tree_embeddings = []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.fp16):
        for i in range(0, len(dataset), args.bs):
            batch = dataset[i:i + args.bs]
            batch = pyg_data.Batch.from_data_list(batch).to(args.device)
            # 只传入模型需要的字段
            tree_data = {
                "x": batch.x,
                "edge_index": batch.edge_index,
                "pos": batch.pos,
                "batch": batch.batch
            }
            tree_emb = model.encode_tree(tree_data)
            tree_embeddings.append(tree_emb.cpu())
    tree_emb = torch.cat(tree_embeddings, dim=0)
    print(f"  树embedding形状: {tree_emb.shape}")

    print("[INFO] 计算相似度...")
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.fp16):
        q_emb = model.encode_dna(dna_tensor.to(args.device)).cpu()  # (1,D)
    sim = q_emb @ tree_emb.T                                        # (1,M)

    # 保存结果
    out_csv = out_dir / "topk_matches.csv"
    save_topk_csv(sim, dataset, Path(args.dna_pt_single).stem,
                  k=args.topk, out_csv=out_csv)
    
    print(f"\n[OK] Top‑{args.topk} 检索结果:")
    # 打印前3个结果
    idx = torch.topk(sim.squeeze(0), k=min(3, args.topk)).indices.tolist()
    for r, j in enumerate(idx):
        sample_id = dataset.sample_ids[j]
        print(f"  {r+1}. {sample_id} (sim={sim[0,j].item():.4f})")
    print(f"\n完整结果已保存至: {out_csv}")


if __name__ == "__main__":
    main() 