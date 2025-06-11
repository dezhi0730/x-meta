#!/usr/bin/env python3
"""
GutCLIP cross-modal retrieval评估脚本
================================================
功能：
1. 加载训练好的GutCLIP权重，对给定测试集(JSONL)批量提取DNA序列embedding与Tree embedding。
2. 计算常用检索指标：Top-1 / Recall@K / MRR / Median Rank。
3. 可选：生成t-SNE对齐图与相似度热图，保存到输出目录。
使用示例：
    python gutclip/cmdline/eval.py \
        --ckpt runs/gutclip/checkpoints/epoch=9-step=12345.ckpt \
        --test_jsonl data/test_pairs.jsonl \
        --bs 256 \
        --out_dir eval_out
"""
import os
import json
import argparse
from pathlib import Path
from math import ceil
from typing import List, Tuple, Dict, Any

import torch
import numpy as np
from tqdm import tqdm
import torch_geometric.data as pyg_data

# —— 可选依赖：可视化 ————————————————————————————
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.manifold import TSNE
except ImportError:
    plt = sns = TSNE = None  # 允许纯评估环境无图形库

from gutclip.models import GutCLIPModel

# ============================ 数据加载 ============================

def load_jsonl(path: str) -> Tuple[List[str], List[str], List[str]]:
    """读取JSONL，返回id列表、dna序列列表、tree路径列表"""
    ids, dna_list, tree_list = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            obj = json.loads(ln)
            ids.append(obj["id"])
            dna_list.append(obj["dna"])
            tree_list.append(obj["tree_path"])
    return ids, dna_list, tree_list

def prepare_tree_data(tree_path: str) -> Dict[str, torch.Tensor]:
    """将树文件转换为PyG格式的输入数据"""
    # TODO: 根据您的树数据格式实现具体的转换逻辑
    # 这里需要返回一个包含 x, edge_index, pos, batch 的字典
    raise NotImplementedError("请实现树数据的预处理逻辑")

# ============================ 检索指标 ============================

def retrieval_metrics(sim: torch.Tensor) -> Dict[str, float]:
    """计算检索指标
    Args:
        sim: N×N 余弦相似度矩阵
    Returns:
        包含各项指标的字典
    """
    with torch.no_grad():
        N = sim.size(0)
        ranks = (-sim).argsort(dim=1)  # 每行从大到小排序
        target = torch.arange(N, device=sim.device).unsqueeze(1)
        correct = (ranks == target)

        top1 = correct[:, 0].float().mean()
        recalls = {}
        for k in (5, 10, 50):
            if k < N:
                recalls[f"recall@{k}"] = correct[:, :k].any(dim=1).float().mean()
        # MRR
        idx = torch.nonzero(correct)[:, 1] + 1  # rank从1开始
        mrr = (1.0 / idx.float()).mean()
        median_rank = idx.median().float()
    metrics = {"top1": top1.item(), "mrr": mrr.item(), "median_rank": median_rank.item()}
    metrics.update({k: v.item() for k, v in recalls.items()})
    return metrics

# ============================ 主评估流程 ============================

def evaluate(model: GutCLIPModel,
             dna_list: List[str],
             tree_list: List[str],
             batch_size: int = 256,
             device: str = "cuda",
             fp16: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """批量编码DNA序列与Tree表征
    Args:
        model: GutCLIP模型
        dna_list: DNA序列列表
        tree_list: 树文件路径列表
        batch_size: 批处理大小
        device: 计算设备
        fp16: 是否使用半精度
    Returns:
        text_embs: DNA序列的embedding
        image_embs: Tree的embedding
    """
    model.eval().to(device)
    text_embs, image_embs = [], []
    dtype_ctx = torch.cuda.amp.autocast(enabled=fp16)
    
    with torch.no_grad(), dtype_ctx:
        for i in tqdm(range(0, len(dna_list), batch_size), desc="Encoding"):
            batch_dna = dna_list[i:i + batch_size]
            batch_tree = tree_list[i:i + batch_size]
            
            # 准备树数据
            tree_data = [prepare_tree_data(p) for p in batch_tree]
            tree_batch = pyg_data.Batch.from_data_list(tree_data)
            tree_batch = tree_batch.to(device)
            
            # 准备DNA数据
            dna_batch = torch.tensor(batch_dna, device=device)
            
            # 编码
            t_feat = model.encode_dna(dna_batch)
            i_feat = model.encode_tree(tree_batch)
            
            text_embs.append(t_feat.cpu())
            image_embs.append(i_feat.cpu())
            
    text_embs = torch.cat(text_embs)
    image_embs = torch.cat(image_embs)
    return text_embs, image_embs

# ============================ 可视化辅助 ============================

def plot_tsne(text_embs: torch.Tensor, image_embs: torch.Tensor, path: str, max_points: int = 2000):
    """生成t-SNE对齐可视化
    Args:
        text_embs: DNA序列的embedding
        image_embs: Tree的embedding
        path: 保存路径
        max_points: 最大采样点数
    """
    if TSNE is None:
        print("[WARN] 未安装 scikit-learn / matplotlib，跳过 t-SNE 可视化")
        return
        
    N = text_embs.size(0)
    if N > max_points:
        idx = torch.randperm(N)[:max_points]
        t_emb, i_emb = text_embs[idx], image_embs[idx]
    else:
        t_emb, i_emb = text_embs, image_embs

    X = torch.cat([t_emb, i_emb]).numpy()
    tsne = TSNE(n_components=2, metric="cosine", init="pca", random_state=0)
    Z = tsne.fit_transform(X)
    n = t_emb.size(0)

    plt.figure(figsize=(8, 6))
    plt.scatter(Z[:n, 0], Z[:n, 1], s=10, alpha=.7, label="DNA")
    plt.scatter(Z[n:, 0], Z[n:, 1], s=10, alpha=.7, label="Tree")
    # 连接线(稀疏画)
    step = max(1, n // 200)
    for i in range(0, n, step):
        plt.plot(Z[[i, n + i], 0], Z[[i, n + i], 1], lw=.5, c="gray", alpha=.3)
    plt.legend(); plt.title("t-SNE Alignment")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def plot_similarity_heatmap(sim: torch.Tensor, path: str, max_points: int = 200):
    """生成相似度热图
    Args:
        sim: 相似度矩阵
        path: 保存路径
        max_points: 最大采样点数
    """
    if sns is None:
        print("[WARN] 未安装 seaborn / matplotlib，跳过热图可视化")
        return
        
    N = sim.size(0)
    if N > max_points:
        # 取diag相似度最高的前max_points条
        idx = torch.topk(sim.diag(), max_points).indices
        sim_sub = sim[idx][:, idx]
    else:
        sim_sub = sim
        
    plt.figure(figsize=(6, 5))
    sns.heatmap(sim_sub.cpu(), square=True, cmap="viridis", vmin=-1, vmax=1)
    plt.title("Similarity Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# ============================ CLI ============================

def main():
    parser = argparse.ArgumentParser(description="GutCLIP retrieval evaluation")
    parser.add_argument("--ckpt", required=True, help="模型权重 *.ckpt")
    parser.add_argument("--test_jsonl", required=True, help="测试集 JSONL 文件路径")
    parser.add_argument("--bs", type=int, default=256, help="batch size")
    parser.add_argument("--device", default="cuda", help="cuda / cpu")
    parser.add_argument("--out_dir", default="eval_out", help="结果输出目录")
    parser.add_argument("--fp16", action="store_true", help="使用半精度推理")
    parser.add_argument("--no_vis", action="store_true", help="跳过可视化")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print("[INFO] 读取测试数据……")
    ids, dna_list, tree_list = load_jsonl(args.test_jsonl)
    print(f"   样本数: {len(ids)}")

    print("[INFO] 加载模型……")
    model = GutCLIPModel.load_from_checkpoint(args.ckpt, map_location=args.device)

    print("[INFO] 提取embedding……")
    text_embs, image_embs = evaluate(model, dna_list, tree_list,
                                   batch_size=args.bs, device=args.device, fp16=args.fp16)

    print("[INFO] 计算检索指标……")
    sim = text_embs @ image_embs.T  # 已归一化 -> 余弦相似度
    metrics = retrieval_metrics(sim)

    # 打印 & 保存
    print("\n===== Retrieval Metrics =====")
    for k, v in metrics.items():
        print(f"{k:>12}: {v:.4f}")
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # 可选保存embedding
    torch.save({"ids": ids, "text": text_embs, "image": image_embs},
               os.path.join(args.out_dir, "embeddings.pt"))

    if not args.no_vis:
        print("[INFO] 生成可视化……")
        if TSNE is not None and plt is not None:
            plot_tsne(text_embs, image_embs, os.path.join(args.out_dir, "tsne_alignment.png"))
        if sns is not None:
            plot_similarity_heatmap(sim, os.path.join(args.out_dir, "similarity_heatmap.png"))
        print("   可视化图片保存在", args.out_dir)

    print("[DONE] 评估完成！")

if __name__ == "__main__":
    main() 