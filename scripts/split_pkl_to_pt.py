#!/usr/bin/env python3
"""
将大型 all_samples.pkl 拆分为多个小文件
Usage: python scripts/split_pkl_to_pt.py --src /path/to/all_samples.pkl --dst /path/to/tree_split
"""
import argparse
import json
import os
import torch
from tqdm import tqdm
from pathlib import Path


def split_pkl_to_pt(src_pkl: str, dst_dir: str):
    """将 pkl 文件拆分为多个 pt 文件"""
    print(f"[INFO] Loading large pkl file: {src_pkl}")
    all_data = torch.load(src_pkl, map_location='cpu')
    
    # 创建目标目录
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)
    
    # 记录索引信息
    records = []
    
    print(f"[INFO] Splitting {len(all_data)} samples...")
    for sample_id, data in tqdm(all_data.items(), desc="Splitting samples"):
        # 保存单个样本
        sample_path = dst_path / f"{sample_id}.pt"
        
        # 提取数据 - 按照 tree_dataset.py 的格式
        sample_data = {
            "x": data.x.numpy(),
            "coords": data.pos.numpy(),  # pos -> coords
            "edge_index": data.edge_index.numpy(),
            "idx": getattr(data, 'sample_index', 0),
            "abundance": getattr(data, 'otu_abundance', None).numpy() if getattr(data, 'otu_abundance', None) is not None else None,
            "sample_id": sample_id
        }
        
        torch.save(sample_data, sample_path)
        
        # 记录索引
        records.append({
            "sid": sample_id,
            "path": f"{sample_id}.pt"  # 只保存文件名，不保存完整路径
        })
    
    # 保存索引文件
    index_path = dst_path / "index.json"
    with open(index_path, 'w') as f:
        json.dump(records, f, indent=2)
    
    print(f"[INFO] Split complete!")
    print(f"[INFO] Total samples: {len(records)}")
    print(f"[INFO] Index file: {index_path}")
    print(f"[INFO] Sample files: {dst_path}/*.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split large pkl to multiple pt files")
    parser.add_argument("--src", required=True, help="Source pkl file path")
    parser.add_argument("--dst", required=True, help="Destination directory")
    
    args = parser.parse_args()
    split_pkl_to_pt(args.src, args.dst) 