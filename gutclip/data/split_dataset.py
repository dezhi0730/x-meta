import os, json
import torch, numpy as np, pandas as pd
from typing import Optional, List, Dict

from torch.utils.data import Dataset
from torch_geometric.data import Data
from gutclip.utils import Graph, TreeEGNNPreprocessor
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm


# -------------------------------------------------
# 1. 解析并验证全局树
# -------------------------------------------------
def build_shared_tree(newick_path: str, otu_list: List[str]) -> Graph:
    tree = Graph()
    tree.build_graph(newick_path)

    print(f"[INFO] Tree nodes:  {len(tree.NODE_DICT):,}")
    print(f"[INFO] Tree leaves: {sum(1 for n in tree.NODE_DICT.values() if not n.children):,}")
    print(f"[INFO] OTUs in list: {len(otu_list):,}")

    # 日志验证：OTU → node
    missing = [otu for otu in otu_list if otu not in tree.NODE_DICT]
    print(f"[INFO] Mapped OTUs:   {len(otu_list) - len(missing):,}/{len(otu_list):,}")
    if missing:
        print(f"[WARNING] {len(missing)} OTUs not found in tree   (showing 5) → {missing[:5]}")

    return tree


# -------------------------------------------------
# 2. 单样本处理函数（会在子进程中运行）
# -------------------------------------------------
def process_one_sample(idx: int,
                       sample_id: str,
                       abundance_vec: np.ndarray,
                       otu_list: List[str],
                       shared_tree: Graph,
                       feat_collapse_warn: float = 0.05):
    """
    传入：样本索引、样本 ID、丰度向量、OTU 列表、共享树
    返回：dict | None
    """
    # *无需 deepcopy：子进程里的 shared_tree 已经是独立副本*
    tree = shared_tree
    otu2node = {otu: tree.NODE_DICT.get(otu) for otu in otu_list}

    pre = TreeEGNNPreprocessor(tree)
    pre.fill_abundance(otu2node, abundance_vec, otu_list)
    tensors = pre.process()

    if tensors is None:
        print(f"[SKIP] {sample_id}: empty graph")
        return None

    x, pos, edge_idx, node_zero = tensors

    # 可选：特征崩溃监测
    try:
        feat_std = x.std(0)
        if (feat_std < feat_collapse_warn).all():
            return None  # silently skip collapse sample
    except Exception as e:
        print(f"[ERR ] {sample_id}: std check failed → {e}")

    # 保存为 numpy，确保在 CPU
    return {
        "x":          x.cpu().numpy(),
        "coords":     pos.cpu().numpy(),
        "edge_index": edge_idx.cpu().numpy(),
        "node_zero":  node_zero.cpu().numpy(),
        "idx":        idx,
        "abundance":  abundance_vec.astype(np.float32),
        "sample_id":  sample_id,
    }


# -------------------------------------------------
# 3. 数据集类
# -------------------------------------------------
class TreeSplitDataset(Dataset):
    """
    root_dir/
        ├─ count_matrix.tsv   (rows = samples, cols = OTUs)
        ├─ otu.csv            (one row = OTU IDs in same顺序 as count_matrix)
        ├─ newick.txt
        └─ tree_split/        (自动生成 .pt + index.json)
    """

    def __init__(self, root_dir: str, num_workers: Optional[int] = None, timeout: int = 300):
        self.root_dir   = root_dir
        self.split_dir  = os.path.join(root_dir, "tree_split")
        self.index_json = os.path.join(self.split_dir, "index.json")
        self.num_workers = num_workers or max(1, cpu_count() // 2)
        self.timeout_sec = timeout

        if not os.path.exists(self.index_json):
            self._preprocess_to_split()

        with open(self.index_json) as fr:
            self.records: List[Dict] = json.load(fr)
        print(f"[INFO] TreeSplitDataset ready · {len(self.records):,} samples")

    # -------------------------------------------------
    # 3-1. 预处理：多进程生成 .pt
    # -------------------------------------------------
    def _preprocess_to_split(self):
        count_path  = os.path.join(self.root_dir, "count_matrix.tsv")
        otu_path    = os.path.join(self.root_dir, "otu.csv")
        newick_path = os.path.join(self.root_dir, "newick.txt")

        # 读取矩阵 & OTU
        df           = pd.read_csv(count_path, sep="\t")
        sample_ids   = [sid.replace(".metaphlan.out", "") for sid in df.iloc[:, 0]]
        count_matrix = df.iloc[:, 1:].astype(np.float32).to_numpy()
        otu_list     = pd.read_csv(otu_path, header=None).iloc[0].tolist()

        assert len(otu_list) == count_matrix.shape[1], (
            f"OTU mismatch: {len(otu_list)} vs {count_matrix.shape[1]} in {count_path}"
        )

        # 全局树（一次解析）
        print("[INFO] Building shared tree ...")
        shared_tree = build_shared_tree(newick_path, otu_list)

        # 目标目录
        os.makedirs(self.split_dir, exist_ok=True)
        records = []

        print(f"[INFO] Launching ProcessPool ({self.num_workers} workers)")
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(
                    process_one_sample,
                    idx,
                    sample_ids[idx],
                    count_matrix[idx],
                    otu_list,
                    shared_tree,
                ): sample_ids[idx]
                for idx in range(len(sample_ids))
            }

            for fut in tqdm(as_completed(futures),
                            total=len(futures),
                            desc="Multiprocessing Samples"):
                sid = futures[fut]
                try:
                    res = fut.result(timeout=self.timeout_sec)
                    if res is None:
                        continue
                    pt_name = f"{sid}.pt"
                    torch.save(res, os.path.join(self.split_dir, pt_name))
                    records.append({"sid": sid, "path": pt_name})
                except Exception as e:
                    print(f"[ERROR] {sid}: {e}")

        # 写索引
        with open(self.index_json, "w") as fw:
            json.dump(records, fw)
        print(f"[INFO] Preprocess done · saved {len(records):,}/{len(sample_ids):,}")

    # -------------------------------------------------
    # 3-2. Dataset 接口
    # -------------------------------------------------
    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec     = self.records[idx]        # {"sid": ..., "path": ...}
        obj     = torch.load(os.path.join(self.split_dir, rec["path"]), map_location="cpu")

        data = Data(
            x          = torch.tensor(obj["x"],          dtype=torch.float),
            pos        = torch.tensor(obj["coords"],     dtype=torch.float),
            edge_index = torch.tensor(obj["edge_index"], dtype=torch.long),
        )
        data.sample_index  = obj["idx"]
        data.otu_abundance = torch.tensor(obj["abundance"], dtype=torch.float)
        data.node_zero     = torch.tensor(obj["node_zero"], dtype=torch.float)
        data.sample_id     = rec["sid"]
        return data

    # 快捷：由 sample_id 查索引
    def get_index_by_sample_id(self, sample_id: str) -> Optional[int]:
        for i, r in enumerate(self.records):
            if r["sid"] == sample_id:
                return i
        return None