import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Data
from gutclip.utils import Graph, TreeEGNNPreprocessor
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm


def build_shared_tree_and_otu_map(newick_path, otu_list):
    """构建全局树并验证OTU映射"""
    tree = Graph()
    tree.build_graph(newick_path)

    # 1. 验证树节点
    print(f"[INFO] Tree nodes: {len(tree.NODE_DICT)}")
    print(f"[INFO] Tree leaves: {sum(1 for node in tree.NODE_DICT.values() if not node.children)}")

    # 2. 验证OTU列表
    print(f"[INFO] OTU list length: {len(otu_list)}")
    print(f"[INFO] First 5 OTUs: {otu_list[:5]}")

    # 3. 可选: 映射检查（只做日志验证）
    otu2node = {}
    missing_otus = []
    for otu in otu_list:
        node = tree.get_node_by_id(otu)
        if node is not None:
            otu2node[otu] = node
        else:
            missing_otus.append(otu)
    print(f"[INFO] Successfully mapped OTUs: {len(otu2node)}/{len(otu_list)}")
    if missing_otus:
        print(f"[WARNING] {len(missing_otus)} OTUs not found in tree")
        print(f"[WARNING] First 5 missing OTUs: {missing_otus[:5]}")
        print(f"[WARNING] First 5 tree node IDs: {list(tree.NODE_DICT.keys())[:5]}")

    # 4. 只返回 tree
    return tree

def process_one_sample(idx, sample_id, abundance_vec, otu_list, global_tree):
    import copy
    tree = copy.deepcopy(global_tree)
    otu2node_local = {otu: tree.get_node_by_id(otu) for otu in otu_list}
    pre = TreeEGNNPreprocessor(tree)
    pre.fill_abundance(otu2node_local, abundance_vec, otu_list)
    result = pre.process()
    if result is None:
        print(f"[SKIP] Sample {sample_id}: Failed or too few nodes.")
        return None
    x, coords, edge_index = result
    
    try:
        feat_std = x.std(0)
        if (feat_std < 0.05).any():
            print(f"[WARNING] Sample {sample_id}: 特征模态collapse风险")
    except Exception as e:
        print(f"[ERROR] Checking stats for {sample_id}: {e}")

    return {
        "x": x.numpy(),
        "coords": coords.numpy(),
        "edge_index": edge_index.numpy(),
        "idx": idx,
        "abundance": abundance_vec,
        "sample_id": sample_id
    }


class TreeSplitDataset(Dataset):

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.split_dir = os.path.join(root_dir, "tree_split")
        self.index_json = os.path.join(self.split_dir, "index.json")

        if not os.path.exists(self.index_json):
            self._preprocess_to_split()

        with open(self.index_json) as fr:
            self.records = json.load(fr)
        print(f"[INFO] TreeSplitDataset: {len(self.records)} samples loaded from {self.index_json}")

    def _preprocess_to_split(self):
        count_path = os.path.join(self.root_dir, 'count_matrix.tsv')
        otu_path = os.path.join(self.root_dir, 'otu.csv')
        newick_path = os.path.join(self.root_dir, 'newick.txt')

        df = pd.read_csv(count_path, sep='\t')
        sample_ids = [sid.replace('.metaphlan.out', '') for sid in df.iloc[:, 0].tolist()]
        count_matrix = df.iloc[:, 1:].astype(np.float32).to_numpy()
        otu_list = pd.read_csv(otu_path, header=None).iloc[0].tolist()
        assert len(otu_list) == count_matrix.shape[1], f"OTU count mismatch: {len(otu_list)} vs {count_matrix.shape[1]}"
        sample_id_to_idx = {sid: i for i, sid in enumerate(sample_ids)}

        print("[INFO] Building shared tree...")
        global_tree = build_shared_tree_and_otu_map(newick_path, otu_list)

        print(f"[INFO] Launching ProcessPool with {cpu_count()} workers")
        os.makedirs(self.split_dir, exist_ok=True)
        records = []

        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            futures = [
                executor.submit(
                    process_one_sample,
                    idx,
                    sample_ids[idx],
                    count_matrix[idx],
                    otu_list,
                    global_tree,
                ) for idx in range(len(sample_ids))
            ]
            for i, f in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Multiprocessing Samples")):
                try:
                    result = f.result(timeout=90)
                    if result is not None:
                        sid = result["sample_id"]
                        pt_filename = f"{sid}.pt"
                        pt_path = os.path.join(self.split_dir, pt_filename)
                        torch.save(result, pt_path)
                        records.append({"sid": sid, "path": pt_filename})
                except Exception as e:
                    print(f"[TIMEOUT/ERROR] Skipping sample {i}: {e}")

        with open(self.index_json, "w") as fw:
            json.dump(records, fw)

        print(f"[INFO] Preprocess complete: {len(records)} samples stored in {self.split_dir}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]                       # {'sid': ..., 'path': ...}
        pt_path = os.path.join(self.split_dir, rec["path"])
        obj = torch.load(pt_path, map_location="cpu")
        
        # 按照 tree_dataset.py 的处理方式
        data = Data(
            x=torch.tensor(obj["x"], dtype=torch.float),
            pos=torch.tensor(obj["coords"], dtype=torch.float),
            edge_index=torch.tensor(obj["edge_index"], dtype=torch.long)
        )
        data.sample_index = obj["idx"]
        data.otu_abundance = torch.tensor(obj["abundance"], dtype=torch.float)
        data.sample_id = rec["sid"]
        return data

    def get_index_by_sample_id(self, sample_id):
        for i, rec in enumerate(self.records):
            if rec["sid"] == sample_id:
                return i
        return None 