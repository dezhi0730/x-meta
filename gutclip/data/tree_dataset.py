import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
from torch_geometric.data import Data, Dataset, Batch
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Optional, Dict, Any
from gutclip.utils import Graph, TreeEGNNPreprocessor

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
    # ⭐ 这里一定要重新 mapping！！
    otu2node_local = {otu: tree.get_node_by_id(otu) for otu in otu_list}
    pre = TreeEGNNPreprocessor(tree)
    pre.fill_abundance(otu2node_local, abundance_vec, otu_list)
    result = pre.process()
    if result is None:
        print(f"[SKIP] Sample {sample_id}: Failed or too few nodes.")
        return None
    x, coords, edge_index = result
    # 打印特征分布
    try:
        feat_mean, feat_std = x.mean(0), x.std(0)
        feat_min, feat_max = x.min(0)[0], x.max(0)[0]
        coords_mean, coords_std = coords.mean(0), coords.std(0)
        print(f"[Sample {sample_id}] Feature mean: {feat_mean.tolist()}")
        print(f"[Sample {sample_id}] Feature std : {feat_std.tolist()}")
        print(f"[Sample {sample_id}] Feature min : {feat_min.tolist()}")
        print(f"[Sample {sample_id}] Feature max : {feat_max.tolist()}")
        print(f"[Sample {sample_id}] Coords mean : {coords_mean.tolist()}")
        print(f"[Sample {sample_id}] Coords std  : {coords_std.tolist()}")
        if (feat_std < 0.05).any():
            print(f"[WARNING] [Sample {sample_id}] 特征模态collapse风险: {feat_std.tolist()}")
    except Exception as e:
        print(f"[ERROR] Printing stats for {sample_id}: {e}")

    return {
        "x": x.numpy(),
        "coords": coords.numpy(),
        "edge_index": edge_index.numpy(),
        "idx": idx,
        "abundance": abundance_vec,
        "sample_id": sample_id
    }

class TreeEGNNDataset(Dataset):
    def __init__(self, dataset_dir: str, force_reprocess: bool = False):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.pickle_path = os.path.join(dataset_dir, 'all_samples.pkl')
        self.meta_path = os.path.join(dataset_dir, 'sample_metadata.pt')
        self.all_data: Dict[str, Data] = {}

        if not force_reprocess and os.path.exists(self.pickle_path) and os.path.exists(self.meta_path):
            print(f"[INFO] Loading cached dataset and metadata from {self.pickle_path}")
            self.all_data = torch.load(self.pickle_path)
            metadata = torch.load(self.meta_path)
            self.sample_ids = metadata['sample_ids']
            self.sample_id_to_idx = metadata['sample_id_to_idx']
            print(f"[INFO] Loaded {len(self.all_data)} tree samples")
            print(f"[INFO] Sample ID mapping: {len(self.sample_id_to_idx)} entries")
            print(f"[INFO] First few sample IDs: {list(self.sample_id_to_idx.keys())[:5]}")
        else:
            print("[INFO] Processing dataset and caching...")
            self.all_data, self.sample_ids, self.sample_id_to_idx = self._process_and_cache()
            torch.save(self.all_data, self.pickle_path)
            torch.save({
                'sample_ids': self.sample_ids,
                'sample_id_to_idx': self.sample_id_to_idx
            }, self.meta_path)
            print(f"[INFO] Dataset cached at {self.pickle_path}")
            print(f"[INFO] Metadata cached at {self.meta_path}")

    def _process_and_cache(self):
        count_path = os.path.join(self.dataset_dir, 'count_matrix.tsv')
        otu_path = os.path.join(self.dataset_dir, 'otu.csv')
        newick_path = os.path.join(self.dataset_dir, 'newick.txt')

        df = pd.read_csv(count_path, sep='\t')
        sample_ids = [sid.replace('.metaphlan.out', '') for sid in df.iloc[:, 0].tolist()]
        count_matrix = df.iloc[:, 1:].astype(np.float32).to_numpy()
        otu_list = pd.read_csv(otu_path, header=None).iloc[0].tolist()
        assert len(otu_list) == count_matrix.shape[1], f"OTU count mismatch: {len(otu_list)} vs {count_matrix.shape[1]}"
        sample_id_to_idx = {sid: i for i, sid in enumerate(sample_ids)}

        print("[INFO] Building shared tree...")
        global_tree = build_shared_tree_and_otu_map(newick_path, otu_list)

        print(f"[INFO] Launching ProcessPool with {cpu_count()} workers")
        all_data = {}
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
                        data = Data(
                            x=torch.tensor(result["x"], dtype=torch.float),
                            pos=torch.tensor(result["coords"], dtype=torch.float),
                            edge_index=torch.tensor(result["edge_index"], dtype=torch.long)
                        )
                        data.sample_index = result["idx"]
                        data.otu_abundance = torch.tensor(result["abundance"], dtype=torch.float)
                        data.sample_id = result["sample_id"]
                        all_data[result["sample_id"]] = data
                except Exception as e:
                    print(f"[TIMEOUT/ERROR] Skipping sample {i}: {e}")

        print(f"[INFO] Successfully processed {len(all_data)} valid samples")
        return all_data, sample_ids, sample_id_to_idx

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        sample_id = list(self.all_data.keys())[idx]
        return self.all_data[sample_id]

    def get_index_by_sample_id(self, sample_id):
        return self.sample_id_to_idx.get(sample_id, None)

    @staticmethod
    def collate_fn(batch: list[Data]) -> Batch:
        return Batch.from_data_list(batch)