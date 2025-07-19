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

def build_shared_tree(newick_path, otu_list):
    tree = Graph(); tree.build_graph(newick_path)
    return tree

def _process(idx, sid, vec, otu_list, gtree):
    t = copy.deepcopy(gtree)
    mapper = {otu: t.get_node_by_id(otu) for otu in otu_list if t.get_node_by_id(otu)}
    pre = TreeEGNNPreprocessor(t)
    valid_otu = list(mapper.keys())
    valid_idx = [otu_list.index(o) for o in valid_otu]
    pre.fill_abundance(mapper, vec[valid_idx], valid_otu)
    res = pre.process()
    if res is None:
        return None
    x, pos, ei, node_zero = res
    d = Data(x=x, pos=pos, edge_index=ei,
             node_zero=node_zero, sample_index=idx,
             otu_abundance=torch.tensor(vec, dtype=torch.float),
             sample_id=sid)
    return d

# ------------------------------------------------------------
class TreeEGNNDataset(Dataset):
    def __init__(self, dataset_dir: str, force: bool = False):
        super().__init__()
        self.dir = dataset_dir
        self.pkl = os.path.join(dataset_dir, "all_samples.pkl")
        self.meta = os.path.join(dataset_dir, "sample_metadata.pt")
        if (not force and os.path.exists(self.pkl) and
                os.path.exists(self.meta)):
            self.data: Dict[str, Data] = torch.load(self.pkl)
            meta = torch.load(self.meta)
            self.sample_ids = meta["sample_ids"]
            self.sid2idx = meta["sample_id_to_idx"]
            return

        cnt = os.path.join(dataset_dir, "count_matrix.tsv")
        otu = os.path.join(dataset_dir, "otu.csv")
        nwk = os.path.join(dataset_dir, "newick.txt")

        df = pd.read_csv(cnt, sep='\t')
        sids = [sid.replace('.metaphlan.out', '') for sid in df.iloc[:, 0]]
        mat = df.iloc[:, 1:].astype(np.float32).to_numpy()
        otu_list = pd.read_csv(otu, header=None).iloc[0].tolist()

        gtree = build_shared_tree(nwk, otu_list)
        self.data = {}
        with ProcessPoolExecutor(max_workers=cpu_count()) as ex:
            futs = [ex.submit(_process, i, sids[i], mat[i],
                              otu_list, gtree)
                    for i in range(len(sids))]
            for f in tqdm(as_completed(futs), total=len(futs)):
                d = f.result()
                if d is not None:
                    self.data[d.sample_id] = d

        self.sample_ids = list(self.data.keys())
        self.sid2idx = {sid: i for i, sid in enumerate(self.sample_ids)}
        torch.save(self.data, self.pkl)
        torch.save({"sample_ids": self.sample_ids,
                    "sample_id_to_idx": self.sid2idx}, self.meta)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        return self.data[sid]

    @staticmethod
    def collate_fn(batch):
        out = Batch.from_data_list(batch)
        if hasattr(out, 'node_zero'):
            out.zero_ratio = out.node_zero.float().mean(dim=1)
        return out