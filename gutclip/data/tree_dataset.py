import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data, Dataset
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count



# === 顶层函数，子进程执行 ===
def process_one_sample(idx, sample_id, abundance_vector, otu_list, newick_str):
    try:
        import tempfile
        import copy
        from utils import Graph, TreeEGNNPreprocessor
        if np.isnan(abundance_vector).any():
            return None

        # 从 newick 字符串构造 graph
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as f:
            f.write(newick_str)
            f.flush()
            tree_path = f.name

        g = Graph()
        g.build_graph(tree_path)
        os.remove(tree_path)

        g.populate_graph(otu_list, abundance_vector)
        preprocessor = TreeEGNNPreprocessor(g)
        x, coords, edge_index = preprocessor.process()

        if torch.isnan(x).any() or torch.isnan(coords).any():
            return None

        return {
            "x": x.numpy(),
            "coords": coords.numpy(),
            "edge_index": edge_index.numpy(),
            "idx": idx,
            "abundance": abundance_vector,
            "sample_id": sample_id
        }

    except Exception as e:
        print(f"[ERROR] Sample {sample_id}: {e}")
        return None


class TreeEGNNDataset(Dataset):
    def __init__(self, dataset_dir: str):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.pickle_path = os.path.join(dataset_dir, 'all_samples.pkl')
        self.meta_path = os.path.join(dataset_dir, 'sample_metadata.pt')

        if os.path.exists(self.pickle_path) and os.path.exists(self.meta_path):
            print(f"[INFO] Loading cached dataset and metadata from {self.pickle_path}")
            self.all_data = torch.load(self.pickle_path)
            metadata = torch.load(self.meta_path)
            self.sample_ids = metadata['sample_ids']
            self.sample_id_to_idx = metadata['sample_id_to_idx']
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

        with open(newick_path, 'r') as f:
            newick_str = f.read()

        print(f"[INFO] Launching ProcessPool with {cpu_count()} workers")
        all_data = []
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            futures = [
                executor.submit(
                    process_one_sample,
                    idx,
                    sample_ids[idx],
                    count_matrix[idx],
                    otu_list,
                    newick_str
                ) for idx in range(len(sample_ids))
            ]

            for i, f in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Multiprocessing Samples")):
                try:
                    result = f.result(timeout=60)
                    if result is not None:
                        data = Data(
                            x=torch.tensor(result["x"], dtype=torch.float),
                            pos=torch.tensor(result["coords"], dtype=torch.float),
                            edge_index=torch.tensor(result["edge_index"], dtype=torch.long)
                        )
                        data.sample_index = result["idx"]
                        data.otu_abundance = torch.tensor(result["abundance"], dtype=torch.float)
                        data.sample_id = result["sample_id"]
                        all_data.append(data)
                except Exception as e:
                    print(f"[TIMEOUT/ERROR] Skipping sample {i}: {e}")

        return all_data, sample_ids, sample_id_to_idx

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return self.all_data[idx]

    def get_index_by_sample_id(self, sample_id):
        return self.sample_id_to_idx.get(sample_id, None)
