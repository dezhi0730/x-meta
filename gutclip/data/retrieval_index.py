# gutclip/data/retrieval_index.py
"""
FAISS 检索索引 (key = z_dna, payload = y)

示例:
# 1. 构建
python -m gutclip.data.retrieval_index build \
       --embed_pt datasets/train_embeddings.pt \
       --out      datasets/faiss_index \
       --gpu 0            # -1=CPU

# 2. 查询
python -m gutclip.data.retrieval_index query \
       --index datasets/faiss_index.faiss \
       --y     datasets/faiss_index.y.npy \
       --z     query_z.npy \
       --k 5 --gpu 0
"""
import faiss, torch, numpy as np, argparse, os, sys

# ---------- utils ----------
def _l2_normalize(a: np.ndarray):
    norm = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    return a / norm

def _to_gpu(index, gpu_id: int):
    if not hasattr(faiss, "StandardGpuResources"):
        print("[INFO] faiss-cpu detected — keeping index on CPU")
        return index
    res = faiss.StandardGpuResources()
    return faiss.index_cpu_to_gpu(res, gpu_id, index)

# ---------- build ----------
def build(embed_pt: str, out_prefix: str, gpu: int = -1,
          nlist: int = 256, use_l2norm: bool = True):
    obj = torch.load(embed_pt, map_location="cpu")
    z = obj["z"].float().numpy()           # (B,D)
    y = obj["y"].float().numpy()           # (B,N)

    sample_ids = obj.get("sample_ids", [f"idx_{i}" for i in range(len(z))])
    ids_np = np.asarray(sample_ids, dtype=np.str_)   # 统一成字符串数组

    if use_l2norm:
        z = _l2_normalize(z)

    d = z.shape[1]
    n_samples = z.shape[0]
    
    # 修复：根据数据量选择合适的索引类型
    if n_samples < 10000:
        # 小数据集：使用简单的FlatIP索引
        index = faiss.IndexFlatIP(d)
    else:
        # 大数据集：使用IVF索引
        # 修复：使用FlatIP作为量化器，适合内积搜索
        quantizer = faiss.IndexFlatIP(d)
        nlist = min(nlist, n_samples // 30)  # 确保每个cluster有足够样本
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        
        n_train = min(n_samples // 2, 100000)
        train_data = z[:n_train]
        index.train(train_data)
    
    # 添加所有数据
    index.add(z)

    # 修复：保存CPU版本的索引
    cpu_index = faiss.index_gpu_to_cpu(_to_gpu(index, gpu)) if gpu >= 0 else index

    # 保存
    faiss.write_index(cpu_index, f"{out_prefix}.faiss")
    np.save(f"{out_prefix}.y.npy", y)
    np.save(f"{out_prefix}.ids.npy", ids_np)
    np.save(f"{out_prefix}.norm.npy", np.array([use_l2norm], dtype=np.bool_))
    print(f"[✓] index → {out_prefix}.faiss | y → {out_prefix}.y.npy")
    print(f"[INFO] {n_samples} samples, {d} dimensions, {nlist} clusters")

# ---------- query ----------
def query(index_path: str, y_path: str, z_query: np.ndarray,
          k: int = 5, gpu: int = -1):
    index = faiss.read_index(index_path)
    if gpu >= 0:
        index = _to_gpu(index, gpu)

    y_all = np.load(y_path)                # (B,N)
    use_norm = np.load(y_path.replace(".y.npy", ".norm.npy")).item()
    if use_norm:
        z_query = _l2_normalize(z_query)

    z_query = z_query.astype("float32")
    if hasattr(index, 'nprobe'):
        index.nprobe = min(64, getattr(index, 'nlist', 64))

    D, I = index.search(z_query, k)  # D: (Q, k), I: (Q, k)
    y_neighbors = y_all[I]           # (Q, k, N)
    w = torch.softmax(torch.from_numpy(D), dim=1)[..., None]  # (Q, k, 1)
    y_prior = (torch.from_numpy(y_neighbors) * w).sum(1).numpy()  # (Q, N)
    return y_prior, D, I

# ---------- CLI ----------
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    sub = pa.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--embed_pt", required=True)
    b.add_argument("--out",      required=True)
    b.add_argument("--gpu", type=int, default=-1)
    b.add_argument("--nlist", type=int, default=256)
    b.add_argument("--no_norm", action="store_true",
                   help="disable L2 normalization before add()")

    q = sub.add_parser("query")
    q.add_argument("--index", required=True)
    q.add_argument("--y",     required=True)
    q.add_argument("--z",     required=True, help="query z  (npy or pt)")
    q.add_argument("--k", type=int, default=5)
    q.add_argument("--gpu", type=int, default=-1)
    q.add_argument("--out", default="y_prior.npy")

    args = pa.parse_args()

    if args.cmd == "build":
        build(args.embed_pt, args.out, gpu=args.gpu,
              nlist=args.nlist, use_l2norm=not args.no_norm)
    else:  # query
        zq = np.load(args.z) if args.z.endswith(".npy") else \
             torch.load(args.z, map_location="cpu").numpy()
        y_prior, D, I = query(args.index, args.y, zq,
                              k=args.k, gpu=args.gpu)
        np.save(args.out, y_prior)
        print(f"[✓] saved {args.out} | shape {y_prior.shape}")