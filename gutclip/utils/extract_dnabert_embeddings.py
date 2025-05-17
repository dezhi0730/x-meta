import argparse, os, gc, logging, queue
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
from multiprocessing import Process, Queue, set_start_method
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger('DNABERT-Chunk')

# ---------- Utils ---------- #

def seq_to_kmer_string(seq: str, k: int = 6) -> str:
    """DNA 序列 → k‑mer 文本串"""
    return ' '.join(seq[i:i + k] for i in range(len(seq) - k + 1))


def fasta_stream(path: Path):
    """逐条 yield 序列字符串"""
    for rec in SeqIO.parse(path, 'fasta'):
        yield str(rec.seq)


def count_sequences(path: Path) -> int:
    """预估序列总数（用于进度条）。"""
    return sum(1 for _ in SeqIO.parse(path, 'fasta'))

# ---------- Worker ---------- #

def gpu_worker(gid: int, task_q: Queue, args):
    torch.cuda.set_device(gid)
    device = torch.device(f'cuda:{gid}')
    logger.info(f'[GPU {gid}] Loading DNABERT…')
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    mdl.to(device).eval().half()
    torch.set_grad_enabled(False)
    d = mdl.config.hidden_size
    logger.info(f'[GPU {gid}] Ready (hidden={d})')

    while True:
        try:
            fp = task_q.get(timeout=5)
        except queue.Empty:
            continue
        if fp is None:
            break

        fp = Path(fp)
        sid = fp.stem.split('_')[0]
        out_path = Path(args.output_dir) / f'{sid}.pt'
        if out_path.exists() and args.skip_existing:
            logger.info(f'[GPU {gid}] Skip {out_path.name}')
            continue

        logger.info(f'[GPU {gid}] Processing {fp.name}')
        chunk_means: List[torch.Tensor] = []
        buf: List[torch.Tensor] = []
        k          = args.kmer
        C          = args.chunk_size
        batch_size = args.batch_size
        batch_k    = []

        def flush():
            """把 buf 中的完整 chunk 转为均值向量并写入列表。"""
            while len(buf) >= C:
                tens = torch.stack(buf[:C], 0)        # (C,d)
                chunk_means.append(tens.mean(0).cpu().half())
                del buf[:C]

        total_seqs = count_sequences(fp)
        for seq in tqdm(fasta_stream(fp), total=total_seqs,
                        desc=f'GPU{gid}:{fp.name}', leave=False):
            batch_k.append(seq_to_kmer_string(seq, k=k))
            if len(batch_k) == batch_size:
                toks = tok(batch_k, return_tensors='pt', padding=True, truncation=True).to(device)
                vecs = mdl(**toks).last_hidden_state[:, 0, :]  # (B,d)
                buf.extend(vecs)                               # 列表追加 Tensor 视图
                batch_k.clear()
                flush()
                torch.cuda.empty_cache()

        # 处理尾批
        if batch_k:
            toks = tok(batch_k, return_tensors='pt', padding=True, truncation=True).to(device)
            vecs = mdl(**toks).last_hidden_state[:, 0, :]
            buf.extend(vecs)
            batch_k.clear()
        flush()  # flush 可能再输出 1 行 (<C 条不会丢)

        mat = torch.stack(chunk_means, 0) if chunk_means else torch.zeros(1, d, dtype=torch.float16)
        torch.save(mat, out_path)
        logger.info(f'[GPU {gid}] Saved {out_path.name}  shape={tuple(mat.shape)}')

        del buf, chunk_means, mat
        torch.cuda.empty_cache(); gc.collect()

    logger.info(f'[GPU {gid}] Shutdown')

# ---------- Main ---------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fasta_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--gpus', type=int, default=torch.cuda.device_count())
    ap.add_argument('--batch_size', type=int, default=128, help='DNABERT 推理批大小')
    ap.add_argument('--chunk_size', type=int, default=1024, help='每多少条 reads 求一次均值')
    ap.add_argument('--kmer', type=int, default=6)
    ap.add_argument('--model', default='zhihan1996/DNA_bert_6')
    ap.add_argument('--skip_existing', action='store_true')
    args = ap.parse_args()

    set_start_method('spawn', force=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    task_q: Queue = Queue(maxsize=args.gpus * 2)
    workers = [Process(target=gpu_worker, args=(gid, task_q, args), daemon=True)
               for gid in range(args.gpus)]
    for p in workers:
        p.start()

    fasta_files = sorted(p for p in Path(args.fasta_dir).iterdir()
                         if p.suffix in ('.fa', '.fasta'))
    logger.info(f'Total FASTA: {len(fasta_files)}')
    for f in tqdm(fasta_files, desc='Dispatch'):  # 任务派发进度
        task_q.put(str(f))

    for _ in workers:
        task_q.put(None)
    for p in workers:
        p.join()
    logger.info('All done ✅')


if __name__ == '__main__':
    main()
