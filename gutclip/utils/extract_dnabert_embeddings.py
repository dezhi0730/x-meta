# -*- coding: utf-8 -*-
"""
多 GPU 高效提取 DNABERT **Chunk‑Level** 向量
================================================
**改动要点 (v2 – chunk 模式)**
1. **chunk_size 生效**：每 `chunk_size` 条 reads 求一次均值，得到 (M, d) 矩阵；M≈N/chunk_size。
2. **显存常数级**：只缓存一个小 buffer <= chunk_size × d。
3. **Attention‑ready**：下游可直接做 Set‑/Attention Pooling。
"""
from __future__ import annotations

import argparse, os, gc, logging, queue
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
from multiprocessing import Process, Queue, set_start_method

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger('DNABERT-Chunk')

# ---------- Utils ---------- #

def seq_to_kmer_string(seq: str, k: int = 6) -> str:
    return ' '.join(seq[i:i + k] for i in range(len(seq) - k + 1))


def fasta_stream(path: Path):
    for rec in SeqIO.parse(path, 'fasta'):
        yield str(rec.seq)

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
    logger.info(f'[GPU {gid}] Ready (d={d})')

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
        k = args.kmer
        C = args.chunk_size

        def flush():
            nonlocal buf
            if not buf:
                return
            tens = torch.stack(buf, 0)   # (m,d)
            chunk_means.append(tens.mean(0).cpu().half())
            buf.clear()

        for seq in fasta_stream(fp):
            km = seq_to_kmer_string(seq, k=k)
            toks = tok(km, return_tensors='pt', truncation=True).to(device)
            vec = mdl(**toks).last_hidden_state[:, 0, :].squeeze(0)  # (d,)
            buf.append(vec)
            if len(buf) == C:
                flush()
                torch.cuda.empty_cache()
        flush()  # 余量

        mat = torch.stack(chunk_means, 0)  # (M,d)
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
    ap.add_argument('--batch_size', type=int, default=128, help='已弃用 (chunk 模式单序列推理)')
    ap.add_argument('--chunk_size', type=int, default=1024, help='每多少条 reads 归并一次')
    ap.add_argument('--kmer', type=int, default=6)
    ap.add_argument('--model', default='zhihan1996/DNA_bert_6')
    ap.add_argument('--skip_existing', action='store_true')
    args = ap.parse_args()

    set_start_method('spawn', force=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    task_q: Queue = Queue(maxsize=args.gpus * 2)
    workers = [Process(target=gpu_worker, args=(gid, task_q, args), daemon=True) for gid in range(args.gpus)]
    for p in workers:
        p.start()

    fasta_files = sorted(p for p in Path(args.fasta_dir).iterdir() if p.suffix in ('.fa', '.fasta'))
    logger.info(f'Total FASTA: {len(fasta_files)}')
    for f in fasta_files:
        task_q.put(str(f))
    for _ in workers:
        task_q.put(None)
    for p in workers:
        p.join()
    logger.info('All done ✅')


if __name__ == '__main__':
    main()
