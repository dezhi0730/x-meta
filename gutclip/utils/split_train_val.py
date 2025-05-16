# gutclip/utils/split_train_val.py
"""
用法：
    python -m gutclip.utils.split_train_val \
        --dna_dir /path/to/dna_embeddings \
        --out_dir /path/to/meta_csv \
        --ratio 0.9 \
        --seed 42
生成：
    train_meta.csv   sample_id 列
    val_meta.csv
"""
import argparse, random
from pathlib import Path
import pandas as pd


def split_ids(dna_dir: Path, ratio: float, seed: int = 42):
    ids = [p.stem for p in dna_dir.glob("*.pt")]
    random.Random(seed).shuffle(ids)

    n_train = int(len(ids) * ratio)
    train_ids, val_ids = ids[:n_train], ids[n_train:]
    return train_ids, val_ids


def write_csv(ids: list[str], path: Path):
    df = pd.DataFrame({"sample_id": ids})
    df.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dna_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--ratio",  type=float, default=0.9,
                    help="train ratio，默认 0.9")
    ap.add_argument("--seed",   type=int,   default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_ids, val_ids = split_ids(args.dna_dir, args.ratio, args.seed)

    write_csv(train_ids, args.out_dir / "train_meta.csv")
    write_csv(val_ids,   args.out_dir / "val_meta.csv")

    print(f"[OK] Train {len(train_ids)}, Val {len(val_ids)}  (ratio={args.ratio})")
    print(f"      -> {args.out_dir / 'train_meta.csv'}")
    print(f"      -> {args.out_dir / 'val_meta.csv'}")


if __name__ == "__main__":
    main()