import os
import torch
import argparse
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf

from gutclip.models import GutCLIPModel
from gutclip.data import GutDataModule

def parse_args():
    parser = argparse.ArgumentParser(description='Extract embeddings from GutCLIP model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model checkpoint')
    parser.add_argument('--cfg', type=str, required=True,
                      help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='embeddings',
                      help='Directory to save embeddings')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers for data loading')
    parser.add_argument('--datasets', nargs='+', choices=['train', 'val', 'all'], default=['all'],
                      help='Which datasets to process: train, val, or all (default)')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    cfg = OmegaConf.load(args.cfg)
    
    # Load model
    model = GutCLIPModel(
        tree_dim=cfg.tree_dim,
        dna_dim=cfg.dna_dim,
        output_dict=True
    )
    
    # Load checkpoint with correct format
    ckpt_data = torch.load(args.model_path, map_location="cpu", weights_only=True)
    state_dict = ckpt_data.get("model", ckpt_data.get("state_dict", ckpt_data))
    clean_state = {k[6:] if k.startswith("model.") else k: v 
                  for k, v in state_dict.items()}
    model.load_state_dict(clean_state)
    model = model.eval().to(device)
    
    # Load dataset using GutDataModule
    dm = GutDataModule(cfg)
    
    # 确保使用命令行参数指定的batch_size
    print(f"[INFO] Using batch_size: {args.batch_size}")
    
    # Determine which dataloaders to use
    dataloaders = {}
    if 'train' in args.datasets or 'all' in args.datasets:
        # 临时修改配置中的batch_size
        original_batch_size = cfg.batch_size
        cfg.batch_size = args.batch_size
        dataloaders['train'] = dm.train_dataloader()
        cfg.batch_size = original_batch_size  # 恢复原配置
        print(f"[INFO] Will process training set")
    if 'val' in args.datasets or 'all' in args.datasets:
        # 临时修改配置中的batch_size
        original_batch_size = cfg.batch_size
        cfg.batch_size = args.batch_size
        dataloaders['val'] = dm.val_dataloader()
        cfg.batch_size = original_batch_size  # 恢复原配置
        print(f"[INFO] Will process validation set")
    
    if not dataloaders:
        raise ValueError("No datasets selected for processing")

    # Extract embeddings
    all_tree_embeddings = []
    all_dna_embeddings = []
    all_sample_ids = []
    
    with torch.no_grad():
        for dataset_name, dataloader in dataloaders.items():
            print(f"[INFO] Processing {dataset_name} dataset...")
            
            for batch in dataloader:
                batch = batch.to(device)
                outputs = model(batch)
                
                tree_emb = outputs['tree_emb']
                dna_emb = outputs['dna_emb']
                
                all_tree_embeddings.append(tree_emb.cpu())
                all_dna_embeddings.append(dna_emb.cpu())
                
                # 安全地获取sample_id
                if hasattr(batch, 'sample_id'):
                    if isinstance(batch.sample_id, list):
                        all_sample_ids.extend(batch.sample_id)
                    else:
                        all_sample_ids.append(batch.sample_id)
                else:
                    # 如果没有sample_id，生成默认ID
                    batch_size = tree_emb.size(0)
                    all_sample_ids.extend([f"{dataset_name}_sample_{len(all_sample_ids) + i}" for i in range(batch_size)])
    
    print(f"[INFO] Total samples collected: {len(all_sample_ids)}")
    
    # Concatenate all embeddings
    if all_tree_embeddings:
        tree_embeddings = torch.cat(all_tree_embeddings, dim=0)
        dna_embeddings = torch.cat(all_dna_embeddings, dim=0)
    else:
        raise RuntimeError("No embeddings were extracted!")
    
    # Save embeddings and sample IDs
    output_path = output_dir / 'embeddings.pt'
    torch.save({
        'tree_embeddings': tree_embeddings,
        'dna_embeddings': dna_embeddings,
        'sample_ids': all_sample_ids
    }, output_path)
    print(f"Saved embeddings to {output_path}")
    
    print(f"Tree embeddings shape: {tree_embeddings.shape}")
    print(f"DNA embeddings shape: {dna_embeddings.shape}")
    print(f"Number of samples processed: {len(all_sample_ids)}")

if __name__ == '__main__':
    main() 