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
    dataloader = dm.val_dataloader()  # Use training dataloader for inference
    
    # Extract embeddings
    all_tree_embeddings = []
    all_dna_embeddings = []
    all_sample_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = model(batch)
            
            tree_emb = outputs['tree_emb']
            dna_emb = outputs['dna_emb']
            
            all_tree_embeddings.append(tree_emb.cpu())
            all_dna_embeddings.append(dna_emb.cpu())
            all_sample_ids.extend(batch.sample_id)
    
    # Concatenate all embeddings
    tree_embeddings = torch.cat(all_tree_embeddings, dim=0)
    dna_embeddings = torch.cat(all_dna_embeddings, dim=0)
    
    # Save embeddings and sample IDs
    torch.save({
        'tree_embeddings': tree_embeddings,
        'dna_embeddings': dna_embeddings,
        'sample_ids': all_sample_ids
    }, output_dir / 'embeddings.pt')
    
    print(f"Saved embeddings to {output_dir / 'embeddings.pt'}")
    print(f"Tree embeddings shape: {tree_embeddings.shape}")
    print(f"DNA embeddings shape: {dna_embeddings.shape}")
    print(f"Number of samples processed: {len(all_sample_ids)}")

if __name__ == '__main__':
    main() 