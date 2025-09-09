#!/usr/bin/env python3
"""
Extract taxa embeddings from count matrix data using pretrained GutCLIP model.
STREAMING VERSION: Process one TSV file at a time and extract embeddings immediately.

This script is specifically designed for processing count matrix TSV files where:
1. Each TSV contains multiple samples with taxa abundance data
2. Each sample needs to be processed independently 
3. We only have taxa data (no DNA pairs)
4. Each sample gets its own tree structure based on its abundance values
5. Process one TSV file at a time to save memory

Usage:
    python extract_taxa_embeddings_v2_streaming.py \
        --model_path /path/to/pretrained/model.ckpt \
        --cfg /path/to/config.yaml \
        --data_dir /path/to/count_matrix_files \
        --output_dir /path/to/output \
        --batch_size 32

Features:
    - Processes one TSV file at a time to save memory
    - Automatically skips files that have already been processed
    - Use --force_reprocess to override existing files
    - Resumes processing from where it left off if interrupted
    - Supports file range selection for parallel processing
    - Can run multiple instances simultaneously for speedup
"""

import os
import sys
import torch
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import warnings
warnings.filterwarnings('ignore')
import copy

# Add the gutclip directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from gutclip.models import GutCLIPModel
from gutclip.utils import Graph, TreeEGNNPreprocessor


def build_shared_tree(newick_path, otu_list):
    """Build shared tree - exactly like original TreeEGNNDataset."""
    tree = Graph()
    tree.build_graph(newick_path)
    return tree

def _process(idx, sid, vec, otu_list, gtree):
    """Process a single sample - exactly like original TreeEGNNDataset."""
    try:
        # Deep copy the shared tree for this sample
        t = copy.deepcopy(gtree)
        
        # Map OTUs to tree nodes
        mapper = {otu: t.get_node_by_id(otu) for otu in otu_list if t.get_node_by_id(otu)}
        pre = TreeEGNNPreprocessor(t)
        valid_otu = list(mapper.keys())
        valid_idx = [otu_list.index(o) for o in valid_otu]
        pre.fill_abundance(mapper, vec[valid_idx], valid_otu)
        res = pre.process()
        if res is None:
            return None
        x, pos, ei, node_zero = res
        
        # Create Data object exactly like original
        d = Data(x=x, pos=pos, edge_index=ei,
                 node_zero=node_zero, sample_index=idx,
                 otu_abundance=torch.tensor(vec, dtype=torch.float),
                 sample_id=sid)
        return d
        
    except Exception as e:
        print(f"Error processing sample {sid}: {e}")
        return None


class StreamingTaxaDataset:
    """Streaming dataset that processes one TSV file at a time.
    
    This class processes TSV files one by one and immediately extracts embeddings,
    saving memory compared to loading all samples at once.
    """
    
    def __init__(self, data_dir, tree_path=None, otu_list_path=None, num_workers=None, use_multiprocess=True, start_idx=None, end_idx=None):
        """
        Initialize streaming dataset.
        
        Args:
            data_dir: Directory containing count matrix TSV files
            tree_path: Path to phylogenetic tree file (Newick format)
            otu_list_path: Path to OTU list file
            num_workers: Number of workers for multiprocessing (default: cpu_count())
            use_multiprocess: Whether to use multiprocessing (default: True)
            start_idx: Starting file index (0-based, default: 0)
            end_idx: Ending file index (exclusive, default: None for all files)
        """
        self.data_dir = Path(data_dir)
        self.tree_path = tree_path
        self.otu_list_path = otu_list_path
        self.num_workers = num_workers or cpu_count()
        self.use_multiprocess = use_multiprocess
        self.start_idx = start_idx or 0
        self.end_idx = end_idx
        
        # Find all TSV files and sort them
        all_tsv_files = sorted(list(self.data_dir.glob("*.tsv")))
        if not all_tsv_files:
            raise ValueError(f"No TSV files found in {data_dir}")
        
        # Apply file range selection
        if self.end_idx is None:
            self.end_idx = len(all_tsv_files)
        
        # Validate indices
        if self.start_idx < 0:
            raise ValueError(f"start_idx must be >= 0, got {self.start_idx}")
        if self.end_idx > len(all_tsv_files):
            raise ValueError(f"end_idx ({self.end_idx}) cannot be greater than total files ({len(all_tsv_files)})")
        if self.start_idx >= self.end_idx:
            raise ValueError(f"start_idx ({self.start_idx}) must be less than end_idx ({self.end_idx})")
        
        # Select files in the specified range
        self.tsv_files = all_tsv_files[self.start_idx:self.end_idx]
        
        print(f"Found {len(all_tsv_files)} total TSV files")
        print(f"Processing files {self.start_idx} to {self.end_idx-1} ({len(self.tsv_files)} files)")
        print(f"File range: {self.tsv_files[0].name} to {self.tsv_files[-1].name}")
        
        # Load phylogenetic tree and OTU list if provided
        self.tree = None
        self.otu_list = None
        if tree_path and otu_list_path:
            self._load_tree_and_otu_list()
        else:
            print("Warning: No tree/OTU list provided. Will use simplified processing.")
    
    def _load_tree_and_otu_list(self):
        """Load phylogenetic tree and OTU list."""
        try:
            # Load tree
            self.tree = Graph()
            self.tree.build_graph(self.tree_path)
            
            # Load OTU list (handle the specific format from your data)
            if self.otu_list_path.endswith('.csv'):
                # Read the first line which contains comma-separated OTU names
                with open(self.otu_list_path, 'r') as f:
                    first_line = f.readline().strip()
                    self.otu_list = first_line.split(',')
            else:
                with open(self.otu_list_path, 'r') as f:
                    self.otu_list = [line.strip() for line in f if line.strip()]
            
            print(f"Loaded tree with {len(self.tree.nodes)} nodes")
            print(f"Loaded {len(self.otu_list)} OTUs")
            
        except Exception as e:
            print(f"Warning: Could not load tree/OTU list: {e}")
            print("Will use simplified processing without phylogenetic structure")
            self.tree = None
            self.otu_list = None
    
    def process_tsv_file(self, tsv_file, model, device, batch_size=32, output_dir=None, force_reprocess=False):
        """
        Process a single TSV file and extract embeddings immediately.
        
        Args:
            tsv_file: Path to TSV file
            model: Loaded GutCLIP model
            device: Device to run model on
            batch_size: Batch size for processing samples
            output_dir: Output directory to check for existing files
            force_reprocess: If True, reprocess even if output file exists
            
        Returns:
            dict: Embeddings and metadata for this file, or None if skipped
        """
        # Check if output file already exists (unless force reprocess is enabled)
        if output_dir is not None and not force_reprocess:
            output_dir = Path(output_dir)
            clean_file_name = tsv_file.name.replace('.tsv', '') if tsv_file.name.endswith('.tsv') else tsv_file.name
            output_path = output_dir / f"{clean_file_name}.pt"
            
            if output_path.exists():
                print(f"Skipping {tsv_file.name} - output file already exists: {output_path}")
                # Load existing data to return metadata
                try:
                    existing_data = torch.load(output_path)
                    return {
                        'tree_embeddings': existing_data['tree_embeddings'],
                        'sample_ids': existing_data['sample_ids'],
                        'file_name': existing_data['file_name'],
                        'taxa_counts': existing_data['taxa_counts'],
                        'has_tree': existing_data['has_tree'],
                        'embedding_dim': existing_data['embedding_dim'],
                        'num_samples': existing_data['num_samples'],
                        'skipped': True  # Mark as skipped
                    }
                except Exception as e:
                    print(f"Warning: Could not load existing file {output_path}: {e}")
                    print(f"Will reprocess {tsv_file.name}")
        
        print(f"Processing {tsv_file.name}...")
        
        try:
            # Read TSV file
            df = pd.read_csv(tsv_file, sep='\t', index_col=0)
            taxa_names = df.columns.tolist()
            
            print(f"  Found {len(df)} samples in {tsv_file.name}")
            
            # Process samples from this TSV file
            samples_data = []
            
            if self.tree is not None and self.otu_list is not None:
                # Process with tree structure
                if self.use_multiprocess:
                    samples_data = self._process_samples_with_tree_multiprocess(df, taxa_names, tsv_file.name)
                else:
                    samples_data = self._process_samples_with_tree_single_process(df, taxa_names, tsv_file.name)
            else:
                # Process without tree structure
                samples_data = self._process_samples_without_tree(df, taxa_names, tsv_file.name)
            
            if not samples_data:
                print(f"  No valid samples found in {tsv_file.name}")
                return None
            
            # Extract embeddings for this file's samples
            embeddings_data = self._extract_embeddings_for_samples(samples_data, model, device, batch_size)
            
            return embeddings_data
            
        except Exception as e:
            print(f"Error processing {tsv_file}: {e}")
            return None
    
    def _process_samples_with_tree_multiprocess(self, df, taxa_names, file_name):
        """Process samples with tree structure using multiprocessing."""
        print(f"  Processing {len(df)} samples with tree structure (multiprocess)...")
        
        # Build shared tree
        gtree = build_shared_tree(self.tree_path, self.otu_list)
        
        samples_data = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
            futs = []
            for idx, (sample_id, row) in enumerate(df.iterrows()):
                abundance_values = row.values.astype(np.float32)
                future = ex.submit(_process, idx, sample_id, abundance_values, taxa_names, gtree)
                futs.append((future, sample_id, abundance_values, file_name))
            
            for f, sample_id, abundance_values, file_name in tqdm(futs, desc=f"  Processing {file_name}", leave=False):
                d = f.result()
                if d is not None:
                    d.file_name = file_name
                    samples_data.append({
                        'data_obj': d,
                        'sample_id': sample_id,
                        'abundance': abundance_values,
                        'taxa_names': taxa_names,
                        'file_name': file_name,
                        'has_tree': True
                    })
        
        return samples_data
    
    def _process_samples_with_tree_single_process(self, df, taxa_names, file_name):
        """Process samples with tree structure using single process."""
        print(f"  Processing {len(df)} samples with tree structure (single process)...")
        
        # Build shared tree
        gtree = build_shared_tree(self.tree_path, self.otu_list)
        
        samples_data = []
        for idx, (sample_id, row) in tqdm(enumerate(df.iterrows()), total=len(df), desc=f"  Processing {file_name}", leave=False):
            try:
                abundance_values = row.values.astype(np.float32)
                d = _process(idx, sample_id, abundance_values, taxa_names, gtree)
                if d is not None:
                    d.file_name = file_name
                    samples_data.append({
                        'data_obj': d,
                        'sample_id': sample_id,
                        'abundance': abundance_values,
                        'taxa_names': taxa_names,
                        'file_name': file_name,
                        'has_tree': True
                    })
            except Exception as e:
                print(f"    Error processing sample {sample_id}: {e}")
                continue
        
        return samples_data
    
    def _process_samples_without_tree(self, df, taxa_names, file_name):
        """Process samples without tree structure."""
        print(f"  Processing {len(df)} samples without tree structure...")
        
        samples_data = []
        for sample_id, row in tqdm(df.iterrows(), total=len(df), desc=f"  Processing {file_name}", leave=False):
            abundance_values = row.values.astype(np.float32)
            
            # Create dummy tree data
            dummy_data = Data(
                x=torch.zeros(1, 9, dtype=torch.float32),
                pos=torch.zeros(1, 3, dtype=torch.float32),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                node_zero=torch.zeros(1, dtype=torch.bool),
                sample_index=0,
                otu_abundance=torch.tensor(abundance_values, dtype=torch.float),
                sample_id=sample_id,
                file_name=file_name
            )
            
            samples_data.append({
                'data_obj': dummy_data,
                'sample_id': sample_id,
                'abundance': abundance_values,
                'taxa_names': taxa_names,
                'file_name': file_name,
                'has_tree': False
            })
        
        return samples_data
    
    def _extract_embeddings_for_samples(self, samples_data, model, device, batch_size):
        """Extract embeddings for a list of samples."""
        if not samples_data:
            return None
        
        print(f"  Extracting embeddings for {len(samples_data)} samples...")
        
        tree_embeddings = []
        sample_ids = []
        taxa_counts = []
        has_tree_flags = []
        
        # Process in batches
        for i in range(0, len(samples_data), batch_size):
            batch_samples = samples_data[i:i + batch_size]
            
            with torch.no_grad():
                for sample_data in batch_samples:
                    try:
                        data_obj = sample_data['data_obj']
                        
                        # Move to device
                        data_obj.x = data_obj.x.to(device)
                        data_obj.pos = data_obj.pos.to(device)
                        data_obj.edge_index = data_obj.edge_index.to(device)
                        data_obj.node_zero = data_obj.node_zero.to(device)
                        
                        # Create batch tensor
                        batch_tensor = torch.zeros(data_obj.x.size(0), dtype=torch.long, device=device)
                        
                        # Create tree data dict
                        tree_data = {
                            "x": data_obj.x,
                            "edge_index": data_obj.edge_index,
                            "pos": data_obj.pos,
                            "batch": batch_tensor,
                            "node_zero": data_obj.node_zero
                        }
                        
                        # Extract tree embedding
                        tree_emb = model.encode_tree(tree_data, normalize=True)
                        
                        tree_embeddings.append(tree_emb.cpu())
                        sample_ids.append(sample_data['sample_id'])
                        taxa_counts.append(len(sample_data['taxa_names']))
                        has_tree_flags.append(sample_data['has_tree'])
                        
                    except Exception as e:
                        print(f"    Error extracting embedding for sample {sample_data['sample_id']}: {e}")
                        continue
        
        if not tree_embeddings:
            return None
        
        # Concatenate embeddings
        tree_embeddings = torch.cat(tree_embeddings, dim=0)
        
        return {
            'tree_embeddings': tree_embeddings,
            'sample_ids': sample_ids,
            'file_name': samples_data[0]['file_name'],
            'taxa_counts': taxa_counts,
            'has_tree': has_tree_flags,
            'embedding_dim': tree_embeddings.shape[1],
            'num_samples': len(sample_ids)
        }
    
    def process_all_files(self, model, device, output_dir, batch_size=32, force_reprocess=False):
        """
        Process all TSV files and save embeddings.
        
        Args:
            model: Loaded GutCLIP model
            device: Device to run model on
            output_dir: Directory to save embeddings
            batch_size: Batch size for processing samples
            force_reprocess: If True, reprocess even if output files exist
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        total_samples = 0
        total_files_processed = 0
        total_files_skipped = 0
        
        print(f"Processing {len(self.tsv_files)} TSV files...")
        if not force_reprocess:
            print("Note: Existing output files will be skipped. Use --force_reprocess to override.")
        
        for tsv_file in tqdm(self.tsv_files, desc="Processing TSV files"):
            # Process this TSV file and extract embeddings immediately
            embeddings_data = self.process_tsv_file(tsv_file, model, device, batch_size, output_dir, force_reprocess)
            
            if embeddings_data is not None:
                # Check if this file was skipped
                if embeddings_data.get('skipped', False):
                    total_files_skipped += 1
                    total_samples += embeddings_data['num_samples']
                    print(f"  Skipped {embeddings_data['num_samples']} embeddings (already exist)")
                else:
                    # Save embeddings for this file
                    clean_file_name = tsv_file.name.replace('.tsv', '') if tsv_file.name.endswith('.tsv') else tsv_file.name
                    output_path = output_dir / f"{clean_file_name}.pt"
                    torch.save(embeddings_data, output_path)
                    
                    total_samples += embeddings_data['num_samples']
                    total_files_processed += 1
                    
                    print(f"  Saved {embeddings_data['num_samples']} embeddings to {output_path}")
            else:
                print(f"  No embeddings extracted from {tsv_file.name}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"Files processed: {total_files_processed}/{len(self.tsv_files)}")
        print(f"Files skipped: {total_files_skipped}/{len(self.tsv_files)}")
        print(f"Total samples processed: {total_samples}")
        
        if total_samples > 0:
            # Get embedding dimension from first processed file
            first_pt_file = next(output_dir.glob("*.pt"), None)
            if first_pt_file:
                first_data = torch.load(first_pt_file)
                print(f"Embedding dimension: {first_data['embedding_dim']}")
            
            # Count samples with/without tree structure
            samples_with_tree = 0
            samples_without_tree = 0
            all_taxa_counts = []
            
            for pt_file in output_dir.glob("*.pt"):
                data = torch.load(pt_file)
                samples_with_tree += sum(data['has_tree'])
                samples_without_tree += len(data['has_tree']) - sum(data['has_tree'])
                all_taxa_counts.extend(data['taxa_counts'])
            
            print(f"Samples with tree structure: {samples_with_tree}")
            print(f"Samples without tree structure: {samples_without_tree}")
            if all_taxa_counts:
                print(f"Average taxa per sample: {np.mean(all_taxa_counts):.1f}")
                print(f"Min taxa per sample: {min(all_taxa_counts)}")
                print(f"Max taxa per sample: {max(all_taxa_counts)}")


def parse_args():
    parser = argparse.ArgumentParser(description='Extract taxa embeddings from count matrix data (streaming version)')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained GutCLIP model checkpoint')
    parser.add_argument('--cfg', type=str, required=True,
                      help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing count matrix TSV files')
    parser.add_argument('--output_dir', type=str, default='taxa_embeddings',
                      help='Directory to save embeddings')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers for data loading')
    parser.add_argument('--tree_path', type=str, default=None,
                      help='Path to phylogenetic tree file (Newick format)')
    parser.add_argument('--otu_list_path', type=str, default=None,
                      help='Path to OTU list file')
    parser.add_argument('--device', type=str, default='auto',
                      help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--no_multiprocess', action='store_true',
                      help='Disable multiprocessing (use single process)')
    parser.add_argument('--force_reprocess', action='store_true',
                      help='Force reprocessing of existing output files (skip check)')
    parser.add_argument('--start_idx', type=int, default=0,
                      help='Starting file index (0-based, default: 0)')
    parser.add_argument('--end_idx', type=int, default=None,
                      help='Ending file index (exclusive, default: None for all files)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load config
    cfg = OmegaConf.load(args.cfg)
    
    # Load model
    print("Loading GutCLIP model...")
    model = GutCLIPModel(
        tree_dim=cfg.tree_dim,
        dna_dim=cfg.dna_dim,
        output_dict=True
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.model_path}")
    ckpt_data = torch.load(args.model_path, map_location="cpu", weights_only=True)
    state_dict = ckpt_data.get("model", ckpt_data.get("state_dict", ckpt_data))
    clean_state = {k[6:] if k.startswith("model.") else k: v 
                  for k, v in state_dict.items()}
    model.load_state_dict(clean_state)
    model = model.eval().to(device)

    print(f"Model loaded successfully")
    
    # Create streaming dataset
    print("Creating streaming dataset...")
    dataset = StreamingTaxaDataset(
        data_dir=args.data_dir,
        tree_path=args.tree_path,
        otu_list_path=args.otu_list_path,
        num_workers=args.num_workers,
        use_multiprocess=not args.no_multiprocess,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )
    
    # Process all files with streaming approach
    print("Starting streaming processing...")
    dataset.process_all_files(
        model=model,
        device=device,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        force_reprocess=args.force_reprocess
    )
    
    print("Streaming processing completed!")


if __name__ == '__main__':
    main()
