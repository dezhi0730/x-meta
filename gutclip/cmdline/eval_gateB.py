#!/usr/bin/env python3
"""
Gate B Evaluation Command Line Interface
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import hydra
from omegaconf import DictConfig

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gutclip.evaluate.eval_gateB import evaluate_gateB, plot_gateB_results
from gutclip.diffusion.schedulers import get_scheduler


def load_model_and_data(cfg: DictConfig, checkpoint_path: str):
    """Load model, data loader, and scheduler."""
    # Import here to avoid circular imports
    from gutclip.cmdline.train_gen_model import setup_model, setup_data
    
    # Setup model
    model = setup_model(cfg)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Setup data
    val_loader = setup_data(cfg, split='val')
    
    # Setup scheduler
    scheduler = get_scheduler(
        cfg.train.scheduler_type,
        cfg.train.num_timesteps
    )
    
    return model, val_loader, scheduler


@hydra.main(version_base=None, config_path="../configs", config_name="train_gen_model")
def main(cfg: DictConfig):
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Gate B: Residual Calibration & Sampling Stability")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="logs/gateB", help="Output directory")
    parser.add_argument("--max_batches", type=int, default=20, help="Maximum batches to evaluate")
    parser.add_argument("--num_ddim_steps", type=int, default=50, help="Number of DDIM steps for trajectory test")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--use_autocast", action="store_true", help="Use automatic mixed precision")
    
    # Parse Hydra overrides
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and data
    print(f"Loading model from {args.checkpoint}...")
    model, val_loader, scheduler = load_model_and_data(cfg, args.checkpoint)
    
    # Move to device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"Running Gate B evaluation on {device}...")
    print(f"Max batches: {args.max_batches}")
    print(f"DDIM steps: {args.num_ddim_steps}")
    print(f"Use autocast: {args.use_autocast}")
    
    # Run evaluation
    verdict, raw_metrics, suggestions = evaluate_gateB(
        model=model,
        val_loader=val_loader,
        scheduler=scheduler,
        device=device,
        max_batches=args.max_batches,
        use_autocast=args.use_autocast,
        num_ddim_steps=args.num_ddim_steps,
    )
    
    # Save results
    results = {
        "verdict": verdict,
        "suggestions": suggestions,
        "config": {
            "checkpoint": args.checkpoint,
            "max_batches": args.max_batches,
            "num_ddim_steps": args.num_ddim_steps,
            "use_autocast": args.use_autocast,
        }
    }
    
    results_path = os.path.join(args.output_dir, "gateB_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Print results
    print("\n" + "="*60)
    print("GATE B EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nPassed: {verdict['passed']}")
    print(f"Overall Score: {verdict['scores']['overall']:.3f}")
    print(f"Residual Calibration Score: {verdict['scores']['residual_calibration']:.3f}")
    print(f"Trajectory Stability Score: {verdict['scores']['trajectory_stability']:.3f}")
    
    if verdict["issues"]:
        print(f"\nIssues Found:")
        for issue in verdict["issues"]:
            print(f"  - {issue}")
    
    print(f"\nSuggestions:")
    for suggestion in suggestions:
        print(f"  {suggestion}")
    
    # Generate plots if requested
    if args.plot:
        plot_path = os.path.join(args.output_dir, "gateB_plots.png")
        plot_gateB_results(raw_metrics, plot_path, "Gate B")
        print(f"\nPlots saved to: {plot_path}")
    
    print(f"\nDetailed results saved to: {results_path}")
    print("="*60)
    
    # Exit with appropriate code
    sys.exit(0 if verdict["passed"] else 1)


if __name__ == "__main__":
    main() 