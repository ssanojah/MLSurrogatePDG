"""
Behavioural-cloning training of the reactive Transformer.

Usage:
    python train_reactive.py data/<dataset> --run_name reactive_w16 -w 16
    python train_reactive.py data/<dataset> --run_name ft \\
        --resume_from runs/reactive_w16 --lr 1e-5
"""

import numpy as np
import torch
import torch.nn as nn
import argparse
import json
import time
import csv
from pathlib import Path
from dataclasses import dataclass, asdict

from windowed_dataset import (
    create_windowed_dataloaders,
    ReactiveNormStats,
    STATE_NAMES,
    ACTION_NAMES,
)
from reactive_model import ReactiveTransformer, ReactiveConfig


# =====================================================================
# Training configuration
# =====================================================================

@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # Data
    window_size: int = 16
    batch_size: int = 256

    # Optimizer
    lr: float = 1e-4
    weight_decay: float = 1e-5

    # LR schedule (ReduceLROnPlateau)
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    min_lr: float = 1e-6

    # Training loop
    epochs: int = 200
    grad_clip: float = 1.0
    early_stop_patience: int = 25    # lower than full-sequence BC (30)

    # Reproducibility
    seed: int = 42

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)


# =====================================================================
# Training and validation steps
# =====================================================================

def train_one_epoch(model, dataloader, optimizer, device):
    """One training epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        window = batch['window'].to(device)        # (B, W, 7)
        label = batch['label'].to(device)           # (B, 3)
        padding_mask = batch['mask'].to(device)     # (B, W)

        # Forward
        pred = model(window, padding_mask=padding_mask)  # (B, 3)

        # Loss — simple MSE, no sequence masking needed
        loss = nn.functional.mse_loss(pred, label)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate(model, dataloader, device):
    """Validation pass."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            window = batch['window'].to(device)
            label = batch['label'].to(device)
            padding_mask = batch['mask'].to(device)

            pred = model(window, padding_mask=padding_mask)
            loss = nn.functional.mse_loss(pred, label)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


# =====================================================================
# Test evaluation — per-component RMSE in physical units
# =====================================================================

def evaluate_test(model, dataloader, norm_stats, device):
    """Evaluate on test set."""
    model.eval()
    all_errors = []  # accumulate (pred - expert) in physical units

    with torch.no_grad():
        for batch in dataloader:
            window = batch['window'].to(device)
            label = batch['label'].to(device)

            pred_norm = model(window, padding_mask=batch['mask'].to(device))

            # Denormalize to physical units (MN)
            pred_phys = norm_stats.unnormalize_action(pred_norm.cpu().numpy())
            label_phys = norm_stats.unnormalize_action(label.cpu().numpy())

            errors = pred_phys - label_phys   # (B, 3) in MN
            all_errors.append(errors)

    all_errors = np.concatenate(all_errors, axis=0)  # (N_test, 3)

    # Per-component statistics (convert MN → kN for readability)
    rmse_kn = np.sqrt((all_errors ** 2).mean(axis=0)) * 1000   # (3,)
    mae_kn = np.abs(all_errors).mean(axis=0) * 1000             # (3,)
    max_err_kn = np.abs(all_errors).max(axis=0) * 1000           # (3,)

    # Overall RMSE as % of thrust range [472, 1179] kN → range = 707 kN
    thrust_range_kn = 707.0
    overall_rmse_kn = np.sqrt((all_errors ** 2).mean()) * 1000
    rmse_pct = overall_rmse_kn / thrust_range_kn * 100

    # Normalized MSE (for comparison with training loss)
    label_all = []
    pred_all = []
    with torch.no_grad():
        for batch in dataloader:
            window = batch['window'].to(device)
            label = batch['label'].to(device)
            pred = model(window, padding_mask=batch['mask'].to(device))
            label_all.append(label.cpu())
            pred_all.append(pred.cpu())
    label_all = torch.cat(label_all, dim=0)
    pred_all = torch.cat(pred_all, dim=0)
    test_mse_norm = nn.functional.mse_loss(pred_all, label_all).item()

    return {
        'rmse_kn': rmse_kn,
        'mae_kn': mae_kn,
        'max_err_kn': max_err_kn,
        'overall_rmse_kn': overall_rmse_kn,
        'rmse_pct': rmse_pct,
        'test_mse_norm': test_mse_norm,
        'n_samples': all_errors.shape[0],
    }


# =====================================================================
# Main training function
# =====================================================================

def train(data_dir: str, run_name: str, train_cfg: TrainConfig,
          model_cfg: ReactiveConfig, resume_from: str = None):
    """Full training pipeline: load data → train → evaluate → save."""
    data_dir = Path(data_dir)
    run_dir = Path('runs') / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Reproducibility ---
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    # --- Load data ---
    print(f"\nLoading data from {data_dir} (W={train_cfg.window_size})...")
    t0 = time.time()
    train_loader, val_loader, test_loader, norm_stats = \
        create_windowed_dataloaders(
            data_dir,
            window_size=train_cfg.window_size,
            batch_size=train_cfg.batch_size,
            seed=train_cfg.seed,
        )
    t_data = time.time() - t0
    print(f"Data loaded in {t_data:.1f}s\n")

    # --- Create model ---
    model_cfg.window_size = train_cfg.window_size  # ensure consistency
    model = ReactiveTransformer(model_cfg).to(device)

    # --- Resume from checkpoint if specified ---
    if resume_from is not None:
        resume_dir = Path(resume_from)
        resume_path = resume_dir / 'best_model.pt'
        if not resume_path.exists():
            resume_path = resume_dir / 'final_model.pt'
        if not resume_path.exists():
            raise FileNotFoundError(f"No model found in {resume_dir}")

        checkpoint = torch.load(resume_path, weights_only=True,
                                map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  RESUMED from {resume_path}")
        print(f"  (epoch {checkpoint.get('epoch', '?')}, "
              f"val_loss={checkpoint.get('val_loss', '?')})")
        print(f"  Fine-tuning with lr={train_cfg.lr}")
    else:
        print("  Training from scratch")

    print(model.summary())
    print()

    # --- Optimizer and scheduler ---
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=train_cfg.scheduler_factor,
        patience=train_cfg.scheduler_patience,
        min_lr=train_cfg.min_lr,
    )

    # --- Save configs before training ---
    train_cfg.save(run_dir / 'train_config.json')
    model_cfg.save(run_dir / 'model_config.json')
    norm_stats.save(run_dir / 'norm_stats.npz')

    # --- Training loop ---
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    n_train_batches = len(train_loader)
    print(f"  Batches/epoch: {n_train_batches}")
    print(f"  Grad steps/epoch: {n_train_batches}")
    print(f"  Max epochs: {train_cfg.epochs}")
    print(f"  Early stopping patience: {train_cfg.early_stop_patience}")
    print()

    best_val_loss = float('inf')
    best_epoch = -1
    epochs_no_improve = 0
    training_log = []
    t_train_start = time.time()

    # CSV log for real-time monitoring
    log_path = run_dir / 'training_log.csv'
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss',
                                                'val_loss', 'lr', 'time_s'])
        writer.writeheader()

    for epoch in range(train_cfg.epochs):
        t_epoch = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_loss = validate(model, val_loader, device)

        # LR schedule
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        epoch_time = time.time() - t_epoch

        # Log
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': current_lr,
            'time_s': epoch_time,
        }
        training_log.append(log_entry)

        # Append to CSV (so you can monitor in real-time)
        with open(log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss',
                                                    'val_loss', 'lr', 'time_s'])
            writer.writerow(log_entry)

        import math
        if not math.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, run_dir / 'best_model.pt')
        else:
            epochs_no_improve += 1
            if math.isnan(val_loss) and epoch == 0:
                print("  WARNING: val_loss is NaN — check mask types")

        # Progress — every 5 epochs, or first/last
        if epoch % 5 == 0 or epoch == train_cfg.epochs - 1 or \
                epochs_no_improve == train_cfg.early_stop_patience:
            elapsed = time.time() - t_train_start
            print(f"  Epoch {epoch:3d}/{train_cfg.epochs}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"lr={current_lr:.1e}  "
                  f"best={best_val_loss:.6f}@{best_epoch}  "
                  f"({epoch_time:.1f}s/epoch, {elapsed/60:.1f}min total)")

        # Early stopping
        if epochs_no_improve >= train_cfg.early_stop_patience:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {train_cfg.early_stop_patience} epochs)")
            break

    t_train_total = time.time() - t_train_start
    print(f"\nTraining complete: {epoch + 1} epochs in {t_train_total/60:.1f} min")
    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")

    # Save final model too (for diagnostics)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
    }, run_dir / 'final_model.pt')

    # --- Load best model for test evaluation ---
    print(f"\n{'=' * 60}")
    print("TEST EVALUATION (best model)")
    print(f"{'=' * 60}")

    best_path = run_dir / 'best_model.pt'
    final_path = run_dir / 'final_model.pt'
    if best_path.exists():
        checkpoint = torch.load(best_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded best model from epoch {checkpoint['epoch']}")
    elif final_path.exists():
        checkpoint = torch.load(final_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  WARNING: best_model.pt not found, using final model")
    else:
        print("  ERROR: no saved model found — skipping test evaluation")
        return run_dir

    if test_loader is not None:
        test_results = evaluate_test(model, test_loader, norm_stats, device)

        # Same format as the full-sequence BC test report
        print(f"\n  Test MSE (normalized): {test_results['test_mse_norm']:.6f}")
        print(f"  Test samples: {test_results['n_samples']:,}")
        print(f"\n  Per-component RMSE (physical units):")
        print(f"  {'Component':>8s}  {'RMSE [kN]':>10s}  {'MAE [kN]':>10s}  "
              f"{'Max err [kN]':>12s}")
        print(f"  {'-' * 46}")
        for i, name in enumerate(ACTION_NAMES):
            print(f"  {name:>8s}  {test_results['rmse_kn'][i]:10.1f}  "
                  f"{test_results['mae_kn'][i]:10.1f}  "
                  f"{test_results['max_err_kn'][i]:12.1f}")
        print(f"\n  Overall RMSE: {test_results['overall_rmse_kn']:.1f} kN "
              f"({test_results['rmse_pct']:.2f}% of thrust range)")

        # Save test report
        with open(run_dir / 'test_report.json', 'w') as f:
            json.dump({
                'test_mse_norm': test_results['test_mse_norm'],
                'rmse_kn': test_results['rmse_kn'].tolist(),
                'mae_kn': test_results['mae_kn'].tolist(),
                'max_err_kn': test_results['max_err_kn'].tolist(),
                'overall_rmse_kn': float(test_results['overall_rmse_kn']),
                'rmse_pct': float(test_results['rmse_pct']),
                'n_samples': test_results['n_samples'],
                'best_epoch': best_epoch,
                'best_val_loss': float(best_val_loss),
                'total_epochs': epoch + 1,
                'training_time_min': t_train_total / 60,
            }, f, indent=2)

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("OUTPUT FILES")
    print(f"{'=' * 60}")
    print(f"  Run directory:    {run_dir}")
    print(f"  Best model:       {run_dir / 'best_model.pt'}")
    print(f"  Final model:      {run_dir / 'final_model.pt'}")
    print(f"  Model config:     {run_dir / 'model_config.json'}")
    print(f"  Train config:     {run_dir / 'train_config.json'}")
    print(f"  Norm stats:       {run_dir / 'norm_stats.npz'}")
    print(f"  Training log:     {run_dir / 'training_log.csv'}")
    if test_loader is not None:
        print(f"  Test report:      {run_dir / 'test_report.json'}")

    return run_dir


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train reactive sliding-window Transformer for PDG',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Required
    parser.add_argument('data_dir', type=str,
                        help='Directory with traj_*.npz files')

    # Run management
    parser.add_argument('--run_name', type=str, default='reactive_001',
                        help='Name for this run (default: reactive_001)')

    # Window size (the main ablation variable)
    parser.add_argument('-w', '--window_size', type=int, default=16,
                        help='Sliding window size W (default: 16)')

    # Resume / fine-tune from existing model
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to a previous run dir to resume from. '
                             'Loads best_model.pt weights as starting point. '
                             'Use with --lr 1e-5 --epochs 50 for fine-tuning.')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)

    # Model architecture (usually leave at defaults)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)

    args = parser.parse_args()

    # Build configs from CLI args
    train_cfg = TrainConfig(
        window_size=args.window_size,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
    )

    model_cfg = ReactiveConfig(
        window_size=args.window_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    )

    print("RETALT1 PDG — Reactive Transformer Training")
    print(f"  Data:        {args.data_dir}")
    print(f"  Run:         runs/{args.run_name}")
    print(f"  Window size: {args.window_size}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  LR:          {args.lr}")
    print(f"  Max epochs:  {args.epochs}")
    print(f"  Seed:        {args.seed}")
    print(f"  d_model:     {args.d_model}, heads={args.nhead}, "
          f"layers={args.num_layers}")
    if args.resume_from:
        print(f"  RESUME FROM: {args.resume_from}")
    print()

    train(args.data_dir, args.run_name, train_cfg, model_cfg,
          resume_from=args.resume_from)


if __name__ == '__main__':
    main()
