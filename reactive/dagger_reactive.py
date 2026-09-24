#!/usr/bin/env python3
"""
DAgger campaign for the reactive Transformer.

Each iteration rolls out the current policy, queries the FORCESPRO expert
bank at the visited states, aggregates the new data with the BC data,
retrains and evaluates.

Usage:
    python dagger_reactive.py --bc_run_dir runs/reactive_ft \\
        --data_dir data/<dataset> --n_iters 5 --n_ics 100 \\
        --lr 1e-5 --epochs 30 --out_dir runs/dagger_reactive
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import json
import time
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict

import matplotlib
matplotlib.use('Agg')

# =====================================================================
# Project imports
# =====================================================================

from reactive_model import ReactiveTransformer, ReactiveConfig
from windowed_dataset import (
    WindowedTrajectoryDataset,
    ReactiveNormStats,
    compute_reactive_norm_stats,
    split_by_trajectory,
)
from train_reactive import train_one_epoch, validate
from forces_expert_bank import ForcesExpertBank
from diagnose_dagger_single import load_reactive_model

from ipopt_pdg import (
    build_dynamics_function, build_rk4_integrator,
    m_dry, m0,
    T_min_MN, T_max_MN,
    cos_tilt, theta_max,
)
from eval_reactive import (
    enforce_thrust_constraints,
    rollout_reactive,
    classify_landing as classify_landing_eval,
    ReactiveEvalConfig,
)


# =====================================================================
# Configuration
# =====================================================================

FILE_OFFSET = 40000       # DAgger trajectory numbering start
ITER_STRIDE = 10000       # file numbering slots per iteration
TARGET_DT   = 0.8         # subsampling target [s]
M_FLOOR     = m_dry + 500 # m_dry + m_overhead — reject below this


@dataclass
class DAggerReactiveConfig:
    """All campaign parameters in one place for reproducibility."""

    # --- Paths ---
    bc_run_dir: str = 'runs/reactive_finetune_v2'
    data_dir: str = 'data/ForcesLargeBatch'
    out_dir: str = 'runs/dagger_reactive_v1'
    solver_dir: str = '.'               # base dir for solver bank

    # --- DAgger loop ---
    n_iters: int = 5
    n_ics: int = 100                    # ICs per iteration (0 = all train)
    seed: int = 42

    # --- Rollout ---
    dt: float = 0.8                     # integration timestep [s]
    max_time: float = 80.0              # max rollout duration [s]

    # --- Training ---
    epochs: int = 30
    lr: float = 1e-5
    batch_size: int = 256
    weight_decay: float = 1e-5
    freeze_norm: bool = False   # If True, keep BC normalization stats throughout

    # --- Evaluation ---
    n_eval_ics: int = 20                # test ICs for quick CL eval

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)


# =====================================================================
# DAgger rollout with integrated expert queries
# =====================================================================

def dagger_rollout_with_expert(model, model_config, norm_stats, x0_7,
                                t_f_original, F_rk4, bank,
                                dt=0.8, max_time=80.0):
    """Roll out the reactive policy and query the expert bank at every step.

    Returns
    -------
    rollout : dict with policy trajectory
    expert_results : list of (step_k, expert_result_dict) for converged queries
    """
    W = model_config.window_size
    state_dim = len(norm_stats.state_mean)  # 7
    max_steps = int(max_time / dt)

    states = [x0_7.copy()]
    u_clipped_list = []
    history = [x0_7.astype(np.float32)]

    expert_results = []
    tf_estimate = t_f_original

    model.eval()

    for k in range(max_steps):
        x_k = states[-1]

        # --- Policy forward pass (matching eval_reactive.py exactly) ---
        n_available = len(history)
        window_raw = np.zeros((W, state_dim), dtype=np.float32)
        mask = np.ones(W, dtype=bool)   # True = padding

        if n_available >= W:
            window_raw[:] = np.array(history[-W:], dtype=np.float32)
            mask[:] = False
        else:
            # Left-pad with x_0, mark as padding
            n_pad = W - n_available
            window_raw[:n_pad] = np.array(history[0], dtype=np.float32)
            window_raw[n_pad:] = np.array(
                history[-n_available:], dtype=np.float32)
            mask[:n_pad] = True
            mask[n_pad:] = False

        window_norm = norm_stats.normalize_state(window_raw)
        window_t = torch.from_numpy(window_norm).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        with torch.no_grad():
            pred = model(window_t, padding_mask=mask_t)

        u_norm = pred[0].cpu().numpy()
        u_phys = norm_stats.unnormalize_action(u_norm)
        u_clip = enforce_thrust_constraints(u_phys)
        u_clipped_list.append(u_clip.copy())

        # --- Expert query ---
        if tf_estimate >= 2.0:
            result = bank.solve(x_k, tf_estimate=tf_estimate)

            if result['exitflag'] == 1:
                # Sanity check: reject if m_f < m_dry + overhead
                if result['m_f'] >= M_FLOOR:
                    expert_results.append((k, result))
                tf_estimate = max(result['tf'] - dt, 2.0)
            else:
                tf_estimate = max(tf_estimate - dt, 2.0)

        # --- Integrate dynamics with POLICY action ---
        x_next = np.array(F_rk4(x_k, u_clip, dt)).flatten()

        # Safety checks (matching eval_reactive.py)
        if x_next[2] < -5.0:
            states.append(x_next)
            break
        if x_next[6] <= m_dry:
            states.append(x_next)
            break
        if np.any(np.isnan(x_next)):
            states.append(x_next)
            break

        states.append(x_next)
        history.append(x_next.astype(np.float32))

    K = len(u_clipped_list)
    rollout = {
        'x_traj': np.array(states),           # (K+1, 7)
        'u_clipped': np.array(u_clipped_list), # (K, 3)
        'n_steps': K,
        'r_z_final': float(states[-1][2]),
        'v_z_final': float(states[-1][5]),
    }
    return rollout, expert_results


# =====================================================================
# Subsample and save expert trajectory
# =====================================================================

def subsample_and_save(expert_result, filepath, target_dt=TARGET_DT):
    """Subsample a FORCESPRO expert trajectory to match deployment dt, strip
    the t_f column, and save as NPZ in the 6-key schema.
    """
    x_full = expert_result['x_traj']      # (n_stages, 8): [r, v, m, t_f]
    u_full = expert_result['u_traj']      # (n_stages-1, 3): MN
    tf_opt = expert_result['tf']
    n_stages = expert_result['n_stages']
    n_int = n_stages - 1

    original_dt = tf_opt / n_int
    stride = max(1, round(target_dt / original_dt))

    # State indices: [0, stride, 2*stride, ...] + terminal
    state_idx = list(range(0, n_stages, stride))
    if state_idx[-1] != n_stages - 1:
        state_idx.append(n_stages - 1)
    control_idx = state_idx[:-1]

    # Strip t_f column — keep only [r, v, m] (first 7 channels)
    x_sub = x_full[state_idx, :7].astype(np.float64)    # (K+1, 7)
    u_sub = u_full[control_idx].astype(np.float64)       # (K, 3)

    np.savez(
        filepath,
        x_traj=x_sub,
        u_traj=u_sub,
        ic=x_sub[0],
        status=0,
        t_f=float(tf_opt),
        m_f=float(x_sub[-1, 6]),
    )
    return True


# =====================================================================
# Dataset assembly
# =====================================================================

def build_combined_files(data_dir, out_dir, iteration, seed=42):
    """Gather BC training files + all DAgger files from iterations 0..i."""
    # BC files — same split as the BC training run
    bc_dir = Path(data_dir)
    all_bc = sorted(bc_dir.glob('traj_*.npz'))
    bc_train, bc_val, bc_test = split_by_trajectory(all_bc, seed=seed)

    # DAgger files from all completed iterations
    dagger_files = []
    for i in range(iteration + 1):
        trajs_dir = Path(out_dir) / f'iter_{i}' / 'trajs'
        if trajs_dir.exists():
            dagger_files.extend(sorted(trajs_dir.glob('traj_*.npz')))

    combined_train = bc_train + dagger_files

    print(f"  Dataset: {len(bc_train)} BC train + {len(dagger_files)} DAgger "
          f"= {len(combined_train)} total train, "
          f"{len(bc_val)} val, {len(bc_test)} test")

    return combined_train, bc_val, bc_test


# =====================================================================
# Training step
# =====================================================================

def train_iteration(model, combined_train_files, val_files,
                    window_size, cfg, device, frozen_norm=None):
    """Fine-tune the model on the combined dataset."""
    if frozen_norm is not None:
        print("  Using frozen BC normalization stats")
        norm_stats = frozen_norm
    else:
        print("  Recomputing normalization stats...")
        norm_stats = compute_reactive_norm_stats(combined_train_files)

    # Build datasets
    train_ds = WindowedTrajectoryDataset(
        combined_train_files, norm_stats, window_size, filter_status=0)
    val_ds = WindowedTrajectoryDataset(
        val_files, norm_stats, window_size, filter_status=0)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, pin_memory=True)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, pin_memory=True)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    best_val_loss = float('inf')
    best_state = None
    history = []

    print(f"  Training: {len(train_ds)} windows, {cfg.epochs} epochs, "
          f"lr={cfg.lr}")

    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        scheduler.step(val_loss)

        history.append({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'lr': float(optimizer.param_groups[0]['lr']),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == cfg.epochs - 1:
            print(f"    Epoch {epoch+1:3d}/{cfg.epochs}: "
                  f"train={train_loss:.6f}, val={val_loss:.6f}, "
                  f"best={best_val_loss:.6f}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return norm_stats, best_val_loss, history


# =====================================================================
# Quick closed-loop evaluation
# =====================================================================


def quick_cl_eval(model, model_config, norm_stats, test_files, F_rk4,
                  n_eval=20, dt=0.8):
    """Quick closed-loop evaluation on test ICs."""
    model.eval()
    eval_cfg = ReactiveEvalConfig(dt=dt, t_max=80.0)

    n_eval = min(n_eval, len(test_files))
    selected = test_files[:n_eval]

    counts = {'soft': 0, 'hard': 0, 'crash': 0, 'airborne': 0}

    for ic_file in selected:
        data = np.load(ic_file)
        x0 = data['x_traj'][0, :7].astype(np.float64)

        result = rollout_reactive(
            model, norm_stats, x0, model_config.window_size,
            F_rk4, eval_cfg)

        cls = result['classification']
        if cls in counts:
            counts[cls] += 1

    total = sum(counts.values())
    pcts = {k: 100.0 * v / max(total, 1) for k, v in counts.items()}
    print(f"  CL eval ({total} ICs): "
          f"soft={counts['soft']} ({pcts['soft']:.0f}%), "
          f"hard={counts['hard']} ({pcts['hard']:.0f}%), "
          f"crash={counts['crash']} ({pcts['crash']:.0f}%), "
          f"airborne={counts['airborne']} ({pcts['airborne']:.0f}%)")

    return {'counts': counts, 'percentages': pcts, 'n_eval': total}


# =====================================================================
# One DAgger iteration
# =====================================================================

def run_one_iteration(iteration, cfg, model, model_config, norm_stats,
                      train_ic_files, val_files, test_files,
                      F_rk4, bank, device, bc_norm=None):
    """One complete DAgger iteration: rollout → query → save → train → eval."""
    print(f"\n{'='*70}")
    print(f"DAGGER ITERATION {iteration}")
    print(f"{'='*70}")
    t_iter_start = time.time()

    iter_dir = Path(cfg.out_dir) / f'iter_{iteration}'
    trajs_dir = iter_dir / 'trajs'
    model_dir = iter_dir / 'model'
    trajs_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Select ICs ---
    rng = np.random.default_rng(cfg.seed + iteration)
    if 0 < cfg.n_ics < len(train_ic_files):
        selected_idx = rng.choice(
            len(train_ic_files), size=cfg.n_ics, replace=False)
        ic_files = [train_ic_files[i] for i in selected_idx]
    else:
        ic_files = train_ic_files

    print(f"\n  Step 1: Rolling out on {len(ic_files)} ICs "
          f"+ querying expert bank...")

    # --- 2. Rollout + expert queries ---
    file_counter = 0
    file_offset = FILE_OFFSET + iteration * ITER_STRIDE

    total_steps = 0
    total_queries_ok = 0
    total_rollout_steps = 0

    for ic_idx, ic_file in enumerate(ic_files):
        data = np.load(ic_file)
        x_traj_bc = data['x_traj']
        x0 = x_traj_bc[0, :7].copy()
        t_f = float(data['t_f'])

        rollout, expert_results = dagger_rollout_with_expert(
            model, model_config, norm_stats, x0, t_f, F_rk4, bank,
            dt=cfg.dt, max_time=cfg.max_time)

        total_rollout_steps += rollout['n_steps']
        total_queries_ok += len(expert_results)

        # Save converged expert trajectories
        for step_k, result in expert_results:
            file_idx = file_offset + file_counter
            filepath = trajs_dir / f'traj_{file_idx:05d}.npz'
            saved = subsample_and_save(result, filepath)
            if saved:
                file_counter += 1

        # Progress
        if (ic_idx + 1) % 10 == 0 or ic_idx == len(ic_files) - 1:
            conv_rate = 100 * total_queries_ok / max(total_rollout_steps, 1)
            print(f"    IC {ic_idx+1:3d}/{len(ic_files)}: "
                  f"{file_counter} trajs saved, "
                  f"{total_queries_ok}/{total_rollout_steps} expert "
                  f"({conv_rate:.0f}%)")

    t_rollout = time.time() - t_iter_start

    print(f"\n  Rollout + expert queries: {t_rollout:.0f}s")
    print(f"  Saved {file_counter} expert trajectories to {trajs_dir}/")

    if file_counter == 0:
        print("  WARNING: no expert trajectories saved — skipping training.")
        return {'iteration': iteration, 'n_trajs_saved': 0, 'skipped': True}

    # --- 3. Build combined dataset ---
    print(f"\n  Step 2: Building combined dataset...")
    combined_train, bc_val, bc_test = build_combined_files(
        cfg.data_dir, cfg.out_dir, iteration, seed=cfg.seed)

    # --- 4. Train ---
    print(f"\n  Step 3: Fine-tuning...")
    t_train_start = time.time()
    norm_stats_new, best_val, history = train_iteration(
        model, combined_train, bc_val,
        model_config.window_size, cfg, device, frozen_norm=bc_norm 
        if cfg.freeze_norm else None)
    t_train = time.time() - t_train_start
    print(f"  Training: {t_train:.0f}s, best val_loss={best_val:.6f}")

    # --- 5. Save model + config + norm stats ---
    torch.save({
        'model_state_dict': model.state_dict(),
        'val_loss': float(best_val),
        'iteration': iteration,
    }, model_dir / 'best_model.pt')
    model_config.save(model_dir / 'model_config.json')
    norm_stats_new.save(model_dir / 'norm_stats.npz')

    # --- 6. Quick CL evaluation ---
    print(f"\n  Step 4: Quick closed-loop evaluation...")
    eval_results = quick_cl_eval(
        model, model_config, norm_stats_new, test_files, F_rk4,
        n_eval=cfg.n_eval_ics, dt=cfg.dt)

    # --- 7. Log ---
    t_iter_total = time.time() - t_iter_start
    iter_log = {
        'iteration': iteration,
        'n_ics': len(ic_files),
        'n_rollout_steps': int(total_rollout_steps),
        'n_expert_converged': int(total_queries_ok),
        'expert_conv_rate': float(total_queries_ok / max(total_rollout_steps, 1)),
        'n_trajs_saved': file_counter,
        'best_val_loss': float(best_val),
        'eval': eval_results,
        'time_rollout_s': float(t_rollout),
        'time_train_s': float(t_train),
        'time_total_s': float(t_iter_total),
        'train_history': history,
    }

    with open(iter_dir / 'iter_log.json', 'w') as f:
        json.dump(iter_log, f, indent=2)

    print(f"\n  Iteration {iteration} complete in {t_iter_total/60:.1f} min")
    bank.print_stats()
    bank.reset_stats()

    return iter_log, norm_stats_new


# =====================================================================
# Full campaign
# =====================================================================

def run_campaign(cfg):
    """Run the full DAgger campaign."""
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / 'config.json')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Build dynamics integrator ---
    print("\nBuilding CasADi RK4 integrator...")
    f_dyn = build_dynamics_function()
    F_rk4 = build_rk4_integrator(f_dyn)

    # --- Load expert bank ---
    print("\nLoading FORCESPRO expert bank...")
    bank = ForcesExpertBank(solver_dir=cfg.solver_dir)

    # --- Load BC splits (fixed across all iterations) ---
    print(f"\nLoading BC data from {cfg.data_dir}...")
    bc_dir = Path(cfg.data_dir)
    all_bc = sorted(bc_dir.glob('traj_*.npz'))
    train_ic_files, val_files, test_files = split_by_trajectory(
        all_bc, seed=cfg.seed)

    # --- Load initial policy ---
    print(f"\nLoading initial policy from {cfg.bc_run_dir}...")
    model, model_config, norm_stats = load_reactive_model(cfg.bc_run_dir)
    model.to(device)
    bc_norm = norm_stats

    # --- Campaign loop ---
    campaign_log = []
    t_campaign_start = time.time()

    for iteration in range(cfg.n_iters):
        result = run_one_iteration(
            iteration, cfg, model, model_config, norm_stats,
            train_ic_files, val_files, test_files,
            F_rk4, bank, device, bc_norm=bc_norm)

        if isinstance(result, tuple):
            iter_log, norm_stats = result
        else:
            iter_log = result
            # norm_stats unchanged if iteration was skipped

        campaign_log.append(iter_log)

        # Save campaign log after each iteration (incremental)
        with open(out_dir / 'campaign_log.json', 'w') as f:
            json.dump(campaign_log, f, indent=2)


    # --- Campaign summary ---
    t_campaign = time.time() - t_campaign_start
    print(f"\n{'='*70}")
    print(f"DAGGER CAMPAIGN COMPLETE")
    print(f"{'='*70}")
    print(f"  Iterations:      {cfg.n_iters}")
    print(f"  Total time:      {t_campaign/60:.1f} min")
    print(f"  Output:          {out_dir}/")

    print(f"\n  Per-iteration summary:")
    print(f"  {'Iter':>4s}  {'Trajs':>6s}  {'Conv%':>6s}  "
          f"{'ValLoss':>9s}  {'Soft%':>6s}  {'Hard%':>6s}  "
          f"{'Crash%':>6s}  {'Air%':>6s}")
    print(f"  {'-'*60}")

    for log in campaign_log:
        if log.get('skipped'):
            print(f"  {log['iteration']:4d}  {'SKIPPED':>6s}")
            continue
        ev = log['eval']['percentages']
        print(f"  {log['iteration']:4d}  {log['n_trajs_saved']:6d}  "
              f"{100*log['expert_conv_rate']:5.1f}%  "
              f"{log['best_val_loss']:9.6f}  "
              f"{ev['soft']:5.1f}%  {ev['hard']:5.1f}%  "
              f"{ev['crash']:5.1f}%  {ev['airborne']:5.1f}%")

    print(f"\n  Next: run full evaluation on the best iteration:")
    print(f"    python eval_reactive.py \\")
    print(f"      --run_dir {out_dir}/iter_<best>/model \\")
    print(f"      --data_dir {cfg.data_dir}")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='DAgger campaign for reactive Transformer (FORCESPRO)',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--bc_run_dir', type=str,
                        default='runs/reactive_finetune_v2',
                        help='Path to initial BC model run directory')
    parser.add_argument('--data_dir', type=str,
                        default='data/ForcesLargeBatch',
                        help='Path to BC trajectory data')
    parser.add_argument('--out_dir', type=str,
                        default='runs/dagger_reactive_v1',
                        help='Output directory for DAgger campaign')
    parser.add_argument('--solver_dir', type=str, default='.',
                        help='Directory containing solver bank folders')

    parser.add_argument('--n_iters', type=int, default=5)
    parser.add_argument('--n_ics', type=int, default=100,
                        help='ICs per iteration (0 = all train)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_eval_ics', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--freeze_norm', action='store_true',
                    help='Freeze normalization to BC stats (skip recomputation)')

    args = parser.parse_args()

    cfg = DAggerReactiveConfig(
        bc_run_dir=args.bc_run_dir,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        solver_dir=args.solver_dir,
        n_iters=args.n_iters,
        n_ics=args.n_ics,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        n_eval_ics=args.n_eval_ics,
        seed=args.seed,
        freeze_norm=args.freeze_norm,
    )

    print("RETALT1 PDG — DAgger Campaign (Reactive Transformer)")
    print(f"  BC model:    {cfg.bc_run_dir}")
    print(f"  BC data:     {cfg.data_dir}")
    print(f"  Output:      {cfg.out_dir}")
    print(f"  Iterations:  {cfg.n_iters}")
    print(f"  ICs/iter:    {cfg.n_ics}")
    print(f"  Epochs/iter: {cfg.epochs}")
    print(f"  LR:          {cfg.lr}")
    print()

    run_campaign(cfg)


if __name__ == '__main__':
    main()
