#!/usr/bin/env python3
"""
Train the PDGTransformer with a time-to-go input channel (BC or BC + DAgger).

Reads a packed dataset (pack_dataset.py) and, optionally, DAgger splice
files (dagger_splice.py). Writes model_ttg.pt, norm_ttg.npz,
model_config.json, config.json and training_log.json to --out_dir.

Usage:
    python ttg_train_v2.py --packed_file data/packed_dataset.npz \\
        --late_weight 5.0 --seed 42 --out_dir runs/ttg_late_w5.0_seed42

    python ttg_train_v2.py --packed_file data/packed_dataset.npz \\
        --dagger_file data/dagger_iter001/dagger_splices.npz \\
        --late_weight 5.0 --seed 42 --out_dir runs/dag1_seed42
"""

import argparse
import json
import os
import time
from pathlib import Path

# ---------------------------------------------------------------------
# THREAD CONTROL, set before numpy/torch are imported.
# ---------------------------------------------------------------------
_THREADS = os.environ.get('PDG_THREADS', '4')
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, _THREADS)

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from model import PDGTransformer, TransformerConfig
from dataset import NormalizationStats
from ipopt_pdg import N as N_STEPS


# =====================================================================
# 1. DATA
# =====================================================================

def load_packed_dataset(packed_file):
    """Return (x_traj, u_traj, t_f, status) from a pack_dataset.py file."""
    print(f"Loading packed dataset: {packed_file}")
    t0 = time.time()
    d = np.load(packed_file)
    out = (d['x_traj'], d['u_traj'], d['t_f'], d['status'])
    print(f"  Loaded {out[0].shape[0]} trajectories in "
          f"{time.time()-t0:.1f}s")
    return out


def split_indices(M, seed, train_frac=0.70, val_frac=0.15):
    """Same permutation as dataset.split_by_ic, applied to array indices."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(M)
    n_tr = max(1, int(M * train_frac))
    n_va = max(1, int(M * val_frac))
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


def build_ttg_states(x_traj, t_f):
    """(M, 60, 8) inputs with channel 7 = t_f*(N-k)/N = (N-k)*dt."""
    M = x_traj.shape[0]
    S = np.zeros((M, N_STEPS, 8), dtype=np.float32)
    S[:, :, :7] = x_traj[:, :N_STEPS, :7]
    fr = np.array([(N_STEPS - k) / N_STEPS for k in range(N_STEPS)],
                  dtype=np.float32)
    S[:, :, 7] = t_f[:, None] * fr[None, :]
    return S


def load_dagger_files(paths):
    """Concatenate splice NPZs -> (states, actions, mask)."""
    S, U, Mk = [], [], []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        s, u, m = d['states'], d['actions'], d['mask']
        assert s.shape[1:] == (N_STEPS, 8), f"{p}: states {s.shape}"
        assert u.shape[1:] == (N_STEPS, 3), f"{p}: actions {u.shape}"
        assert m.shape[1:] == (N_STEPS,), f"{p}: mask {m.shape}"
        # The splice contract: labels exist exactly where mask is set.
        assert np.all(np.isfinite(u[m.astype(bool)])), \
            f"{p}: NaN label inside the mask"
        assert np.all(np.isnan(u[~m.astype(bool)])), \
            f"{p}: finite label outside the mask"
        S.append(s.astype(np.float32))
        U.append(u.astype(np.float32))
        Mk.append(m.astype(bool))
        print(f"  {Path(p).name}: {s.shape[0]} sequences, "
              f"{int(m.sum()):,} labels")
    return (np.concatenate(S), np.concatenate(U), np.concatenate(Mk))


# =====================================================================
# 2. NORMALISATION
# =====================================================================

def compute_norm_stats(S, U, mask, eps=1e-6):
    """State stats over ALL training states; action stats over LABELLED actions
    only.
    """
    s_flat = S.reshape(-1, 8)
    u_flat = U.reshape(-1, 3)[mask.reshape(-1)]
    return NormalizationStats(
        state_mean=s_flat.mean(axis=0).astype(np.float32),
        state_std=np.maximum(s_flat.std(axis=0), eps).astype(np.float32),
        action_mean=u_flat.mean(axis=0).astype(np.float32),
        action_std=np.maximum(u_flat.std(axis=0), eps).astype(np.float32),
    )


# =====================================================================
# 3. LOSS
# =====================================================================

def masked_losses(pred, target, mask, tz_weight, timestep_weights):
    """Returns (weighted_loss, unweighted_mse)."""
    se = (pred - target) ** 2                              # (B, N, 3)
    w = torch.ones_like(se)
    if tz_weight != 1.0:
        w = w * torch.tensor([1.0, 1.0, tz_weight], device=pred.device)
    if timestep_weights is not None:
        w = w * timestep_weights.view(1, -1, 1)
    m = mask.unsqueeze(-1).to(se.dtype)                     # (B, N, 1)

    weighted = (se * w * m).sum() / (w * m).sum().clamp_min(1.0)
    plain = (se * m).sum() / (m.expand_as(se).sum()).clamp_min(1.0)
    return weighted, plain


# =====================================================================
# 4. TRAINING
# =====================================================================

def train_model(S_tr, U_tr, M_tr, S_va, U_va, M_va, norm, args, device):
    """Train one PDGTransformer and return the best-validation checkpoint with
    its config, best weighted / unweighted validation loss and the per-epoch
    log.
    """
    def prep(S, U, Mk):
        Sn = torch.from_numpy(norm.normalize_state(S))
        Un = torch.from_numpy(
            np.nan_to_num(norm.normalize_action(U), nan=0.0))
        return Sn, Un, torch.from_numpy(Mk)

    Sn, Un, Mn = prep(S_tr, U_tr, M_tr)
    Svn, Uvn, Mvn = prep(S_va, U_va, M_va)

    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(TensorDataset(Sn, Un, Mn), batch_size=32,
                              shuffle=True, generator=g)
    val_loader = DataLoader(TensorDataset(Svn, Uvn, Mvn), batch_size=64,
                            shuffle=False)

    torch.manual_seed(args.seed)
    cfg = TransformerConfig()
    model = PDGTransformer(cfg).to(device)
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} "
          f"parameters")

    Opt = torch.optim.AdamW if args.optimizer == 'adamw' else torch.optim.Adam
    optimiser = Opt(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=0.5, patience=15, min_lr=1e-6)

    tw = None
    if args.late_weight != 1.0:
        tw = torch.linspace(1.0, args.late_weight, N_STEPS, device=device)

    best_val, best_state, best_plain, no_improve = float('inf'), None, \
        float('nan'), 0
    log = []
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        for xb, yb, mb in train_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            loss, _ = masked_losses(model(xb), yb, mb, args.tz_weight, tw)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        model.eval()
        vw, vp = [], []
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                a, b = masked_losses(model(xb.to(device)), yb.to(device),
                                     mb.to(device), args.tz_weight, tw)
                vw.append(a.item())
                vp.append(b.item())
        val_w, val_p = float(np.mean(vw)), float(np.mean(vp))
        scheduler.step(val_w)

        if val_w < best_val:
            best_val, best_plain = val_w, val_p
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        log.append({'epoch': epoch, 'val_weighted': val_w,
                    'val_unweighted': val_p,
                    'lr': optimiser.param_groups[0]['lr']})

        if epoch % 25 == 0 or epoch == args.epochs - 1:
            print(f"    epoch {epoch:3d}  val_w={val_w:.6f}  "
                  f"val_mse={val_p:.6f}  best_w={best_val:.6f}  "
                  f"lr={optimiser.param_groups[0]['lr']:.1e}  "
                  f"[{time.time()-t0:.0f}s]")

        if no_improve >= 30:
            print(f"    Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    print(f"  Done: {epoch+1} epochs in {time.time()-t0:.0f}s")
    print(f"  best weighted val = {best_val:.6f}")
    print(f"  UNWEIGHTED val MSE at that checkpoint = {best_plain:.6f}  "
          f"<-- compare configs on THIS")
    return model, cfg, best_val, best_plain, log


# =====================================================================
# 5. MAIN
# =====================================================================

def main():
    p = argparse.ArgumentParser(description='TTG Transformer training (v2)')
    p.add_argument('--packed_file', required=True)
    p.add_argument('--dagger_file', nargs='*', default=[],
                   help='one or more splice NPZs from dagger_splice.py')
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--max_train', type=int, default=0)
    p.add_argument('--tz_weight', type=float, default=1.0)
    p.add_argument('--late_weight', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42,
                   help='weights, batch order, dropout — VARY for seeds')
    p.add_argument('--split_seed', type=int, default=42,
                   help='train/val/test split — KEEP AT 42 ALWAYS. Must '
                        'match standalone_eval_v2 --seed and the campaign.')
    p.add_argument('--optimizer', choices=['adam', 'adamw'], default='adam')
    p.add_argument('--out_dir', default='runs/ttg_default')
    args = p.parse_args()

    device = torch.device('cpu')

    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass          # already initialised; harmless
    print(f"  Threads: torch intra-op={torch.get_num_threads()}, "
          f"inter-op={torch.get_num_interop_threads()}, "
          f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")
    print(f"  (PBS ncpus must be >= this, and 9 concurrent jobs x this "
          f"count must fit the node)")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    x_traj, u_traj, t_f, status = load_packed_dataset(args.packed_file)
    M_full = x_traj.shape[0]

    # ---- SPLIT FIRST, FILTER SECOND ------------------------------------
    tr_i, va_i, te_i = split_indices(M_full, seed=args.split_seed)
    ok = (status == 0)
    tr_i = tr_i[ok[tr_i]]
    va_i = va_i[ok[va_i]]
    te_i = te_i[ok[te_i]]
    print(f"\n  Split over the FULL {M_full} trajectories "
          f"(split_seed={args.split_seed}), failures dropped within "
          f"subsets:")
    print(f"    train={len(tr_i)}, val={len(va_i)}, test={len(te_i)}  "
          f"({int((~ok).sum())} failed solves removed)")
    print(f"    This must match standalone_eval_v2.py --seed "
          f"{args.split_seed}. Verify with verify_split_alignment.py.")

    if args.max_train > 0:
        tr_i = tr_i[:args.max_train]

    S_tr = build_ttg_states(x_traj[tr_i], t_f[tr_i])
    U_tr = u_traj[tr_i, :N_STEPS, :].astype(np.float32)
    M_tr = np.ones(S_tr.shape[:2], dtype=bool)
    S_va = build_ttg_states(x_traj[va_i], t_f[va_i])
    U_va = u_traj[va_i, :N_STEPS, :].astype(np.float32)
    M_va = np.ones(S_va.shape[:2], dtype=bool)

    n_bc = int(M_tr.sum())
    n_dag = 0
    if args.dagger_file:
        print(f"\n  DAgger splices:")
        S_d, U_d, M_d = load_dagger_files(args.dagger_file)
        n_dag = int(M_d.sum())
        S_tr = np.concatenate([S_tr, S_d])
        U_tr = np.concatenate([U_tr, U_d])
        M_tr = np.concatenate([M_tr, M_d])
        print(f"  Aggregate: {n_bc:,} BC + {n_dag:,} DAgger labels "
              f"= {n_bc+n_dag:,}")
        print(f"  DAgger fraction: {n_dag/(n_bc+n_dag):.1%}  "
              f"(target 15-20%; below ~10% the gradient signal is "
              f"arithmetically thin)")
        print(f"  NOTE: validation stays PURE BC val split, identical in "
              f"definition to\n        the BC arm, so the two runs are "
              f"comparable. Early stopping therefore\n        selects for "
              f"BC-distribution fit — a known limitation, worth stating.")

    norm = compute_norm_stats(S_tr, U_tr, M_tr)
    norm.save(out / 'norm_ttg.npz')
    print(f"\n  Normalisation (TTG channel is the one to watch — drift "
          f"there changes\n  what the learned braking threshold means "
          f"physically):")
    print(f"    TTG    mean={norm.state_mean[7]:.4f}  "
          f"std={norm.state_std[7]:.4f}")
    print(f"    action std={np.array2string(norm.action_std, precision=4)} "
          f"MN")

    print(f"\n{'='*64}")
    print(f"  Training  |  {args.epochs} epochs  |  init seed {args.seed}  "
          f"|  split seed {args.split_seed}")
    print(f"{'='*64}")
    model, cfg, best_w, best_p, log = train_model(
        S_tr, U_tr, M_tr, S_va, U_va, M_va, norm, args, device)

    torch.save(model.state_dict(), out / 'model_ttg.pt')
    cfg.save(out / 'model_config.json')      # architecture, read by the evaluator
    with open(out / 'config.json', 'w') as f:
        json.dump({**vars(args),
                   'n_train_traj': len(tr_i), 'n_val_traj': len(va_i),
                   'n_test_traj': len(te_i),
                   'n_bc_labels': n_bc, 'n_dagger_labels': n_dag,
                   'best_val_weighted': best_w,
                   'best_val_unweighted': best_p,
                   'split_over_full_set': True}, f, indent=2)
    with open(out / 'training_log.json', 'w') as f:
        json.dump(log, f, indent=2)

    print(f"\n  Saved to {out}/  "
          f"(model_ttg.pt, norm_ttg.npz, model_config.json)")
    print(f"  Score with:  python standalone_eval_v2.py --run_dir {out} "
          f"--data_dir <dir> --mode closed_loop --ttg --seed "
          f"{args.split_seed}")


if __name__ == '__main__':
    main()
