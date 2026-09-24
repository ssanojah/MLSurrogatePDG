"""
Sliding-window dataset for the reactive Transformer.

Turns each expert trajectory into up to 60 windows of the last W physical
states [r, v, m], each labelled with one thrust command. Windows at the
start are left-padded with the initial state. Split by trajectory, seed 42.

Usage:
    python windowed_dataset.py data/<dataset> --window_size 16

    train_dl, val_dl, test_dl, norm = create_windowed_dataloaders(
        'data/<dataset>', window_size=16, batch_size=128)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional, Union
import argparse
import time


# =====================================================================
# Configuration — which state channels to use
# =====================================================================

STATE_CHANNELS = list(range(7))  # [0, 1, 2, 3, 4, 5, 6]
STATE_DIM = len(STATE_CHANNELS)  # 7
ACTION_DIM = 3                   # [T_cx, T_cy, T_cz] in MN

STATE_NAMES = ['r_x', 'r_y', 'r_z', 'v_x', 'v_y', 'v_z', 'm']
ACTION_NAMES = ['T_cx', 'T_cy', 'T_cz']


# =====================================================================
# Normalization — adapted for 7-dim states
# =====================================================================

class ReactiveNormStats:
    """Per-channel mean/std for the 7-dim reactive state and 3-dim actions."""

    def __init__(self, state_mean, state_std, action_mean, action_std):
        self.state_mean = np.asarray(state_mean, dtype=np.float32)
        self.state_std = np.asarray(state_std, dtype=np.float32)
        self.action_mean = np.asarray(action_mean, dtype=np.float32)
        self.action_std = np.asarray(action_std, dtype=np.float32)

        assert self.state_mean.shape == (STATE_DIM,), \
            f"state_mean shape {self.state_mean.shape}, expected ({STATE_DIM},)"
        assert self.action_mean.shape == (ACTION_DIM,), \
            f"action_mean shape {self.action_mean.shape}, expected ({ACTION_DIM},)"

    # --- numpy (used in Dataset.__getitem__) ---

    def normalize_state(self, x: np.ndarray) -> np.ndarray:
        """Normalize state: (x - mean) / std."""
        return (x - self.state_mean) / self.state_std

    def normalize_action(self, u: np.ndarray) -> np.ndarray:
        """Normalize action: (u - mean) / std."""
        return (u - self.action_mean) / self.action_std

    def unnormalize_state(self, x_norm: np.ndarray) -> np.ndarray:
        return x_norm * self.state_std + self.state_mean

    def unnormalize_action(self, u_norm: np.ndarray) -> np.ndarray:
        return u_norm * self.action_std + self.action_mean

    # --- torch (used during inference / evaluation) ---

    def normalize_state_torch(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.from_numpy(self.state_mean).to(x.device, x.dtype)
        std = torch.from_numpy(self.state_std).to(x.device, x.dtype)
        return (x - mean) / std

    def unnormalize_action_torch(self, u_norm: torch.Tensor) -> torch.Tensor:
        mean = torch.from_numpy(self.action_mean).to(u_norm.device, u_norm.dtype)
        std = torch.from_numpy(self.action_std).to(u_norm.device, u_norm.dtype)
        return u_norm * std + mean

    # --- persistence ---

    def save(self, filepath: Union[str, Path]) -> None:
        np.savez(
            filepath,
            state_mean=self.state_mean,
            state_std=self.state_std,
            action_mean=self.action_mean,
            action_std=self.action_std,
            state_channels=np.array(STATE_CHANNELS),  # record which channels
        )

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ReactiveNormStats':
        data = np.load(filepath)
        return cls(
            state_mean=data['state_mean'],
            state_std=data['state_std'],
            action_mean=data['action_mean'],
            action_std=data['action_std'],
        )

    def summary(self) -> str:
        lines = ['ReactiveNormStats (7-dim state, no t_f)', '=' * 50]
        lines.append('\nStates:')
        for i, name in enumerate(STATE_NAMES):
            lines.append(f'  {name:>4s}: mean = {self.state_mean[i]:12.4f}, '
                         f'std = {self.state_std[i]:12.4f}')
        lines.append('\nActions:')
        for i, name in enumerate(ACTION_NAMES):
            lines.append(f'  {name:>4s}: mean = {self.action_mean[i]:12.4f}, '
                         f'std = {self.action_std[i]:12.4f}')
        return '\n'.join(lines)


def compute_reactive_norm_stats(
    traj_files: List[Union[str, Path]],
    eps: float = 1e-6,
) -> ReactiveNormStats:
    """Compute per-channel mean/std from training trajectory files.

    Parameters
    ----------
    traj_files : list of paths to trajectory NPZ files
    eps        : minimum std to prevent division by zero
    """
    all_states = []
    all_actions = []

    for f in traj_files:
        data = np.load(f)
        # States at control nodes (exclude terminal), select channels
        x = data['x_traj'][:-1].astype(np.float64)  # (N, 8)
        all_states.append(x[:, STATE_CHANNELS])       # (N, 7)
        all_actions.append(data['u_traj'].astype(np.float64))  # (N, 3)

    all_states = np.concatenate(all_states, axis=0)    # (total_steps, 7)
    all_actions = np.concatenate(all_actions, axis=0)   # (total_steps, 3)

    state_mean = all_states.mean(axis=0).astype(np.float32)
    state_std = all_states.std(axis=0).astype(np.float32)
    state_std = np.maximum(state_std, eps)

    action_mean = all_actions.mean(axis=0).astype(np.float32)
    action_std = all_actions.std(axis=0).astype(np.float32)
    action_std = np.maximum(action_std, eps)

    print(f"[compute_reactive_norm_stats] Computed from {len(traj_files)} files, "
          f"{all_states.shape[0]} total state samples")

    return ReactiveNormStats(state_mean, state_std, action_mean, action_std)


# =====================================================================
# Train / val / test split — reuses logic from dataset.py
# =====================================================================

def split_by_trajectory(
    traj_files: List[Union[str, Path]],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split trajectory files into train/val/test by trajectory (not window)."""
    files = [Path(f) for f in traj_files]
    n = len(files)

    if n < 5:
        print(f"[split] Only {n} trajectories — all go to train.")
        return files, [], []

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))

    train_files = [files[i] for i in indices[:n_train]]
    val_files = [files[i] for i in indices[n_train:n_train + n_val]]
    test_files = [files[i] for i in indices[n_train + n_val:]]

    print(f"[split] {n} trajectories -> "
          f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    return train_files, val_files, test_files


# =====================================================================
# Windowed Dataset
# =====================================================================

class WindowedTrajectoryDataset(Dataset):
    """Sliding-window dataset for reactive Transformer training."""

    def __init__(
        self,
        traj_files: List[Union[str, Path]],
        norm_stats: ReactiveNormStats,
        window_size: int = 16,
        filter_status: Optional[int] = 0,
    ):
        """Parameters
        ----------
        traj_files    : paths to trajectory NPZ files
        norm_stats    : ReactiveNormStats computed from training set
        window_size   : W, number of states in the sliding window
        filter_status : only include trajectories with this solver status
                        (0 = optimal). Set to None to include all.
        """
        self.norm_stats = norm_stats
        self.W = window_size

        # Load trajectories — store only the channels we need
        self.trajectories = []
        n_filtered = 0
        for f in traj_files:
            data = np.load(f)
            status = int(data['status'])
            if filter_status is not None and status != filter_status:
                n_filtered += 1
                continue

            x_full = data['x_traj'].astype(np.float32)   # (N+1, 8)
            u_full = data['u_traj'].astype(np.float32)    # (N, 3)

            self.trajectories.append({
                'states': x_full[:-1, STATE_CHANNELS],    # (N, 7) control nodes
                'actions': u_full,                         # (N, 3)
                'filepath': str(f),
            })

        if n_filtered > 0:
            print(f"[WindowedDataset] Filtered {n_filtered} trajectories "
                  f"with status != {filter_status}")

        if len(self.trajectories) == 0:
            raise ValueError("No valid trajectories found after filtering!")

        self.index_map = []
        total_steps = 0
        n_lengths = {}
        for traj_idx, traj in enumerate(self.trajectories):
            n_steps = traj['states'].shape[0]
            for step_k in range(n_steps):
                self.index_map.append((traj_idx, step_k))
            total_steps += n_steps
            n_lengths[n_steps] = n_lengths.get(n_steps, 0) + 1

        n_trajs = len(self.trajectories)
        n_windows = len(self.index_map)

        # Report trajectory length distribution
        if len(n_lengths) == 1:
            N_val = list(n_lengths.keys())[0]
            print(f"[WindowedDataset] {n_trajs} trajectories × {N_val} steps "
                  f"= {n_windows} windows (W={self.W})")
        else:
            len_str = ', '.join(f'{n}steps:{count}'
                                for n, count in sorted(n_lengths.items()))
            print(f"[WindowedDataset] {n_trajs} trajectories, "
                  f"variable lengths ({len_str}), "
                  f"total {n_windows} windows (W={self.W})")

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> dict:
        traj_idx, step_k = self.index_map[idx]
        traj = self.trajectories[traj_idx]

        states = traj['states']   # (N, 7), raw
        actions = traj['actions']  # (N, 3), raw

        # --- Extract window of W states ending at step_k ---
        window = np.zeros((self.W, STATE_DIM), dtype=np.float32)
        mask = np.ones(self.W, dtype=bool)  # True = padding (PyTorch convention)

        if step_k >= self.W - 1:
            # Full window available — no padding needed
            start = step_k - self.W + 1
            window[:] = states[start:step_k + 1]
            mask[:] = False  # all positions are real
        else:
            # Need left-padding: step_k+1 real states, W-(step_k+1) padded
            n_real = step_k + 1
            n_pad = self.W - n_real

            # Pad with the initial state (x_0) of this trajectory
            window[:n_pad] = states[0]           # repeat x_0
            window[n_pad:] = states[:n_real]     # real states
            mask[:n_pad] = True                   # padded positions
            mask[n_pad:] = False                  # real positions

        # --- Label: expert control at the current step ---
        label = actions[step_k].copy()  # (3,)

        # --- Normalize ---
        window = self.norm_stats.normalize_state(window)
        label = self.norm_stats.normalize_action(label)

        return {
            'window': torch.from_numpy(window),   # (W, 7)
            'label': torch.from_numpy(label),      # (3,)
            'mask': torch.from_numpy(mask),        # (W,) bool
        }

    def get_raw(self, idx: int) -> dict:
        """Return unnormalized data for inspection/debugging."""
        traj_idx, step_k = self.index_map[idx]
        traj = self.trajectories[traj_idx]
        return {
            'traj_idx': traj_idx,
            'step_k': step_k,
            'state': traj['states'][step_k].copy(),
            'action': traj['actions'][step_k].copy(),
            'filepath': traj['filepath'],
        }


# =====================================================================
# Convenience: create DataLoaders
# =====================================================================

def create_windowed_dataloaders(
    traj_dir: Union[str, Path],
    window_size: int = 16,
    batch_size: int = 128,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
    filter_status: Optional[int] = 0,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], ReactiveNormStats]:
    """End-to-end: discover NPZ files, split by trajectory, compute norms,
    create windowed DataLoaders.

    Parameters
    ----------
    traj_dir      : directory containing traj_*.npz files
    window_size   : W, number of states in each window
    batch_size    : batch size (can be larger than for full sequences:
                    samples are (W, 7) instead of (60, 8))
    train_frac    : fraction for training split
    val_frac      : fraction for validation split
    seed          : random seed (same as dataset.py default for consistency)
    num_workers   : DataLoader workers (0 = main process)
    filter_status : only keep trajectories with this solver status

    Returns
    -------
    (train_loader, val_loader, test_loader, norm_stats)
    """
    traj_dir = Path(traj_dir)
    all_files = sorted(traj_dir.glob('traj_*.npz'))

    if len(all_files) == 0:
        raise FileNotFoundError(
            f"No trajectory files (traj_*.npz) found in {traj_dir}")

    print(f"Found {len(all_files)} trajectory files in {traj_dir}")

    # Split by trajectory (not by window) — same logic as dataset.py
    train_files, val_files, test_files = split_by_trajectory(
        all_files, train_frac, val_frac, seed)

    # Compute normalization from training trajectories only
    norm_stats = compute_reactive_norm_stats(train_files)
    norm_stats.save(traj_dir / 'reactive_norm_stats.npz')
    print(norm_stats.summary())

    # Create datasets
    train_ds = WindowedTrajectoryDataset(
        train_files, norm_stats, window_size, filter_status)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)

    val_loader = None
    if val_files:
        val_ds = WindowedTrajectoryDataset(
            val_files, norm_stats, window_size, filter_status)
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

    test_loader = None
    if test_files:
        test_ds = WindowedTrajectoryDataset(
            test_files, norm_stats, window_size, filter_status)
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, norm_stats


# =====================================================================
# Standalone verification
# =====================================================================

def verify_dataset(traj_dir: str, window_size: int = 16, seed: int = 42):
    """Run standalone to verify the windowed dataset pipeline."""
    traj_dir = Path(traj_dir)
    print("=" * 60)
    print("WINDOWED DATASET VERIFICATION")
    print("=" * 60)
    print(f"  Data dir:    {traj_dir}")
    print(f"  Window size: {window_size}")
    print(f"  Seed:        {seed}")
    print()

    all_files = sorted(traj_dir.glob('traj_*.npz'))
    print(f"Found {len(all_files)} trajectory files.\n")

    if len(all_files) == 0:
        print("ERROR: No files found. Check the path.")
        return

    # --- 1. Split ---
    train_files, val_files, test_files = split_by_trajectory(
        all_files, seed=seed)

    # --- 2. Normalization ---
    t0 = time.time()
    norm_stats = compute_reactive_norm_stats(train_files)
    t_norm = time.time() - t0
    print(f"\nNormalization computed in {t_norm:.1f}s")
    print(norm_stats.summary())
    print()

    # --- 3. Create dataset and check shapes ---
    ds = WindowedTrajectoryDataset(
        train_files, norm_stats, window_size, filter_status=None)

    print(f"\nDataset length: {len(ds)}")

    # Check first, last, and a middle sample
    for desc, idx in [("First", 0), ("Middle", len(ds) // 2), ("Last", len(ds) - 1)]:
        sample = ds[idx]
        raw = ds.get_raw(idx)
        print(f"\n  {desc} sample (idx={idx}):")
        print(f"    traj_idx={raw['traj_idx']}, step_k={raw['step_k']}")
        print(f"    window shape: {sample['window'].shape}  "
              f"(expect ({window_size}, {STATE_DIM}))")
        print(f"    label shape:  {sample['label'].shape}  "
              f"(expect ({ACTION_DIM},))")
        print(f"    mask shape:   {sample['mask'].shape}  "
              f"(expect ({window_size},))")
        print(f"    n_padded:     {sample['mask'].sum().item()}")

    # --- 4. Round-trip normalization check ---
    sample = ds[len(ds) // 2]
    raw = ds.get_raw(len(ds) // 2)
    label_denorm = norm_stats.unnormalize_action(sample['label'].numpy())
    err = np.abs(label_denorm - raw['action']).max()
    print(f"\n  Round-trip normalization error (action): {err:.2e}")
    assert err < 1e-5, f"Normalization round-trip error too large: {err}"

    # --- 5. Padding check for early steps ---
    sample_0 = ds[0]  # step_k=0 of first trajectory
    n_pad = sample_0['mask'].sum().item()
    expected_pad = window_size - 1
    print(f"\n  Step 0 padding: {n_pad} positions "
          f"(expect {expected_pad} for W={window_size})")
    assert n_pad == expected_pad, \
        f"Padding mismatch: got {n_pad}, expected {expected_pad}"

    # Check that padded positions contain the initial state
    raw_0 = ds.get_raw(0)
    traj_0_states = ds.trajectories[raw_0['traj_idx']]['states']
    x0_norm = norm_stats.normalize_state(traj_0_states[0])
    for p in range(n_pad):
        pad_err = np.abs(sample_0['window'][p].numpy() - x0_norm).max()
        assert pad_err < 1e-6, f"Padded position {p} doesn't match x_0"
    print("  Padded positions correctly contain x_0 ✓")

    # --- 6. No data leakage check ---
    train_set = set(str(f) for f in train_files)
    test_set = set(str(f) for f in test_files)
    overlap = train_set & test_set
    assert len(overlap) == 0, f"DATA LEAKAGE: {len(overlap)} files in both sets!"
    print(f"  No data leakage between train/test ✓")

    # --- 7. DataLoader batch check ---
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    batch = next(iter(loader))
    print(f"\n  DataLoader batch shapes:")
    print(f"    window: {batch['window'].shape}  "
          f"(expect (64, {window_size}, {STATE_DIM}))")
    print(f"    label:  {batch['label'].shape}  "
          f"(expect (64, {ACTION_DIM}))")
    print(f"    mask:   {batch['mask'].shape}  "
          f"(expect (64, {window_size}))")

    # --- 8. Statistics summary ---
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Trajectories: {len(all_files)} total "
          f"({len(train_files)} train / {len(val_files)} val / "
          f"{len(test_files)} test)")
    print(f"  Window size: {window_size}")
    print(f"  State dim: {STATE_DIM} (dropped t_f)")
    print(f"  Action dim: {ACTION_DIM}")
    print(f"  Total training windows: {len(ds):,}")

    print(f"\n  All checks passed ✓")


# =====================================================================
# CLI
# =====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Verify the windowed dataset pipeline for reactive BC')
    parser.add_argument('traj_dir', type=str,
                        help='Directory containing traj_*.npz files')
    parser.add_argument('--window_size', '-w', type=int, default=16,
                        help='Sliding window size W (default: 16)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for split (default: 42)')
    args = parser.parse_args()

    verify_dataset(args.traj_dir, args.window_size, args.seed)
