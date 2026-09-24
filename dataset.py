"""
Expert trajectory dataset utilities.

Trajectory NPZ format:
    x_traj : (N+1, 8) states [r_x, r_y, r_z, v_x, v_y, v_z, m, t_f]
    u_traj : (N, 3)   thrust [MN]
    ic     : (8,)     initial condition
    status : int      solver status (0 = optimal)
    t_f    : float    final time [s]
    m_f    : float    final mass [kg]

Contents: save/load helpers, NormalizationStats, split_by_ic (train/val/test
split by trajectory, seed 42), TrajectoryDataset and create_dataloaders.

Usage:
    files = sorted(Path('data/trajectories').glob('traj_*.npz'))
    train_files, val_files, test_files = split_by_ic(files)
    norm_stats = compute_norm_stats(train_files)
    train_ds = TrajectoryDataset(train_files, norm_stats)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional, Union
import json


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_trajectory(
    filepath: Union[str, Path],
    x_traj: np.ndarray,
    u_traj: np.ndarray,
    status: int,
    t_f: float,
    m_f: float,
) -> None:
    """Save one solved trajectory to an NPZ file.

    Parameters
    ----------
    filepath : path to save the .npz file
    x_traj   : (N+1, 8) state trajectory
    u_traj   : (N, 3) control trajectory in MN
    status   : OCP solver status (0 = optimal)
    t_f      : optimal final time [s]
    m_f      : final mass [kg]
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    x_traj = np.asarray(x_traj, dtype=np.float64)
    u_traj = np.asarray(u_traj, dtype=np.float64)

    assert x_traj.ndim == 2 and x_traj.shape[1] == 8, \
        f"x_traj must be (N+1, 8), got {x_traj.shape}"
    assert u_traj.ndim == 2 and u_traj.shape[1] == 3, \
        f"u_traj must be (N, 3), got {u_traj.shape}"
    assert x_traj.shape[0] == u_traj.shape[0] + 1, \
        f"x_traj has {x_traj.shape[0]} rows but u_traj has {u_traj.shape[0]} " \
        f"(expected N+1 and N)"

    np.savez(
        filepath,
        x_traj=x_traj,
        u_traj=u_traj,
        ic=x_traj[0],           # redundant but convenient for filtering
        status=int(status),
        t_f=float(t_f),
        m_f=float(m_f),
    )


def load_trajectory(filepath: Union[str, Path]) -> dict:
    """Load a trajectory NPZ and return a dict with all fields."""
    data = np.load(filepath, allow_pickle=False)
    return {
        'x_traj': data['x_traj'],     # (N+1, 8)
        'u_traj': data['u_traj'],     # (N, 3)
        'ic':     data['ic'],         # (8,)
        'status': int(data['status']),
        't_f':    float(data['t_f']),
        'm_f':    float(data['m_f']),
    }


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class NormalizationStats:
    """Per-channel mean/std for states and actions."""

    def __init__(
        self,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        action_mean: np.ndarray,
        action_std: np.ndarray,
    ):
        self.state_mean = np.asarray(state_mean, dtype=np.float32)
        self.state_std = np.asarray(state_std, dtype=np.float32)
        self.action_mean = np.asarray(action_mean, dtype=np.float32)
        self.action_std = np.asarray(action_std, dtype=np.float32)

    # --- numpy operations (used in Dataset.__getitem__) ---

    def normalize_state(self, x: np.ndarray) -> np.ndarray:
        """Normalize state array: (x - mean) / std."""
        return (x - self.state_mean) / self.state_std

    def normalize_action(self, u: np.ndarray) -> np.ndarray:
        """Normalize action array: (u - mean) / std."""
        return (u - self.action_mean) / self.action_std

    def unnormalize_state(self, x_norm: np.ndarray) -> np.ndarray:
        """Reverse state normalization."""
        return x_norm * self.state_std + self.state_mean

    def unnormalize_action(self, u_norm: np.ndarray) -> np.ndarray:
        """Reverse action normalization."""
        return u_norm * self.action_std + self.action_mean

    # --- torch operations (used during inference / evaluation) ---

    def unnormalize_action_torch(self, u_norm: torch.Tensor) -> torch.Tensor:
        """Reverse action normalization on a torch tensor."""
        mean = torch.from_numpy(self.action_mean).to(u_norm.device)
        std = torch.from_numpy(self.action_std).to(u_norm.device)
        return u_norm * std + mean

    def normalize_state_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize state torch tensor."""
        mean = torch.from_numpy(self.state_mean).to(x.device)
        std = torch.from_numpy(self.state_std).to(x.device)
        return (x - mean) / std

    # --- persistence ---

    def save(self, filepath: Union[str, Path]) -> None:
        """Save normalization stats to an NPZ file."""
        np.savez(
            filepath,
            state_mean=self.state_mean,
            state_std=self.state_std,
            action_mean=self.action_mean,
            action_std=self.action_std,
        )

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'NormalizationStats':
        """Load normalization stats from an NPZ file."""
        data = np.load(filepath)
        return cls(
            state_mean=data['state_mean'],
            state_std=data['state_std'],
            action_mean=data['action_mean'],
            action_std=data['action_std'],
        )

    def summary(self) -> str:
        """Human-readable summary of the normalization statistics."""
        state_names = ['r_x', 'r_y', 'r_z', 'v_x', 'v_y', 'v_z', 'm', 't_f']
        action_names = ['T_cx', 'T_cy', 'T_cz']
        lines = ['Normalization Statistics', '=' * 50]
        lines.append('\nStates:')
        for i, name in enumerate(state_names):
            lines.append(f'  {name:>4s}: mean = {self.state_mean[i]:12.4f}, '
                         f'std = {self.state_std[i]:12.4f}')
        lines.append('\nActions:')
        for i, name in enumerate(action_names):
            lines.append(f'  {name:>4s}: mean = {self.action_mean[i]:12.4f}, '
                         f'std = {self.action_std[i]:12.4f}')
        return '\n'.join(lines)


def compute_norm_stats(
    traj_files: List[Union[str, Path]],
    eps: float = 1e-6,
) -> NormalizationStats:
    """Compute per-channel mean and std from a list of trajectory files.

    Parameters
    ----------
    traj_files : list of paths to trajectory NPZ files
    eps        : minimum std to prevent division by zero

    Returns
    -------
    NormalizationStats with computed mean/std for states and actions
    """
    all_states = []
    all_actions = []

    for f in traj_files:
        data = np.load(f)
        # States at control nodes only (not terminal)
        all_states.append(data['x_traj'][:-1].astype(np.float64))
        all_actions.append(data['u_traj'].astype(np.float64))

    all_states = np.concatenate(all_states, axis=0)    # (total_steps, 8)
    all_actions = np.concatenate(all_actions, axis=0)   # (total_steps, 3)

    state_mean = all_states.mean(axis=0).astype(np.float32)
    state_std = all_states.std(axis=0).astype(np.float32)
    state_std = np.maximum(state_std, eps)

    action_mean = all_actions.mean(axis=0).astype(np.float32)
    action_std = all_actions.std(axis=0).astype(np.float32)
    action_std = np.maximum(action_std, eps)

    return NormalizationStats(state_mean, state_std, action_mean, action_std)


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------

def split_by_ic(
    traj_files: List[Union[str, Path]],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split trajectory files into train / val / test sets.

    Parameters
    ----------
    traj_files : list of paths to trajectory NPZ files
    train_frac : fraction for training (default 0.70)
    val_frac   : fraction for validation (default 0.15)
    seed       : random seed for reproducibility

    Returns
    -------
    (train_files, val_files, test_files) — each a list of Path objects
    """
    files = [Path(f) for f in traj_files]
    n = len(files)

    if n < 5:
        # Too few to split meaningfully — put everything in train
        print(f"[split_by_ic] Only {n} trajectories — using all for training "
              f"(need >= 5 for a meaningful split).")
        return files, [], []

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    # test gets the remainder
    n_test = n - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_files = [files[i] for i in train_idx]
    val_files = [files[i] for i in val_idx]
    test_files = [files[i] for i in test_idx]

    print(f"[split_by_ic] {n} trajectories -> "
          f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    return train_files, val_files, test_files


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class TrajectoryDataset(Dataset):
    """PyTorch Dataset of expert trajectories for behavioral cloning."""

    def __init__(
        self,
        traj_files: List[Union[str, Path]],
        norm_stats: NormalizationStats,
        filter_status: int = 0,
    ):
        """Parameters
        ----------
        traj_files    : list of paths to trajectory NPZ files
        norm_stats    : NormalizationStats computed from training set
        filter_status : only include trajectories with this solver status
                        (default 0 = successful solve). Set to None to
                        include all.
        """
        self.norm_stats = norm_stats

        # Load and filter trajectories
        self.trajectories = []
        n_filtered = 0
        for f in traj_files:
            data = np.load(f)
            status = int(data['status'])
            if filter_status is not None and status != filter_status:
                n_filtered += 1
                continue
            self.trajectories.append({
                'x_traj': data['x_traj'].astype(np.float32),  # (N+1, 8)
                'u_traj': data['u_traj'].astype(np.float32),  # (N, 3)
                't_f': float(data['t_f']),
                'm_f': float(data['m_f']),
                'filepath': str(f),
            })

        if n_filtered > 0:
            print(f"[TrajectoryDataset] Filtered out {n_filtered} trajectories "
                  f"with status != {filter_status}")

        if len(self.trajectories) == 0:
            raise ValueError("No valid trajectories found after filtering!")

        # Verify consistent sequence length
        self.N = self.trajectories[0]['u_traj'].shape[0]
        for traj in self.trajectories:
            if traj['u_traj'].shape[0] != self.N:
                raise ValueError(
                    f"Inconsistent sequence length: expected N={self.N}, "
                    f"got {traj['u_traj'].shape[0]} in {traj['filepath']}")

        print(f"[TrajectoryDataset] Loaded {len(self.trajectories)} trajectories, "
              f"N={self.N} control nodes each")

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> dict:
        traj = self.trajectories[idx]

        # States at control nodes (exclude terminal state)
        states = traj['x_traj'][:-1].copy()    # (N, 8)
        actions = traj['u_traj'].copy()          # (N, 3)

        # Apply normalization
        states = self.norm_stats.normalize_state(states)
        actions = self.norm_stats.normalize_action(actions)

        return {
            'states': torch.from_numpy(states),     # (N, 8)
            'actions': torch.from_numpy(actions),   # (N, 3)
        }

    def get_raw(self, idx: int) -> dict:
        """Return raw (unnormalized) trajectory data for plotting/inspection.

        Returns
        -------
        dict with keys:
            'x_traj' : (N+1, 8) full state trajectory (including terminal)
            'u_traj' : (N, 3) control trajectory in MN
            't_f'    : optimal final time [s]
            'm_f'    : final mass [kg]
        """
        traj = self.trajectories[idx]
        return {
            'x_traj': traj['x_traj'].copy(),
            'u_traj': traj['u_traj'].copy(),
            't_f': traj['t_f'],
            'm_f': traj['m_f'],
        }


# ---------------------------------------------------------------------------
# Convenience: create DataLoaders
# ---------------------------------------------------------------------------

def create_dataloaders(
    traj_dir: Union[str, Path],
    batch_size: int = 16,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], NormalizationStats]:
    """End-to-end convenience function: discover NPZ files, split, normalize,
    create DataLoaders.

    Parameters
    ----------
    traj_dir    : directory containing trajectory NPZ files
    batch_size  : batch size for DataLoaders
    train_frac  : fraction for training split
    val_frac    : fraction for validation split
    seed        : random seed
    num_workers : DataLoader worker processes (0 = main process)

    Returns
    -------
    (train_loader, val_loader, test_loader, norm_stats)
    val_loader and test_loader may be None if too few trajectories.
    """
    traj_dir = Path(traj_dir)
    all_files = sorted(traj_dir.glob('traj_*.npz'))

    if len(all_files) == 0:
        raise FileNotFoundError(
            f"No trajectory files (traj_*.npz) found in {traj_dir}")

    print(f"Found {len(all_files)} trajectory files in {traj_dir}")

    # Split
    train_files, val_files, test_files = split_by_ic(
        all_files, train_frac, val_frac, seed)

    # Compute normalization from training set only
    norm_stats = compute_norm_stats(train_files)
    norm_stats.save(traj_dir / 'norm_stats.npz')
    print(norm_stats.summary())

    # Create datasets
    train_ds = TrajectoryDataset(train_files, norm_stats)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)

    val_loader = None
    if val_files:
        val_ds = TrajectoryDataset(val_files, norm_stats)
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

    test_loader = None
    if test_files:
        test_ds = TrajectoryDataset(test_files, norm_stats)
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, norm_stats
