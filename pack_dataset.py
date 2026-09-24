"""
Pack a directory of traj_*.npz files into a single compressed NPZ.

Usage:
    python pack_dataset.py --data_dir data/trajectories --out packed_dataset.npz

Output keys: x_traj (M, N+1, 8), u_traj (M, N, 3), t_f (M,), m_f (M,),
status (M,). Rows are in sorted filename order.
"""
import argparse
import numpy as np
from pathlib import Path
import time


def pack_dataset(data_dir: str, out_file: str) -> None:
    """Read every traj_*.npz in data_dir (sorted) and write one packed NPZ."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob('traj_*.npz'))
    print(f"Found {len(files)} trajectory files in {data_dir}")

    if len(files) == 0:
        raise FileNotFoundError(f"No traj_*.npz files in {data_dir}")

    # Load first file to get dimensions
    sample = np.load(files[0])
    N_plus_1 = sample['x_traj'].shape[0]   # N+1
    N = sample['u_traj'].shape[0]           # N
    print(f"Trajectory shape: x=({N_plus_1}, 8), u=({N}, 3)")

    M = len(files)

    # Pre-allocate arrays
    x_all = np.zeros((M, N_plus_1, 8), dtype=np.float32)
    u_all = np.zeros((M, N, 3), dtype=np.float32)
    tf_all = np.zeros(M, dtype=np.float32)
    mf_all = np.zeros(M, dtype=np.float32)
    status_all = np.zeros(M, dtype=np.int32)

    t0 = time.time()
    for i, f in enumerate(files):
        data = np.load(f)
        x_all[i] = data['x_traj'].astype(np.float32)
        u_all[i] = data['u_traj'].astype(np.float32)
        tf_all[i] = float(data['t_f'])
        mf_all[i] = float(data['m_f'])
        status_all[i] = int(data['status'])

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (M - i - 1) / rate
            print(f"  loaded {i+1}/{M} ({rate:.0f} files/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"Loaded {M} files in {elapsed:.1f}s")

    # Save as single file
    np.savez_compressed(
        out_file,
        x_traj=x_all,     # (M, N+1, 8)
        u_traj=u_all,      # (M, N, 3)
        t_f=tf_all,        # (M,)
        m_f=mf_all,        # (M,)
        status=status_all, # (M,)
    )

    # Report size
    out_path = Path(out_file if out_file.endswith('.npz') else out_file + '.npz')
    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved packed dataset: {out_path} ({size_mb:.1f} MB)")
    print(f"Contains {M} trajectories, each with {N} control nodes")
    print(f"\nTo use: data = np.load('{out_path}')")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pack trajectory dataset')
    parser.add_argument('--data_dir', required=True, help='Directory with traj_*.npz files')
    parser.add_argument('--out', default='packed_dataset.npz', help='Output filename')
    args = parser.parse_args()
    pack_dataset(args.data_dir, args.out)
