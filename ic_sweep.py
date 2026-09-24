#!/usr/bin/env python3
"""
IC sweep: expert dataset generation with FORCESPRO.

Samples dispersed initial conditions (Latin Hypercube), solves each with the
solver in ./pdg_expert/ and saves one traj_XXXXX.npz per converged IC, plus
sweep_log.csv.

Usage:
    python ic_sweep.py --dry_run
    python ic_sweep.py --n_samples 5000 --output_dir data/trajectories
"""

import numpy as np
import time
import argparse
import csv
from pathlib import Path
from scipy.stats.qmc import LatinHypercube

import forcespro.nlp                                   # for Solver.from_directory
from forces_pdg import solve_ocp, save_trajectory, m0

SOLVER_DIR = 'pdg_expert'   # directory produced by generate_solver()


# =====================================================================
# Nominal IC and dispersion ranges
# =====================================================================

NOMINAL_POS  = np.array([-2386.0, 223.0, 6038.0])   # [m]  Guadagnini 2025
NOMINAL_VEL  = np.array([199.0, -18.0, -251.0])     # [m/s]
NOMINAL_MASS = m0                                    # 70,000 kg

POS_DISP = np.array([500.0, 500.0, 500.0])          # [m]
VEL_DISP = np.array([30.0, 30.0, 30.0])             # [m/s]

TF_GUESS_PRIMARY = 50.0      # [s]  (nominal converged with this)
TF_GUESS_RETRY   = 65.0      # [s]  different guess to escape a bad basin


# =====================================================================
# IC generation
# =====================================================================

def generate_ics(n_samples: int, seed: int = 42) -> np.ndarray:
    """Dispersed ICs via Latin Hypercube Sampling; mass fixed at nominal."""
    sampler = LatinHypercube(d=6, seed=seed)
    unit_samples = sampler.random(n=n_samples)

    pos = NOMINAL_POS + (2.0 * unit_samples[:, :3] - 1.0) * POS_DISP
    vel = NOMINAL_VEL + (2.0 * unit_samples[:, 3:] - 1.0) * VEL_DISP

    ics = np.zeros((n_samples, 7))
    ics[:, :3] = pos
    ics[:, 3:6] = vel
    ics[:, 6] = NOMINAL_MASS
    return ics


def check_ic_feasibility(ics: np.ndarray, gamma_gs_deg: float = 30.0) -> np.ndarray:
    """Geometric pre-check: does each IC satisfy the (linear) glide-slope at
    t=0? tan(gamma)*||r_horiz|| <= r_z, matching the NLP's linear cone.
    """
    gamma_gs = np.deg2rad(gamma_gs_deg)
    r_z = ics[:, 2]
    r_horiz = np.sqrt(ics[:, 0]**2 + ics[:, 1]**2)
    feasible = r_z >= np.tan(gamma_gs) * r_horiz
    feasible &= (r_z > 0)
    return feasible


# =====================================================================
# Status mapping   (FORCESPRO exitflag -> dataset status int)
# =====================================================================

def forces_exitflag_to_int(result: dict, res_tol: float = 1e-4) -> int:
    """Map FORCESPRO result to dataset.py status convention."""
    ef = result.get('exitflag', -99)
    if ef == 1:
        return 0
    if ef == 0 and result.get('res_eq', np.inf) < res_tol \
              and result.get('res_ineq', np.inf) < res_tol:
        return 1
    return 2


def _failed_result() -> dict:
    """Placeholder result for an exception during solve."""
    return {'exitflag': -99, 'tf': 0.0, 'x_traj': None, 'u_traj': None,
            'solvetime': 0.0, 'res_eq': np.inf, 'res_ineq': np.inf,
            'm_f': 0.0, 'it': 0}


# =====================================================================
# Main sweep
# =====================================================================

def run_sweep(n_samples: int, output_dir: str, seed: int = 42,
              dry_run: bool = False, accept_level: int = 1) -> None:
    """Sample ICs, solve each one, save accepted trajectories and log every
    attempt. accept_level sets the worst status saved (0 = optimal only, 1 =
    also acceptable).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Generate ICs ----
    print(f"Generating {n_samples} ICs via Latin Hypercube Sampling (seed={seed})...")
    ics = generate_ics(n_samples, seed=seed)

    feasible = check_ic_feasibility(ics)
    n_prefail = (~feasible).sum()
    if n_prefail > 0:
        print(f"  WARNING: {n_prefail}/{n_samples} ICs fail glide-slope pre-check "
              f"and will be skipped.")
        ics = ics[feasible]
        n_samples = len(ics)
        print(f"  Proceeding with {n_samples} feasible ICs.")

    print(f"\nIC dispersion statistics:")
    labels = ['r_x [m]', 'r_y [m]', 'r_z [m]', 'v_x [m/s]', 'v_y [m/s]', 'v_z [m/s]', 'm [kg]']
    for i, label in enumerate(labels):
        print(f"  {label:>12s}: min={ics[:, i].min():10.1f}, "
              f"max={ics[:, i].max():10.1f}, mean={ics[:, i].mean():10.1f}")

    if dry_run:
        print("\n[DRY RUN] No solves performed. Exiting.")
        return

    # ---- Load the pre-generated solver ONCE (never inside the loop) ----
    print(f"\nLoading FORCESPRO solver from '{SOLVER_DIR}/' ...")
    solver = forcespro.Solver.from_directory(SOLVER_DIR)
    print("  Solver loaded.\n")

    # ---- Solve loop ----
    log_path = output_dir / 'sweep_log.csv'
    log_fields = ['ic_index', 'exitflag', 'status_int', 'tf_guess_used',
                  't_f', 'm_f', 'fuel_kg', 'iters', 'solve_time_s', 'retried', 'saved']

    n_success = n_acceptable = n_failed = n_saved = 0
    total_solve_time = 0.0
    t_sweep_start = time.time()

    with open(log_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=log_fields)
        writer.writeheader()

        for i in range(n_samples):
            x0 = ics[i]
            retried = False

            # ---- First attempt ----
            try:
                result = solve_ocp(solver, x0, tf_guess=TF_GUESS_PRIMARY)
            except Exception as e:
                print(f"  [{i+1:5d}/{n_samples}] EXCEPTION: {e}")
                result = _failed_result()
            status_int = forces_exitflag_to_int(result)

            # ---- Retry on failure with a different guess ----
            if status_int > accept_level:
                retried = True
                try:
                    result = solve_ocp(solver, x0, tf_guess=TF_GUESS_RETRY)
                except Exception as e:
                    print(f"  [{i+1:5d}/{n_samples}] RETRY EXCEPTION: {e}")
                    result = _failed_result()
                status_int = forces_exitflag_to_int(result)

            total_solve_time += result.get('solvetime', 0.0)

            # ---- Classify ----
            if status_int == 0:
                n_success += 1
            elif status_int == 1:
                n_acceptable += 1
            else:
                n_failed += 1

            # ---- Save if good enough ----
            saved = False
            if status_int <= accept_level and result['x_traj'] is not None:
                # x_traj is already (61, 8) = [r, v, m, t_f] -- no padding needed
                traj_path = output_dir / f'traj_{i:05d}.npz'
                save_trajectory(
                    filepath=traj_path,
                    x_traj=result['x_traj'],
                    u_traj=result['u_traj'],
                    status=status_int,
                    t_f=result['tf'],
                    m_f=result['m_f'],
                )
                n_saved += 1
                saved = True

            # ---- Log ----
            tf_guess_used = TF_GUESS_RETRY if retried else TF_GUESS_PRIMARY
            m_f = result.get('m_f', 0.0) if result['x_traj'] is not None else 0.0
            writer.writerow({
                'ic_index': i,
                'exitflag': result.get('exitflag', -99),
                'status_int': status_int,
                'tf_guess_used': tf_guess_used,
                't_f': result.get('tf', 0.0),
                'm_f': m_f,
                'fuel_kg': (NOMINAL_MASS - m_f) if result['x_traj'] is not None else 0.0,
                'iters': result.get('it', 0),
                'solve_time_s': result.get('solvetime', 0.0),
                'retried': retried,
                'saved': saved,
            })

            # ---- Progress ----
            elapsed = time.time() - t_sweep_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_samples - i - 1) / rate if rate > 0 else 0
            symbol = '\u2713' if saved else ('~' if status_int <= 1 else '\u2717')
            retry_str = ' [R]' if retried else '    '
            print(f"  [{i+1:5d}/{n_samples}] {symbol}{retry_str} "
                  f"exitflag={result.get('exitflag', -99):>3d}  "
                  f"t_f={result.get('tf', 0.0):6.1f}s  "
                  f"m_f={m_f/1000:6.2f}t  "
                  f"it={result.get('it', 0):4d}  "
                  f"solve={result.get('solvetime', 0.0)*1e3:6.1f}ms  "
                  f"ETA={eta/60:5.1f}min")

    # ---- Summary ----
    t_sweep_total = time.time() - t_sweep_start
    print(f"\n{'='*70}\nSWEEP COMPLETE\n{'='*70}")
    print(f"  Total ICs attempted:   {n_samples}")
    print(f"  Optimal (exitflag 1):  {n_success}  ({100*n_success/n_samples:.1f}%)")
    print(f"  Acceptable (feasible): {n_acceptable}  ({100*n_acceptable/n_samples:.1f}%)")
    print(f"  Failed:                {n_failed}  ({100*n_failed/n_samples:.1f}%)")
    print(f"  Saved to disk:         {n_saved}")
    print(f"  Convergence rate:      {100*n_saved/n_samples:.1f}%")
    print(f"  Total wall time:       {t_sweep_total/60:.1f} min")
    print(f"  Total solver time:     {total_solve_time:.1f} s")
    print(f"  Mean solve time:       {1e3*total_solve_time/n_samples:.1f} ms")
    print(f"  Output directory:      {output_dir}")
    print(f"  Log file:              {log_path}")

    if n_saved > 0:
        saved_files = sorted(output_dir.glob('traj_*.npz'))
        tfs = np.array([float(np.load(f)['t_f']) for f in saved_files])
        mfs = np.array([float(np.load(f)['m_f']) for f in saved_files])
        print(f"\n  Dataset statistics ({n_saved} trajectories):")
        print(f"    t_f:  min={tfs.min():.1f}, max={tfs.max():.1f}, "
              f"mean={tfs.mean():.1f} +/- {tfs.std():.1f} s")
        print(f"    m_f:  min={mfs.min()/1000:.2f}, max={mfs.max()/1000:.2f}, "
              f"mean={mfs.mean()/1000:.2f} +/- {mfs.std()/1000:.2f} t")
        print(f"    fuel: mean={(NOMINAL_MASS-mfs.mean())/1000:.2f} t")


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='RETALT1 PDG Expert Dataset - IC Sweep Harness (FORCESPRO)',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n_samples', type=int, default=5000)
    parser.add_argument('--output_dir', type=str, default='data/trajectories')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--accept_level', type=int, default=1, choices=[0, 1],
                        help='Max status to save: 0=optimal only, 1=include acceptable')
    args = parser.parse_args()

    print("RETALT1 PDG - IC Sweep Harness (FORCESPRO)")
    print(f"  Nominal IC: r = {NOMINAL_POS}, v = {NOMINAL_VEL}")
    print(f"  Dispersion: +/-{POS_DISP} m,  +/-{VEL_DISP} m/s")
    print(f"  Samples:    {args.n_samples}   Seed: {args.seed}   "
          f"Accept: status <= {args.accept_level}\n")

    run_sweep(n_samples=args.n_samples, output_dir=args.output_dir,
              seed=args.seed, dry_run=args.dry_run, accept_level=args.accept_level)


if __name__ == '__main__':
    main()
