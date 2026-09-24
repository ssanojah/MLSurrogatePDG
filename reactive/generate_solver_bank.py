#!/usr/bin/env python3
"""
Generate the medium and short solvers of the reactive expert bank.

    pdg_expert/         N = 61, t_f in [10, 80] s  (must already exist)
    pdg_expert_med/     N = 31, t_f in [ 5, 25] s
    pdg_expert_short/   N = 16, t_f in [ 2, 12] s

Existing directories are skipped. A smoke test on the nominal IC runs after
generation.

Usage:
    python generate_solver_bank.py
"""

import time
import sys
from pathlib import Path

from forces_pdg import generate_solver, solve_ocp, m0
import forcespro.nlp
import numpy as np


# =====================================================================
# Solver bank configuration
# =====================================================================

SOLVER_CONFIGS = {
    'pdg_expert_med': {
        'n_stages': 31,
        'tf_lo': 5.0,
        'tf_hi': 25.0,
        # t_ref defaults to midpoint = 15.0, giving COST_SCALE ≈ 433
    },
    'pdg_expert_short': {
        'n_stages': 16,
        'tf_lo': 2.0,
        'tf_hi': 12.0,
        # t_ref defaults to midpoint = 7.0, giving COST_SCALE ≈ 464
    },
}

# Guadagnini IC for smoke-test verification
GUADAGNINI_IC = np.array([-2386.0, 223.0, 6038.0, 199.0, -18.0, -251.0, 70000.0])
GUADAGNINI_MF_REF = 61490.0  # kg, from IPOPT reference (61.49 t)


def main():
    print("=" * 70)
    print("FORCESPRO Solver Bank Generation")
    print("=" * 70)
    print()
    print("Solvers to generate:")
    for name, cfg in SOLVER_CONFIGS.items():
        print(f"  {name}: N={cfg['n_stages']}, "
              f"tf=[{cfg['tf_lo']:.0f}, {cfg['tf_hi']:.0f}] s")
    print()
    print("NOTE: pdg_expert/ (N=61) is NOT regenerated — using existing.")
    print()

    # ------------------------------------------------------------------
    # Check that the existing long solver directory exists
    # ------------------------------------------------------------------
    if not Path('pdg_expert').is_dir():
        print("WARNING: pdg_expert/ directory not found.")
        print("  The long solver must exist before running the bank.")
        print("  Generate it with: python forces_pdg.py")
        print()

    # ------------------------------------------------------------------
    # Generate medium and short solvers
    # ------------------------------------------------------------------
    for name, cfg in SOLVER_CONFIGS.items():
        print("-" * 70)
        print(f"Generating '{name}' ...")
        print("-" * 70)

        if Path(name).is_dir():
            print(f"  Directory '{name}/' already exists — skipping.")
            print(f"  Delete the directory to force regeneration.")
            continue

        t0 = time.time()
        model, solver = generate_solver(
            n_stages=cfg['n_stages'],
            tf_lo=cfg['tf_lo'],
            tf_hi=cfg['tf_hi'],
            name=name,
        )
        t_gen = time.time() - t0
        print(f"  Generated in {t_gen:.1f} s")
        print()

    # ------------------------------------------------------------------
    # Smoke test: solve Guadagnini IC with each solver that exists
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Smoke Test: Guadagnini IC on all available solvers")
    print("=" * 70)


    all_configs = {
        'pdg_expert': {'n_stages': 61, 'tf_lo': 10.0, 'tf_hi': 80.0},
        **SOLVER_CONFIGS,
    }

    for name, cfg in all_configs.items():
        if not Path(name).is_dir():
            print(f"\n  {name}: directory not found, skipping.")
            continue

        print(f"\n  {name} (N={cfg['n_stages']}):")
        try:
            slv = forcespro.nlp.Solver.from_directory(name)
            # Use tf_guess that's reasonable for each solver's range
            tf_guess = min(50.0, cfg['tf_hi'] * 0.8)
            result = solve_ocp(slv, GUADAGNINI_IC, tf_guess=tf_guess,
                               n_stages=cfg['n_stages'])
            ef = result['exitflag']
            print(f"    exitflag={ef}, t_f={result['tf']:.2f} s, "
                  f"m_f={result['m_f']/1000:.2f} t, "
                  f"iters={result['it']}, "
                  f"solve={result['solvetime']*1e3:.1f} ms, "
                  f"res_eq={result['res_eq']:.2e}")

            if ef == 1 and name == 'pdg_expert':
                err_kg = abs(result['m_f'] - GUADAGNINI_MF_REF)
                print(f"    m_f error vs IPOPT: {err_kg:.0f} kg "
                      f"({100*err_kg/GUADAGNINI_MF_REF:.4f}%)")

            if ef != 1 and name != 'pdg_expert':
                print(f"    (expected — Guadagnini IC tf≈50 s is outside "
                      f"[{cfg['tf_lo']}, {cfg['tf_hi']}] s range)")

        except Exception as e:
            print(f"    EXCEPTION: {e}")

    print("\n" + "=" * 70)
    print("Generation complete.")
    print("=" * 70)
    print("\nNext: python verify_solver_bank.py")


if __name__ == '__main__':
    main()
