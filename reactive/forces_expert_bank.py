#!/usr/bin/env python3
"""
Three-solver FORCESPRO expert bank for the reactive DAgger campaign.

Dispatches each query to the long, medium or short solver according to
the estimated remaining flight time, with fallback to the next longer
solver on failure.

Usage:
    bank = ForcesExpertBank(solver_dir='.')
    result = bank.solve(x0_7, tf_estimate=25.0)
"""

import numpy as np
import time

import forcespro.nlp
from forces_pdg import solve_ocp


# =====================================================================
# Solver bank configuration
# =====================================================================

SOLVER_SPECS = [
    {
        'name': 'pdg_expert',
        'n_stages': 61,
        'tf_lo': 10.0,
        'tf_hi': 80.0,
        'label': 'long',
    },
    {
        'name': 'pdg_expert_med',
        'n_stages': 31,
        'tf_lo': 5.0,
        'tf_hi': 25.0,
        'label': 'medium',
    },
    {
        'name': 'pdg_expert_short',
        'n_stages': 16,
        'tf_lo': 2.0,
        'tf_hi': 12.0,
        'label': 'short',
    },
]

DISPATCH_THRESHOLDS = [
    (15.0, 'long'),
    (8.0,  'medium'),
    (0.0,  'short'),    # catch-all
]

# Fallback order: if primary fails, try the next one up
FALLBACK = {
    'short':  'medium',
    'medium': 'long',
    'long':   None,      # no fallback for the longest solver
}


class ForcesExpertBank:
    """Dispatch layer over three FORCESPRO solvers of different horizon
    lengths.
    """

    def __init__(self, solver_dir='.'):
        """Load all three solvers from their directories.

        Parameters
        ----------
        solver_dir : str
            Base directory containing pdg_expert/, pdg_expert_med/,
            pdg_expert_short/ subdirectories. Default: current directory.
        """
        from pathlib import Path
        base = Path(solver_dir)

        self._solvers = {}
        self._specs = {}

        print("ForcesExpertBank: loading solvers...")
        for spec in SOLVER_SPECS:
            label = spec['label']
            solver_path = base / spec['name']
            if not solver_path.is_dir():
                print(f"  WARNING: {solver_path}/ not found — "
                      f"'{label}' solver unavailable")
                continue

            t0 = time.time()
            slv = forcespro.nlp.Solver.from_directory(str(solver_path))
            t_load = time.time() - t0

            self._solvers[label] = slv
            self._specs[label] = spec
            print(f"  {label:>7s}: N={spec['n_stages']:2d}, "
                  f"tf=[{spec['tf_lo']:.0f}, {spec['tf_hi']:.0f}] s  "
                  f"({t_load:.2f}s)")

        if not self._solvers:
            raise RuntimeError("No solvers loaded — check solver directories")

        # Per-solver statistics
        self._stats = {label: {'queries': 0, 'success': 0, 'fallback': 0,
                               'total_time': 0.0}
                       for label in self._solvers}
        self._stats['_total'] = {'queries': 0, 'success': 0, 'fallback_used': 0}

        print(f"  Bank ready: {len(self._solvers)} solver(s) loaded.\n")

    def select_solver(self, tf_estimate):
        """Select the solver label based on estimated remaining time."""
        # Walk thresholds from longest to shortest
        for threshold, label in DISPATCH_THRESHOLDS:
            if tf_estimate > threshold and label in self._solvers:
                return label

        # If no threshold matched (shouldn't happen with 0.0 catch-all),
        # return whatever is available
        for label in ['short', 'medium', 'long']:
            if label in self._solvers:
                return label

        raise RuntimeError("No solvers available")

    def solve(self, x0_phys, tf_estimate, fallback=True):
        """Solve the OCP from physical state x0, dispatching to the best solver.

        Parameters
        ----------
        x0_phys : array (7,) — physical IC [r, v, m] in SI units.
        tf_estimate : float — estimated remaining time [s].
            Used for solver selection AND as tf_guess for the solve.
        fallback : bool
            If True and primary solver fails, retry with the adjacent
            longer-horizon solver.

        Returns
        -------
        dict — same as forces_pdg.solve_ocp(), plus:
            'solver_used' : str — label of the solver that produced this result
            'fallback_used' : bool — True if the result came from a fallback
        """
        self._stats['_total']['queries'] += 1

        primary_label = self.select_solver(tf_estimate)
        primary_spec = self._specs[primary_label]

        # Clamp tf_guess to the solver's valid range (with margin)
        tf_guess = np.clip(tf_estimate,
                           primary_spec['tf_lo'] * 1.05,
                           primary_spec['tf_hi'] * 0.95)

        # Primary solve
        result = self._try_solve(primary_label, x0_phys, tf_guess)

        if result['exitflag'] == 1:
            self._stats['_total']['success'] += 1
            result['solver_used'] = primary_label
            result['fallback_used'] = False
            return result

        # Fallback: try the next solver up
        if fallback:
            fb_label = FALLBACK.get(primary_label)
            if fb_label and fb_label in self._solvers:
                fb_spec = self._specs[fb_label]
                tf_guess_fb = np.clip(tf_estimate,
                                     fb_spec['tf_lo'] * 1.05,
                                     fb_spec['tf_hi'] * 0.95)

                fb_result = self._try_solve(fb_label, x0_phys, tf_guess_fb)
                self._stats[fb_label]['fallback'] += 1
                self._stats['_total']['fallback_used'] += 1

                if fb_result['exitflag'] == 1:
                    self._stats['_total']['success'] += 1
                    fb_result['solver_used'] = fb_label
                    fb_result['fallback_used'] = True
                    return fb_result

        # Both failed — return the primary result (with failure exitflag)
        result['solver_used'] = primary_label
        result['fallback_used'] = False
        return result

    def _try_solve(self, label, x0_phys, tf_guess):
        """Attempt a single solve with the named solver."""
        spec = self._specs[label]
        solver = self._solvers[label]
        self._stats[label]['queries'] += 1

        try:
            result = solve_ocp(solver, x0_phys, tf_guess=tf_guess,
                               n_stages=spec['n_stages'])
            if result['exitflag'] == 1:
                self._stats[label]['success'] += 1
            self._stats[label]['total_time'] += result['solvetime']
            return result

        except Exception as e:
            return {
                'x_traj': None, 'u_traj': None,
                'tf': 0.0, 't_f': 0.0, 'm_f': 0.0,
                'exitflag': -99, 'it': 0, 'solvetime': 0.0,
                'res_eq': np.inf, 'res_ineq': np.inf,
                'tf_spread': np.inf, 'n_stages': spec['n_stages'],
                '_exception': str(e),
            }

    @property
    def stats(self):
        """Return a copy of the per-solver statistics."""
        return dict(self._stats)

    def print_stats(self):
        """Print a summary of solver bank usage."""
        total = self._stats['_total']
        n = total['queries']
        if n == 0:
            print("  No queries yet.")
            return

        print(f"\n  ForcesExpertBank statistics ({n} total queries):")
        print(f"    Overall success rate: "
              f"{total['success']}/{n} ({100*total['success']/n:.1f}%)")
        print(f"    Fallback used: {total['fallback_used']} times")

        for label in ['long', 'medium', 'short']:
            if label not in self._stats:
                continue
            s = self._stats[label]
            if s['queries'] == 0:
                continue
            rate = 100 * s['success'] / s['queries'] if s['queries'] > 0 else 0
            mean_t = 1e3 * s['total_time'] / s['queries'] if s['queries'] > 0 else 0
            print(f"    {label:>7s}: {s['queries']:5d} queries, "
                  f"{rate:.0f}% success, {mean_t:.1f} ms mean, "
                  f"{s['fallback']} fallback")

    def reset_stats(self):
        """Reset all statistics (e.g. between DAgger iterations)."""
        for label in self._stats:
            for key in self._stats[label]:
                self._stats[label][key] = 0 if isinstance(
                    self._stats[label][key], int) else 0.0


# =====================================================================
# Quick self-test
# =====================================================================

if __name__ == '__main__':
    print("ForcesExpertBank — self-test")
    print("=" * 50)

    bank = ForcesExpertBank()

    # Test dispatch logic (no solves, just selection)
    test_cases = [50.0, 20.0, 12.0, 8.0, 5.0, 3.0]
    print("\nDispatch test:")
    for tf_est in test_cases:
        label = bank.select_solver(tf_est)
        print(f"  tf_estimate={tf_est:5.1f} s  →  {label}")

    # Solve Guadagnini IC (should go to long solver)
    from forces_pdg import m0
    x0 = np.array([-2386.0, 223.0, 6038.0, 199.0, -18.0, -251.0, m0])

    print(f"\nSolving Guadagnini IC (tf_estimate=50)...")
    result = bank.solve(x0, tf_estimate=50.0)
    print(f"  solver_used: {result['solver_used']}")
    print(f"  exitflag: {result['exitflag']}")
    print(f"  t_f: {result['tf']:.2f} s")
    print(f"  m_f: {result['m_f']/1000:.2f} t")
    print(f"  solve time: {result['solvetime']*1e3:.1f} ms")

    bank.print_stats()
