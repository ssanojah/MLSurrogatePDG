#!/usr/bin/env python3
"""
IPOPT expert interface for DAgger.

Wraps ipopt_pdg.py behind query(), which returns the expert's first action
for a given state.

Usage:
    expert = ExpertSolver(verbose=False)
    result = expert.query(x0_7, tf_guess=50.0)
    if result['feasible']:
        u_opt = result['u_opt']
"""

import numpy as np
import time
import sys

# =====================================================================
# solve_pdg return format:
# =====================================================================
from ipopt_pdg import solve_pdg, ParametricPDGSolver


class ExpertSolver:
    """Solver-agnostic expert interface for DAgger.

    Parameters
    ----------
    verbose : bool
        If True, print IPOPT output for each solve.
        Set to False for DAgger batch queries (hundreds of solves).
    parametric : bool
        If True (default), use ParametricPDGSolver -- builds the NLP
        once and reuses it across queries (~2x faster per query).
        If False, use solve_pdg(), which rebuilds the NLP every call
        (slower, but useful for debugging).
    tf_min, tf_max : float, optional
        Free-final-time bounds passed to ParametricPDGSolver. Leave as
        None to use the dataset-generation defaults ([10, 80] s).
        DAgger expert instances should pass tf_min ~ 3 s: at late
        query nodes the natural remaining flight time falls below
        10 s, and an active tf floor produces time-dilated labels
        that are not the optimal continuation of the trajectory.
        Only supported with parametric=True (solve_pdg uses the
        module-level bounds and cannot be overridden per-call).
    """

    # IPOPT status strings that count as convergence
    _SUCCESS_STATUSES = {'Solve_Succeeded', 'Solved_To_Acceptable_Level'}

    def __init__(self, verbose=False, parametric=True,
                 tf_min=None, tf_max=None):
        self.verbose = verbose
        self._parametric = parametric

        if parametric:
            print("ExpertSolver: building parametric NLP (one-time cost)...")
            self._solver = ParametricPDGSolver(
                print_level=5 if verbose else 0,
                tf_min=tf_min,
                tf_max=tf_max,
            )
            print(f"  Built in {self._solver.build_time:.2f} s")
            print(f"  t_f bounds: [{self._solver._tf_min:.1f}, "
                  f"{self._solver._tf_max:.1f}] s")
        else:
            if tf_min is not None or tf_max is not None:
                raise ValueError(
                    "Custom t_f bounds require parametric=True "
                    "(solve_pdg uses module-level bounds).")
            self._solver = None
            print("ExpertSolver: using non-parametric solve_pdg() "
                  "(NLP rebuilt each call)")

        self._n_queries = 0
        self._n_success = 0
        self._n_failed = 0
        self._total_solve_time = 0.0

    def query(self, x0, tf_guess=50.0):
        """Solve the fuel-optimal OCP from physical state x0.

        Parameters
        ----------
        x0 : np.ndarray, shape (7,)
            Physical state [r_x, r_y, r_z, v_x, v_y, v_z, m].
            Units: meters, m/s, kg.
        tf_guess : float
            Initial guess for the free final time [s]. Better guesses
            reduce IPOPT iteration count and solve time. For DAgger
            queries along the same rollout, the previous query's tf_opt
            minus elapsed time is a good guess.

        Returns
        -------
        dict with:
            'u_opt'      : np.ndarray (3,) -- optimal first action [MN],
                           or None if solver failed
            'tf_opt'     : float -- optimal final time [s], or None
            'status'     : str -- IPOPT return status string
            'feasible'   : bool -- True if solver converged
            'solve_time' : float -- wall-clock solve time [s]
        """
        self._n_queries += 1
        t_start = time.time()

        try:
            if self._parametric:
                result = self._solver.solve(
                    x0_phys=np.asarray(x0, dtype=float),
                    tf_guess=float(tf_guess),
                    verbose=self.verbose,
                )
            else:
                result = solve_pdg(
                    x0_phys=np.asarray(x0, dtype=float),
                    tf_guess=float(tf_guess),
                    verbose=self.verbose,
                )
            solve_time = time.time() - t_start
            return self._extract_result(result, solve_time)

        except Exception as e:
            solve_time = time.time() - t_start
            self._n_failed += 1
            return {
                'u_opt': None,
                'tf_opt': None,
                'status': f'exception: {type(e).__name__}: {e}',
                'feasible': False,
                'solve_time': solve_time,
            }

    def _extract_result(self, result, solve_time):
        """Extract the DAgger-relevant fields from solve_pdg output."""
        status = result['status']
        feasible = status in self._SUCCESS_STATUSES

        if feasible:
            self._n_success += 1
            self._total_solve_time += solve_time

            # First-step action: row 0 of the control trajectory
            u_opt = np.array(result['u_traj'][0], dtype=float).flatten()

            # Handle both key names for final time
            if 't_f' in result:
                tf_opt = float(result['t_f'])
            else:
                tf_opt = float(result['tf'])

            return {
                'u_opt': u_opt,       # (3,) thrust in MN
                'tf_opt': tf_opt,     # optimal final time in seconds
                'status': status,
                'feasible': True,
                'solve_time': solve_time,
            }
        else:
            self._n_failed += 1
            return {
                'u_opt': None,
                'tf_opt': None,
                'status': status,
                'feasible': False,
                'solve_time': solve_time,
            }

    def query_batch(self, states, tf_guesses=None, subsample_indices=None,
                    progress=True):
        """Query the expert at multiple states from a single rollout.

        Parameters
        ----------
        states : np.ndarray, shape (T, 7)
            Physical states at each timestep of a policy rollout.
        tf_guesses : np.ndarray, shape (T,), optional
            Per-state tf guess. If None, uses 50.0 for all.
        subsample_indices : array-like of int, optional
            Which timestep indices to actually query. States at other
            indices are skipped (result entry is None). If None, queries
            all states.
        progress : bool
            If True, print a progress line every 10 queries.

        Returns
        -------
        list of (dict or None), length T.
            query() result at each index, or None for skipped indices.
        """
        T = len(states)
        if tf_guesses is None:
            tf_guesses = np.full(T, 50.0)
        if subsample_indices is None:
            query_set = set(range(T))
        else:
            query_set = set(subsample_indices)

        results = [None] * T
        n_queried = 0
        n_feasible = 0

        for i in range(T):
            if i not in query_set:
                continue

            result = self.query(states[i], tf_guesses[i])
            results[i] = result
            n_queried += 1
            if result['feasible']:
                n_feasible += 1

            if progress and n_queried % 10 == 0:
                rate = n_feasible / n_queried if n_queried > 0 else 0
                print(f"  Expert queries: {n_queried}/{len(query_set)} "
                      f"done, {rate:.0%} feasible, "
                      f"last solve: {result['solve_time']:.1f}s",
                      flush=True)

        if progress:
            rate = n_feasible / n_queried if n_queried > 0 else 0
            print(f"  Batch complete: {n_queried} queries, "
                  f"{n_feasible} feasible ({rate:.0%})")

        return results

    @property
    def stats(self):
        """Cumulative query statistics as a dict."""
        total = self._n_queries
        return {
            'total_queries': total,
            'successful': self._n_success,
            'failed': self._n_failed,
            'success_rate': self._n_success / max(total, 1),
            'mean_solve_time_s': (self._total_solve_time
                                  / max(self._n_success, 1)),
        }

    def reset_stats(self):
        """Reset cumulative statistics (e.g. between DAgger iterations)."""
        self._n_queries = 0
        self._n_success = 0
        self._n_failed = 0
        self._total_solve_time = 0.0

    def __repr__(self):
        s = self.stats
        mode = 'parametric' if self._parametric else 'non-parametric'
        return (f"ExpertSolver({mode}, queries={s['total_queries']}, "
                f"success_rate={s['success_rate']:.1%}, "
                f"mean_time={s['mean_solve_time_s']:.2f}s)")
