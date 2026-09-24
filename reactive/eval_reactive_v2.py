"""
Closed-loop evaluation of the reactive Transformer, with engine cutoff.

As eval_reactive.py, plus an engine cutoff when the vehicle starts climbing
(v_z > --cutoff_vz), followed by an unpowered coast of up to --coast_time
seconds. --no_cutoff disables it.

Usage:
    python eval_reactive_v2.py --run_dir runs/reactive_w16 \\
        --data_dir data/<dataset> --out_dir results/reactive_eval_v2
"""

import numpy as np
import torch
import argparse
import json
import time
import csv
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

# Dynamics from ipopt_pdg
from ipopt_pdg import (
    build_dynamics_function, build_rk4_integrator,
    nx, nu,
    m_dry, m0,
    T_min_MN, T_max_MN,
    cos_tilt, theta_max,
    gamma_gs,
    g0,
)

from reactive_model import ReactiveTransformer, ReactiveConfig
from windowed_dataset import (
    ReactiveNormStats, split_by_trajectory, STATE_CHANNELS, STATE_DIM,
)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except ImportError:                                   # pragma: no cover
    _HAVE_MPL = False


# =====================================================================
# Landing criteria constants
# =====================================================================

THR_RH_SOFT = 10.0     # m     — GR-03  horizontal position, norm
THR_VZ_SOFT = 2.0      # m/s   — GR-04  vertical speed
THR_VXY_SOFT = 1.5     # m/s   — GR-05/06 horizontal speed, PER AXIS
THR_VZ_HARD = 10.0     # m/s   — Rosa et al.
THR_VH_HARD = 3.0      # m/s   — Rosa et al., NORM (not per axis)

# Outcome categories in a fixed order used by counts, CSV and plots.
OUTCOME_ORDER = ['soft', 'hard', 'crash', 'airborne',
                 'fuel_exhausted', 'diverged', 'nan']

PLOT_CLASSES = ['soft', 'hard', 'crash', 'airborne', 'fuel_exhausted']

CLASS_COLORS = {
    'soft':           '#2E7D32',   # green
    'hard':           '#F9A825',   # amber
    'crash':          '#C62828',   # red
    'airborne':       '#1565C0',   # blue
    'fuel_exhausted': '#6A1B9A',   # purple
}

CLASS_LABELS = {
    'soft':           'Soft',
    'hard':           'Hard',
    'crash':          'Crash',
    'airborne':       'Airborne (timeout)',
    'fuel_exhausted': 'Fuel exhausted',
}


# =====================================================================
# Configuration
# =====================================================================

@dataclass
class ReactiveEvalConfig:
    """All evaluation parameters in one place for reproducibility."""

    # Paths
    data_dir: str = 'data/ForcesLargeBatch'
    run_dir: str = 'runs/reactive_w16'
    out_dir: str = 'results/reactive_eval'

    # Integration
    dt: float = 0.8
    t_max: float = 80.0        # maximum simulation time [s], POWERED phase

    # --- Engine cutoff + ballistic coast -----------------------------
    cutoff_on_ascent: bool = True    # MECO the moment the vehicle climbs
    cutoff_vz_ms: float = 0.0        # v_z above this [m/s] triggers cutoff
    coast_time: float = 10.0         # cap on the unpowered fall [s]
    coast_on_fuel: bool = False      # also coast after fuel exhaustion

    # Constraint enforcement
    clip_thrust: bool = True    # clip ‖T‖ to [T_min, T_max]
    enforce_tilt: bool = True   # project thrust to satisfy tilt constraint

    # Safety limits (early termination)
    max_position_norm: float = 50_000.0   # 50 km — clearly diverged

    # Plotting
    make_plots: bool = True
    plot_tilt_source: str = 'raw'       # 'raw' (network output) or 'enforced'
    plot_include_airborne: bool = True
    plot_bins: int = 30
    plot_log_y: bool = False

    # Data split (MUST match training split exactly)
    seed: int = 42
    train_frac: float = 0.70
    val_frac: float = 0.15

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @property
    def max_steps(self) -> int:
        """Maximum number of POWERED simulation steps before timeout."""
        return int(self.t_max / self.dt) + 1

    @property
    def max_coast_steps(self) -> int:
        """Maximum number of unpowered steps after engine cutoff."""
        return max(1, int(round(self.coast_time / self.dt)))


# =====================================================================
# Landing classification
# =====================================================================

def classify_landing(terminal_state: np.ndarray,
                     reached_ground: bool,
                     term_reason: str = 'timeout') -> str:
    """Classify the outcome of a single rollout.

    Parameters
    ----------
    terminal_state : (7,) [r_x, r_y, r_z, v_x, v_y, v_z, m]
    reached_ground : True if r_z reached 0
    term_reason    : 'touchdown' | 'timeout' | 'fuel_exhausted'
                     | 'diverged' | 'nan'

    Returns
    -------
    One of OUTCOME_ORDER.
    """
    if term_reason == 'nan' or not np.all(np.isfinite(terminal_state)):
        return 'nan'
    if term_reason == 'diverged':
        return 'diverged'

    if not reached_ground:
        return 'fuel_exhausted' if term_reason == 'fuel_exhausted' \
            else 'airborne'

    r_x, r_y, r_z, v_x, v_y, v_z, m = terminal_state

    r_h_norm = np.sqrt(r_x**2 + r_y**2)
    v_h_norm = np.sqrt(v_x**2 + v_y**2)

    # GR-03..GR-06: soft landing. Horizontal velocity is PER AXIS here.
    soft = (r_h_norm <= THR_RH_SOFT and
            abs(v_z) <= THR_VZ_SOFT and
            abs(v_x) <= THR_VXY_SOFT and
            abs(v_y) <= THR_VXY_SOFT)
    if soft:
        return 'soft'

    # Rosa et al.: hard landing. Horizontal velocity is a NORM here.
    hard = (abs(v_z) <= THR_VZ_HARD and v_h_norm <= THR_VH_HARD)
    if hard:
        return 'hard'

    return 'crash'


# =====================================================================
# Constraint enforcement
# =====================================================================

def enforce_thrust_constraints(
    T_c: np.ndarray,
    clip_thrust: bool = True,
    enforce_tilt: bool = True,
) -> np.ndarray:
    """Apply physical thrust constraints to the raw NN output.

    Parameters
    ----------
    T_c : (3,) raw thrust in MN [T_x, T_y, T_z]

    Returns
    -------
    T_c : (3,) constrained thrust in MN
    """
    T_c = T_c.copy()

    if enforce_tilt:
        T_mag = np.linalg.norm(T_c)
        if T_mag > 1e-10:
            cos_angle = T_c[2] / T_mag    # T_z / ‖T‖
            if cos_angle < cos_tilt:
                # Project: keep horizontal direction, increase T_z
                T_h = T_c[:2]
                T_h_mag = np.linalg.norm(T_h)
                if T_h_mag > 1e-10:
                    # Set T_z such that angle = θ_max exactly
                    T_z_new = T_mag * cos_tilt
                    # Scale horizontal to preserve magnitude
                    T_h_new_mag = np.sqrt(max(T_mag**2 - T_z_new**2, 0.0))
                    T_c[:2] = T_h * (T_h_new_mag / T_h_mag)
                    T_c[2] = T_z_new
                else:
                    # Purely vertical — no tilt issue, but may point down
                    T_c[2] = abs(T_c[2])

    # 2. Thrust magnitude clipping
    if clip_thrust:
        T_mag = np.linalg.norm(T_c)
        if T_mag > 1e-10:
            if T_mag < T_min_MN:
                T_c *= T_min_MN / T_mag
            elif T_mag > T_max_MN:
                T_c *= T_max_MN / T_mag

    return T_c


def tilt_angle_deg(T: np.ndarray) -> float:
    """Angle of a thrust vector from vertical, in degrees."""
    T_mag = np.linalg.norm(T)
    if T_mag < 1e-10:
        return float('nan')
    return float(np.degrees(np.arccos(np.clip(T[2] / T_mag, -1.0, 1.0))))


# =====================================================================
# Touchdown bisection
# =====================================================================

def bisect_touchdown(
    x_before: np.ndarray,
    x_after: np.ndarray,
    u: np.ndarray,
    dt: float,
    F_rk4,
    tol: float = 0.1,
    max_iter: int = 20,
) -> np.ndarray:
    """Bisection to find the state at r_z ≈ 0 between two integration steps."""
    dt_lo, dt_hi = 0.0, dt
    x_td = x_after.copy()

    for _ in range(max_iter):
        dt_mid = (dt_lo + dt_hi) / 2.0
        x_mid = np.array(F_rk4(x_before, u, dt_mid)).flatten()

        if abs(x_mid[2]) < tol:
            x_td = x_mid
            break
        elif x_mid[2] > 0:
            dt_lo = dt_mid
            x_td = x_mid
        else:
            dt_hi = dt_mid
            x_td = x_mid

    # Clamp altitude to exactly 0 for classification
    x_td[2] = 0.0
    return x_td


# =====================================================================
# Single-IC rollout
# =====================================================================

def rollout_reactive(
    model: torch.nn.Module,
    norm_stats: ReactiveNormStats,
    x0_7: np.ndarray,
    window_size: int,
    F_rk4,
    config: ReactiveEvalConfig,
    device: torch.device = torch.device('cpu'),
) -> dict:
    """Roll out the reactive policy from one IC until a physical stopping
    condition fires, then coast unpowered if the engine was cut.

    Parameters
    ----------
    model       : trained ReactiveTransformer (in eval mode)
    norm_stats  : ReactiveNormStats from training
    x0_7        : initial state [r_x, r_y, r_z, v_x, v_y, v_z, m] (7,)
    window_size : W, must match model's window_size
    F_rk4       : CasADi RK4 integrator function
    config      : evaluation config
    device      : torch device

    Returns
    -------
    dict with rollout data and metadata
    """
    dt = config.dt
    max_steps = config.max_steps
    W = window_size

    x_history = [x0_7.copy()]
    u_history = []              # enforced thrust, actually integrated
    u_raw_history = []          # raw network output, before enforcement
    tilt_deg_history = []       # tilt of enforced thrust
    tilt_raw_deg_history = []   # tilt of raw network output
    T_mag_history = []          # magnitude of enforced thrust
    T_mag_raw_history = []      # magnitude of raw network output
    gs_margin_history = []

    term_reason = 'timeout'
    reached_ground = False

    # --- cutoff bookkeeping ---
    engine_cutoff = False
    cutoff_reason = ''
    cutoff_state = None

    for k in range(max_steps):
        x_k = x_history[-1]   # current physical state (7,)

        # ---- 1. Build sliding window (W, 7) ----
        n_available = len(x_history)   # how many states we have (k+1)
        window_raw = np.zeros((W, STATE_DIM), dtype=np.float32)
        mask = np.ones(W, dtype=bool)   # True = padding

        if n_available >= W:
            # Full window: take last W states
            window_raw[:] = np.array(x_history[-W:], dtype=np.float32)
            mask[:] = False
        else:
            # Left-pad with x_0
            n_pad = W - n_available
            window_raw[:n_pad] = np.array(x_history[0], dtype=np.float32)
            window_raw[n_pad:] = np.array(x_history[-n_available:],
                                           dtype=np.float32)
            mask[:n_pad] = True
            mask[n_pad:] = False

        window_norm = norm_stats.normalize_state(window_raw)

        # ---- 2. Forward pass ----
        window_t = torch.from_numpy(window_norm).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_norm = model(window_t, padding_mask=mask_t)  # (1, 3)

        # ---- 3. Denormalize to physical thrust [MN] ----
        T_raw = norm_stats.unnormalize_action(pred_norm[0].cpu().numpy())

        # ---- 4. Enforce constraints ----
        T_c = enforce_thrust_constraints(
            T_raw, config.clip_thrust, config.enforce_tilt)

        # ---- 5. Record pre-integration metrics ----
        T_mag = float(np.linalg.norm(T_c))
        T_mag_raw = float(np.linalg.norm(T_raw))
        r_h = np.sqrt(x_k[0]**2 + x_k[1]**2)
        gs_margin = x_k[2] - np.tan(gamma_gs) * r_h

        u_history.append(T_c.copy())
        u_raw_history.append(T_raw.copy())
        T_mag_history.append(T_mag)
        T_mag_raw_history.append(T_mag_raw)
        tilt_deg_history.append(tilt_angle_deg(T_c))
        tilt_raw_deg_history.append(tilt_angle_deg(T_raw))
        gs_margin_history.append(gs_margin)

        # ---- 6. Propagate dynamics ----
        x_next = np.array(F_rk4(x_k, T_c, dt)).flatten()

        # ---- 7. Check stopping conditions ----

        if np.any(np.isnan(x_next)):
            x_history.append(x_next)
            term_reason = 'nan'
            break

        # Ground contact
        if x_next[2] <= 0.0:
            x_td = bisect_touchdown(x_k, x_next, T_c, dt, F_rk4)
            x_history.append(x_td)
            reached_ground = True
            term_reason = 'touchdown'
            break

        if config.cutoff_on_ascent and x_next[5] > config.cutoff_vz_ms:
            x_history.append(x_next)
            engine_cutoff = True
            cutoff_reason = 'ascent'
            cutoff_state = x_next.copy()
            break

        # Divergence
        if np.linalg.norm(x_next[:3]) > config.max_position_norm:
            x_history.append(x_next)
            term_reason = 'diverged'
            break

        # Fuel exhaustion
        if x_next[6] <= m_dry:
            x_history.append(x_next)
            term_reason = 'fuel_exhausted'
            if config.coast_on_fuel:
                engine_cutoff = True
                cutoff_reason = 'fuel_exhausted'
                cutoff_state = x_next.copy()
            break

        x_history.append(x_next)

    else:
        # Loop completed without break → timeout
        term_reason = 'timeout'

    n_powered_steps = len(u_history)

    # ---- Terminal attitude proxy: LAST COMMANDED thrust ----
    terminal_tilt_deg = (float(tilt_deg_history[-1])
                         if tilt_deg_history else float('nan'))
    terminal_tilt_raw_deg = (float(tilt_raw_deg_history[-1])
                             if tilt_raw_deg_history else float('nan'))

    # =================================================================
    # BALLISTIC COAST — engine off, no network influence
    # =================================================================
    n_coast_steps = 0
    if engine_cutoff:
        u_zero = np.zeros(3)
        coast_outcome = 'still airborne'

        for _j in range(config.max_coast_steps):
            x_k = x_history[-1]

            u_history.append(u_zero.copy())
            u_raw_history.append(u_zero.copy())
            T_mag_history.append(0.0)
            T_mag_raw_history.append(0.0)
            tilt_deg_history.append(float('nan'))
            tilt_raw_deg_history.append(float('nan'))
            r_h = np.sqrt(x_k[0]**2 + x_k[1]**2)
            gs_margin_history.append(x_k[2] - np.tan(gamma_gs) * r_h)

            # No thrust enforcement during the coast (u = 0).
            x_next = np.array(F_rk4(x_k, u_zero, dt)).flatten()
            n_coast_steps += 1

            if np.any(np.isnan(x_next)):
                x_history.append(x_next)
                term_reason = 'nan'
                coast_outcome = 'NaN'
                break

            if x_next[2] <= 0.0:
                x_td = bisect_touchdown(x_k, x_next, u_zero, dt, F_rk4)
                x_history.append(x_td)
                reached_ground = True
                term_reason = 'touchdown'
                coast_outcome = 'ground contact'
                break

            if np.linalg.norm(x_next[:3]) > config.max_position_norm:
                x_history.append(x_next)
                term_reason = 'diverged'
                coast_outcome = 'diverged'
                break

            x_history.append(x_next)
        else:
            term_reason = ('fuel_exhausted'
                           if cutoff_reason == 'fuel_exhausted'
                           else 'timeout')
            coast_outcome = 'still airborne'
    else:
        coast_outcome = ''

    # ---- Build result arrays ----
    x_traj = np.array(x_history)          # (n_steps+1, 7)
    u_traj = np.array(u_history)          # (n_steps, 3)
    u_raw = np.array(u_raw_history)       # (n_steps, 3)
    n_steps = len(u_history)

    terminal_state = x_traj[-1]
    classification = classify_landing(terminal_state, reached_ground,
                                      term_reason)

    elapsed_time = n_steps * dt

    return {
        'x_traj': x_traj,
        'u_traj': u_traj,
        'u_raw': u_raw,
        'n_steps': n_steps,
        'elapsed_time': elapsed_time,
        'terminal_state': terminal_state,
        'classification': classification,
        'term_reason': term_reason,
        'reached_ground': reached_ground,
        'tilt_deg': np.array(tilt_deg_history),
        'tilt_raw_deg': np.array(tilt_raw_deg_history),
        'T_mag_MN': np.array(T_mag_history),
        'T_mag_raw_MN': np.array(T_mag_raw_history),
        'gs_margin': np.array(gs_margin_history),
        'terminal_tilt_deg': terminal_tilt_deg,
        'terminal_tilt_raw_deg': terminal_tilt_raw_deg,
        'fuel_used_kg': float(m0 - terminal_state[6]),
        # --- cutoff / coast diagnostics ---
        'engine_cutoff': engine_cutoff,
        'cutoff_reason': cutoff_reason,
        'cutoff_state': cutoff_state,
        'coast_outcome': coast_outcome,
        'n_powered_steps': n_powered_steps,
        'n_coast_steps': n_coast_steps,
        'powered_time': n_powered_steps * dt,
        'coast_time_s': n_coast_steps * dt,
    }


# =====================================================================
# Batch evaluation
# =====================================================================

def evaluate_batch(
    model, norm_stats, test_files, window_size, F_rk4, config, device,
):
    """Evaluate the reactive policy on all test ICs."""
    results = []
    counts = {k: 0 for k in OUTCOME_ORDER}
    n_total = len(test_files)

    t_start = time.time()

    for i, f in enumerate(test_files):
        data = np.load(f)
        x0_7 = data['x_traj'][0, :7].astype(np.float64)
        expert_tf = float(data['t_f'])
        expert_mf = float(data['m_f'])

        result = rollout_reactive(
            model, norm_stats, x0_7, window_size, F_rk4, config, device)

        result['ic_index'] = i
        result['filepath'] = str(f)
        result['expert_tf'] = expert_tf
        result['expert_mf'] = expert_mf
        results.append(result)

        cls = result['classification']
        counts[cls] += 1

        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed
        eta = (n_total - i - 1) / rate if rate > 0 else 0

        ts = result['terminal_state']
        symbol = {'soft': '✓', 'hard': '~', 'crash': '✗',
                  'airborne': '↑', 'diverged': '→', 'nan': '!',
                  'fuel_exhausted': '⛽'}.get(cls, '?')
        meco = (f" MECO+{result['coast_time_s']:.1f}s"
                if result['engine_cutoff'] else "")

        print(f"  [{i+1:4d}/{n_total}] {symbol} {cls:>14s}  "
              f"steps={result['n_steps']:3d}  "
              f"t={result['elapsed_time']:.1f}s  "
              f"r_z={ts[2]:.1f}m  "
              f"v_z={ts[5]:.1f}m/s{meco}  "
              f"ETA={eta:.0f}s")

    t_total = time.time() - t_start

    summary = {
        'n_total': n_total,
        'counts': counts,
        'rates': {k: v / n_total * 100 for k, v in counts.items()},
        'eval_time_s': t_total,
        'dt': config.dt,
        't_max': config.t_max,
        'clip_thrust': config.clip_thrust,
        'enforce_tilt': config.enforce_tilt,
        'cutoff_on_ascent': config.cutoff_on_ascent,
        'cutoff_vz_ms': config.cutoff_vz_ms,
        'coast_time_cap_s': config.coast_time,
        'coast_on_fuel': config.coast_on_fuel,
    }

    # --- Engine-cutoff statistics ---
    cut = [r for r in results if r['engine_cutoff']]
    summary['cutoff_stats'] = {
        'n_cutoff': len(cut),
        'rate_pct': len(cut) / n_total * 100 if n_total else 0.0,
        'n_reached_ground_during_coast': sum(1 for r in cut
                                             if r['reached_ground']),
        'coast_time_mean_s': (float(np.mean([r['coast_time_s']
                                             for r in cut]))
                              if cut else 0.0),
        'coast_time_max_s': (float(np.max([r['coast_time_s']
                                           for r in cut]))
                             if cut else 0.0),
        'cutoff_altitude_mean_m': (
            float(np.mean([r['cutoff_state'][2] for r in cut]))
            if cut else float('nan')),
        'by_class': {c: sum(1 for r in cut if r['classification'] == c)
                     for c in OUTCOME_ORDER},
        'by_reason': {reason: sum(1 for r in cut
                                  if r['cutoff_reason'] == reason)
                      for reason in ('ascent', 'fuel_exhausted')},
    }

    # Terminal statistics for ICs that actually reached the ground
    landed = [r for r in results if r['reached_ground']]
    if landed:
        ts_arr = np.array([r['terminal_state'] for r in landed])
        tilt_arr = np.array([r['terminal_tilt_raw_deg'] for r in landed])
        tilt_arr = tilt_arr[np.isfinite(tilt_arr)]
        summary['landed_stats'] = {
            'n_landed': len(landed),
            'r_h_mean': float(np.sqrt(ts_arr[:, 0]**2 + ts_arr[:, 1]**2).mean()),
            'r_h_max': float(np.sqrt(ts_arr[:, 0]**2 + ts_arr[:, 1]**2).max()),
            'v_z_mean': float(np.abs(ts_arr[:, 5]).mean()),
            'v_z_max': float(np.abs(ts_arr[:, 5]).max()),
            'v_h_mean': float(np.sqrt(ts_arr[:, 3]**2 + ts_arr[:, 4]**2).mean()),
            'v_h_max': float(np.sqrt(ts_arr[:, 3]**2 + ts_arr[:, 4]**2).max()),
            'tilt_raw_mean_deg': (float(tilt_arr.mean())
                                  if tilt_arr.size else float('nan')),
            'tilt_raw_max_deg': (float(tilt_arr.max())
                                 if tilt_arr.size else float('nan')),
            'fuel_mean_kg': float(np.array(
                [r['fuel_used_kg'] for r in landed]).mean()),
            'steps_mean': float(np.array(
                [r['n_steps'] for r in landed]).mean()),
        }

    aloft = [r for r in results
             if not r['reached_ground']
             and r['classification'] in ('airborne', 'fuel_exhausted')]
    if aloft:
        rz = np.array([r['terminal_state'][2] for r in aloft])
        rz = rz[np.isfinite(rz)]
        if rz.size:
            summary['aloft_stats'] = {
                'n_aloft': len(aloft),
                'r_z_mean': float(rz.mean()),
                'r_z_median': float(np.median(rz)),
                'r_z_min': float(rz.min()),
                'r_z_max': float(rz.max()),
                'n_below_50m': int((rz <= 50.0).sum()),
            }

    return results, summary


# =====================================================================
# Reporting
# =====================================================================

def print_summary(summary: dict):
    """Print human-readable evaluation summary."""
    print(f"\n{'=' * 60}")
    print("REACTIVE EVALUATION SUMMARY")
    print(f"{'=' * 60}")

    n = summary['n_total']
    c = summary['counts']
    r = summary['rates']

    print(f"\n  Test ICs:        {n}")
    print(f"  Integration dt:  {summary['dt']:.2f} s")
    print(f"  Max time:        {summary['t_max']:.0f} s")
    print(f"  Thrust clipping: {'ON' if summary['clip_thrust'] else 'OFF'}")
    print(f"  Tilt enforcement:{'ON' if summary['enforce_tilt'] else 'OFF'}")
    if summary.get('cutoff_on_ascent'):
        print(f"  Engine cutoff:   ON  (v_z > "
              f"{summary['cutoff_vz_ms']:.2f} m/s), then up to "
              f"{summary['coast_time_cap_s']:.0f} s ballistic coast")
    else:
        print(f"  Engine cutoff:   OFF (v1 behaviour)")

    print(f"\n  Landing outcomes:")
    for k in OUTCOME_ORDER:
        print(f"    {CLASS_LABELS.get(k, k.capitalize()):<20s} "
              f"{c[k]:4d}  ({r[k]:.1f}%)")

    reached = c['soft'] + c['hard'] + c['crash']
    print(f"\n    Reached ground:      {reached:4d}  ({reached/n*100:.1f}%)")

    if 'cutoff_stats' in summary and summary['cutoff_stats']['n_cutoff']:
        cs = summary['cutoff_stats']
        print(f"\n  Engine cutoff fired on {cs['n_cutoff']}/{n} ICs "
              f"({cs['rate_pct']:.1f}%):")
        print(f"    reached ground during coast: "
              f"{cs['n_reached_ground_during_coast']}/{cs['n_cutoff']}")
        print(f"    cutoff altitude: mean {cs['cutoff_altitude_mean_m']:.1f} m")
        print(f"    coast duration:  mean {cs['coast_time_mean_s']:.2f} s,  "
              f"max {cs['coast_time_max_s']:.2f} s")
        print(f"    post-coast classification: "
              + ", ".join(f"{CLASS_LABELS.get(k, k)}={v}"
                          for k, v in cs['by_class'].items() if v))

    if 'landed_stats' in summary:
        ls = summary['landed_stats']
        print(f"\n  Landed IC statistics (n={ls['n_landed']}):")
        print(f"    ‖r_h‖:  mean = {ls['r_h_mean']:.1f} m,  "
              f"max = {ls['r_h_max']:.1f} m")
        print(f"    |v_z|:  mean = {ls['v_z_mean']:.1f} m/s,  "
              f"max = {ls['v_z_max']:.1f} m/s")
        print(f"    ‖v_h‖:  mean = {ls['v_h_mean']:.1f} m/s,  "
              f"max = {ls['v_h_max']:.1f} m/s")
        print(f"    tilt (raw NN): mean = {ls['tilt_raw_mean_deg']:.1f}°,  "
              f"max = {ls['tilt_raw_max_deg']:.1f}°")
        print(f"    fuel:   mean = {ls['fuel_mean_kg']/1000:.2f} t")
        print(f"    steps:  mean = {ls['steps_mean']:.0f}")

    if 'aloft_stats' in summary:
        als = summary['aloft_stats']
        print(f"\n  Non-landed IC altitude (n={als['n_aloft']}):")
        print(f"    r_z:    mean = {als['r_z_mean']:.1f} m,  "
              f"median = {als['r_z_median']:.1f} m")
        print(f"            min  = {als['r_z_min']:.1f} m,  "
              f"max = {als['r_z_max']:.1f} m")
        print(f"    within 50 m of ground: {als['n_below_50m']}"
              f"/{als['n_aloft']}")

    print(f"\n  Evaluation time: {summary['eval_time_s']:.1f} s")


def save_per_ic_csv(results: list, filepath: Path):
    """Save per-IC results to CSV for post-processing."""
    fields = ['ic_index', 'classification', 'term_reason', 'reached_ground',
              'n_steps', 'elapsed_time',
              'r_x', 'r_y', 'r_z', 'v_x', 'v_y', 'v_z', 'm',
              'terminal_tilt_deg', 'terminal_tilt_raw_deg',
              'fuel_used_kg', 'expert_tf', 'expert_mf',
              # --- cutoff / coast diagnostics ---
              'engine_cutoff', 'cutoff_reason', 'coast_outcome',
              'n_powered_steps', 'n_coast_steps',
              'powered_time', 'coast_time_s', 'cutoff_r_z', 'cutoff_v_z']

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            ts = r['terminal_state']
            cs = r['cutoff_state']
            writer.writerow({
                'ic_index': r['ic_index'],
                'classification': r['classification'],
                'term_reason': r['term_reason'],
                'reached_ground': int(bool(r['reached_ground'])),
                'n_steps': r['n_steps'],
                'elapsed_time': r['elapsed_time'],
                'r_x': ts[0], 'r_y': ts[1], 'r_z': ts[2],
                'v_x': ts[3], 'v_y': ts[4], 'v_z': ts[5],
                'm': ts[6],
                'terminal_tilt_deg': r['terminal_tilt_deg'],
                'terminal_tilt_raw_deg': r['terminal_tilt_raw_deg'],
                'fuel_used_kg': r['fuel_used_kg'],
                'expert_tf': r['expert_tf'],
                'expert_mf': r['expert_mf'],
                'engine_cutoff': int(bool(r['engine_cutoff'])),
                'cutoff_reason': r['cutoff_reason'],
                'coast_outcome': r['coast_outcome'],
                'n_powered_steps': r['n_powered_steps'],
                'n_coast_steps': r['n_coast_steps'],
                'powered_time': r['powered_time'],
                'coast_time_s': r['coast_time_s'],
                'cutoff_r_z': (cs[2] if cs is not None else ''),
                'cutoff_v_z': (cs[5] if cs is not None else ''),
            })


# =====================================================================
# Terminal-state histograms
# =====================================================================
# =====================================================================

def _extract_terminal_quantities(results, tilt_source='raw'):
    """Pull terminal quantities out of the per-IC result dicts, grouped by
    outcome class.
    """
    if tilt_source not in ('raw', 'enforced'):
        raise ValueError("tilt_source must be 'raw' or 'enforced'")

    tilt_key = ('terminal_tilt_raw_deg' if tilt_source == 'raw'
                else 'terminal_tilt_deg')

    by_class = {c: {'v_norm': [], 'v_z': [], 'r_z': [], 'm_f': [],
                    'tilt': [], 'r_h': [], 'landed': []}
                for c in PLOT_CLASSES}
    n_excluded = {}

    for r in results:
        cls = r['classification']
        if cls not in by_class:
            n_excluded[cls] = n_excluded.get(cls, 0) + 1
            continue

        ts = np.asarray(r['terminal_state'], dtype=float)
        if not np.all(np.isfinite(ts)):
            n_excluded['nan'] = n_excluded.get('nan', 0) + 1
            continue

        r_x, r_y, r_z, v_x, v_y, v_z, m = ts

        d = by_class[cls]
        d['v_norm'].append(float(np.sqrt(v_x**2 + v_y**2 + v_z**2)))
        d['v_z'].append(float(abs(v_z)))
        d['r_z'].append(float(r_z))
        d['m_f'].append(float(m) / 1000.0)                 # kg -> t
        d['tilt'].append(float(r.get(tilt_key, np.nan)))
        d['r_h'].append(float(np.sqrt(r_x**2 + r_y**2)))
        d['landed'].append(float(bool(r['reached_ground'])))

    out = {c: {k: np.asarray(v, dtype=float) for k, v in d.items()}
           for c, d in by_class.items()}
    return out, n_excluded


def _stacked_hist(ax, data_by_class, key, xlabel, title,
                  thresholds=(), n_bins=30, log_y=False,
                  clip_percentile=None):
    """Draw one stacked, class-coloured histogram."""
    series, colors, labels, pooled = [], [], [], []

    for c in PLOT_CLASSES:
        if c not in data_by_class:
            continue
        v = data_by_class[c][key]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        series.append(v)
        colors.append(CLASS_COLORS[c])
        labels.append(f"{CLASS_LABELS[c]} (n={v.size})")
        pooled.append(v)

    if not series:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                transform=ax.transAxes, color='0.5')
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=10)
        return

    pooled = np.concatenate(pooled)
    lo, hi = float(pooled.min()), float(pooled.max())

    clipped = False
    if clip_percentile is not None and pooled.size > 20:
        hi_p = float(np.percentile(pooled, clip_percentile))
        if lo < hi_p < hi:
            hi = hi_p
            clipped = True

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0

    # Keep threshold lines inside the visible range
    for tv, _, _ in thresholds:
        if lo <= tv:
            hi = max(hi, tv * 1.05)

    bins = np.linspace(lo, hi, n_bins + 1)
    if clipped:
        series = [np.clip(s, lo, hi - 1e-12) for s in series]

    ax.hist(series, bins=bins, stacked=True, color=colors,
            label=labels, edgecolor='white', linewidth=0.4)

    for tv, tlabel, tstyle in thresholds:
        if lo <= tv <= hi:
            ax.axvline(tv, color='k', linestyle=tstyle, linewidth=1.2,
                       alpha=0.8)
            ax.text(tv, ax.get_ylim()[1] * 0.97, f' {tlabel}',
                    rotation=90, va='top', ha='left', fontsize=7,
                    color='k', alpha=0.8)

    if log_y:
        ax.set_yscale('log')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('count')
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    if clipped:
        ax.text(0.99, 0.90,
                f'x clipped at p{clip_percentile}\n(overflow in last bin)',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=6.5, color='0.35')


def plot_terminal_histograms(results, out_dir, tilt_source='raw',
                             include_airborne=True, theta_max_deg=15.0,
                             n_bins=30, log_y=False, dpi=160,
                             filename='terminal_histograms'):
    """Build the 2x3 terminal-state histogram figure and write PNG + PDF."""
    if not _HAVE_MPL:
        print("  WARNING: matplotlib not available — skipping histograms")
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data, n_excluded = _extract_terminal_quantities(
        results, tilt_source=tilt_source)

    if not include_airborne:
        for cls in ('airborne', 'fuel_exhausted'):
            for k in data[cls]:
                data[cls][k] = np.array([], dtype=float)

    n_shown = sum(int(data[c]['v_z'].size) for c in PLOT_CLASSES)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))

    _stacked_hist(
        axes[0, 0], data, 'v_norm',
        r'$\|\mathbf{v}\|$  [m/s]', 'Terminal speed',
        thresholds=(), n_bins=n_bins, log_y=log_y, clip_percentile=99.0)

    _stacked_hist(
        axes[0, 1], data, 'v_z',
        r'$|v_z|$  [m/s]', 'Terminal vertical speed',
        thresholds=((THR_VZ_SOFT, f'soft {THR_VZ_SOFT:g}', '--'),
                    (THR_VZ_HARD, f'hard {THR_VZ_HARD:g}', ':')),
        n_bins=n_bins, log_y=log_y, clip_percentile=99.0)

    # --- Terminal altitude: NON-LANDED ICs only ---
    alt_data = {c: {'r_z': data[c]['r_z'][data[c]['landed'] < 0.5]}
                for c in PLOT_CLASSES}
    n_landed_omitted = int(sum((data[c]['landed'] >= 0.5).sum()
                               for c in PLOT_CLASSES))

    _stacked_hist(
        axes[0, 2], alt_data, 'r_z',
        r'$r_z$  [m]', 'Terminal altitude (non-landed ICs only)',
        thresholds=(), n_bins=n_bins, log_y=log_y, clip_percentile=99.0)
    axes[0, 2].text(
        0.99, 0.78,
        f'{n_landed_omitted} landed ICs omitted\n'
        r'($r_z \equiv 0$ by bisection clamp)',
        transform=axes[0, 2].transAxes, ha='right', va='top',
        fontsize=6.5, color='0.35')

    _stacked_hist(
        axes[1, 0], data, 'm_f',
        r'$m_f$  [t]', 'Terminal mass',
        thresholds=(), n_bins=n_bins, log_y=log_y)

    tilt_title = ('Terminal thrust tilt (raw NN output)'
                  if tilt_source == 'raw'
                  else 'Terminal thrust tilt (after enforcement)')
    _stacked_hist(
        axes[1, 1], data, 'tilt',
        'tilt from vertical  [deg]', tilt_title,
        thresholds=((theta_max_deg,
                     f'{theta_max_deg:g}$^\\circ$ cone', '--'),),
        n_bins=n_bins, log_y=log_y, clip_percentile=99.5)

    _stacked_hist(
        axes[1, 2], data, 'r_h',
        r'$\|\mathbf{r}_h\|$  [m]', 'Terminal horizontal position',
        thresholds=((THR_RH_SOFT, f'soft {THR_RH_SOFT:g}', '--'),),
        n_bins=n_bins, log_y=log_y, clip_percentile=99.0)

    # Shared legend built from the union of all panels
    seen, handles, labels = set(), [], []
    for ax in axes.ravel():
        h, l = ax.get_legend_handles_labels()
        for hi_, li_ in zip(h, l):
            base = li_.split(' (n=')[0]
            if base not in seen:
                seen.add(base)
                handles.append(hi_)
                labels.append(li_)
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=len(handles),
                   frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.005))

    n_cut = sum(1 for r in results if r.get('engine_cutoff'))

    subtitle = f'{n_shown} ICs shown'
    if not include_airborne:
        subtitle += '  |  airborne + fuel-exhausted excluded'
    if n_excluded:
        excl = ', '.join(f'{k}={v}' for k, v in sorted(n_excluded.items()))
        subtitle += f'  |  excluded: {excl}'
    if tilt_source == 'enforced':
        subtitle += '  |  tilt censored at cone boundary'
    if n_cut:
        subtitle += f'  |  {n_cut} ICs reached engine cutoff (MECO)'

    fig.suptitle('Terminal state distributions by landing outcome\n'
                 + subtitle, fontsize=12)
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))

    paths = []
    for ext in ('png', 'pdf'):
        p = out_dir / f'{filename}.{ext}'
        fig.savefig(p, dpi=dpi, bbox_inches='tight')
        paths.append(p)
    plt.close(fig)

    return paths


# =====================================================================
# Main
# =====================================================================

def run_eval(config: ReactiveEvalConfig):
    """Full evaluation pipeline."""

    run_dir = Path(config.run_dir)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Load model ---
    print(f"\nLoading model from {run_dir}...")
    model_cfg = ReactiveConfig.load(run_dir / 'model_config.json')
    model = ReactiveTransformer(model_cfg).to(device)

    best_path = run_dir / 'best_model.pt'
    final_path = run_dir / 'final_model.pt'
    if best_path.exists():
        checkpoint = torch.load(best_path, weights_only=True,
                                map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        val = checkpoint.get('val_loss',
                             checkpoint.get('best_val_loss', float('nan')))
        print(f"  Loaded best model ("
              f"epoch {checkpoint.get('epoch', 'N/A')}, "
              f"val_loss={val:.6f})")
    elif final_path.exists():
        checkpoint = torch.load(final_path, weights_only=True,
                                map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  WARNING: using final model (no best_model.pt found)")
    else:
        raise FileNotFoundError(f"No model found in {run_dir}")

    model.eval()

    # --- Load normalization ---
    norm_path = run_dir / 'norm_stats.npz'
    norm_stats = ReactiveNormStats.load(norm_path)
    print(f"  Norm stats loaded from {norm_path}")

    # --- Load test ICs ---
    data_dir = Path(config.data_dir)
    all_files = sorted(data_dir.glob('traj_*.npz'))
    print(f"\n  Found {len(all_files)} trajectory files in {data_dir}")

    _, _, test_files = split_by_trajectory(
        all_files, config.train_frac, config.val_frac, config.seed)
    print(f"  Test ICs: {len(test_files)}")

    # --- Build dynamics ---
    print("\n  Building CasADi dynamics...")
    f_dyn = build_dynamics_function()
    F_rk4 = build_rk4_integrator(f_dyn)
    print("  Dynamics ready.\n")

    # --- Run evaluation ---
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)
    print(f"  Window size: {model_cfg.window_size}")
    print(f"  dt: {config.dt} s")
    print(f"  Max steps: {config.max_steps}")
    print(f"  Thrust clipping: {config.clip_thrust}")
    print(f"  Tilt enforcement: {config.enforce_tilt}")
    if config.cutoff_on_ascent:
        print(f"  Engine cutoff: ON — v_z > {config.cutoff_vz_ms:.2f} m/s "
              f"triggers MECO, then up to {config.coast_time:.1f} s "
              f"({config.max_coast_steps} steps) of unpowered fall")
    else:
        print(f"  Engine cutoff: OFF (v1 behaviour)")
    print(f"  Coast on fuel exhaustion: {config.coast_on_fuel}")
    print()

    results, summary = evaluate_batch(
        model, norm_stats, test_files, model_cfg.window_size,
        F_rk4, config, device)

    # --- Report ---
    print_summary(summary)

    # --- Save outputs ---
    config.save(out_dir / 'eval_config.json')

    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    save_per_ic_csv(results, out_dir / 'per_ic_results.csv')

    plot_paths = []
    if config.make_plots:
        theta_max_deg = float(np.degrees(np.arccos(cos_tilt)))
        plot_paths = plot_terminal_histograms(
            results, out_dir,
            tilt_source=config.plot_tilt_source,
            include_airborne=config.plot_include_airborne,
            theta_max_deg=theta_max_deg,
            n_bins=config.plot_bins,
            log_y=config.plot_log_y,
        )

    print(f"\n  Output directory: {out_dir}")
    print(f"  Summary:          {out_dir / 'summary.json'}")
    print(f"  Per-IC CSV:       {out_dir / 'per_ic_results.csv'}")
    print(f"  Eval config:      {out_dir / 'eval_config.json'}")
    for p in plot_paths:
        print(f"  Histograms:       {p}")

    return results, summary


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Reactive Transformer closed-loop evaluation (v2)',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--run_dir', type=str, default='runs/reactive_w16',
                        help='Directory containing trained model')
    parser.add_argument('--data_dir', type=str, default='data/ForcesLargeBatch',
                        help='Directory containing traj_*.npz files')
    parser.add_argument('--out_dir', type=str,
                        default='results/reactive_eval_v2',
                        help='Output directory for results')

    parser.add_argument('--dt', type=float, default=0.8,
                        help='Integration timestep [s] (default: 0.8)')
    parser.add_argument('--t_max', type=float, default=80.0,
                        help='Max POWERED simulation time [s] (default: 80). '
                             'The ballistic coast is additional to this.')
    parser.add_argument('--seed', type=int, default=42)

    # --- Engine cutoff + coast ---
    parser.add_argument('--no_cutoff', action='store_true',
                        help='Disable the ascent engine cutoff and the '
                             'ballistic coast, restoring v1 behaviour')
    parser.add_argument('--cutoff_vz', type=float, default=0.0,
                        help='v_z above this [m/s] triggers engine cutoff '
                             '(default: 0.0, i.e. any ascent)')
    parser.add_argument('--coast_time', type=float, default=10.0,
                        help='Cap on the unpowered fall after cutoff [s] '
                             '(default: 10). If the ground is not reached '
                             'by then the IC stays airborne.')
    parser.add_argument('--coast_on_fuel', action='store_true',
                        help='Also coast after fuel exhaustion (physically '
                             'the same situation, but OFF by default so '
                             'that class stays comparable with v1)')

    parser.add_argument('--no_clip', action='store_true',
                        help='Disable thrust magnitude clipping')
    parser.add_argument('--no_tilt', action='store_true',
                        help='Disable tilt constraint enforcement')

    parser.add_argument('--no_plots', action='store_true',
                        help='Skip histogram generation')
    parser.add_argument('--plot_tilt_source', choices=['raw', 'enforced'],
                        default='raw',
                        help='Plot raw NN tilt or post-enforcement tilt')
    parser.add_argument('--plot_no_airborne', action='store_true',
                        help='Restrict histograms to ICs that reached ground')
    parser.add_argument('--plot_bins', type=int, default=30)
    parser.add_argument('--plot_log_y', action='store_true',
                        help='Log-scale the histogram counts')

    args = parser.parse_args()

    if args.coast_time <= 0 and not args.no_cutoff:
        parser.error('--coast_time must be > 0 (or use --no_cutoff)')

    config = ReactiveEvalConfig(
        data_dir=args.data_dir,
        run_dir=args.run_dir,
        out_dir=args.out_dir,
        dt=args.dt,
        t_max=args.t_max,
        seed=args.seed,
        cutoff_on_ascent=not args.no_cutoff,
        cutoff_vz_ms=args.cutoff_vz,
        coast_time=args.coast_time,
        coast_on_fuel=args.coast_on_fuel,
        clip_thrust=not args.no_clip,
        enforce_tilt=not args.no_tilt,
        make_plots=not args.no_plots,
        plot_tilt_source=args.plot_tilt_source,
        plot_include_airborne=not args.plot_no_airborne,
        plot_bins=args.plot_bins,
        plot_log_y=args.plot_log_y,
    )

    print("RETALT1 PDG — Reactive Transformer Evaluation (v2)")
    print(f"  Model:  {args.run_dir}")
    print(f"  Data:   {args.data_dir}")
    print(f"  Output: {args.out_dir}")
    print(f"  dt={config.dt}s, t_max={config.t_max}s")
    print(f"  Clip thrust: {config.clip_thrust}, "
          f"Enforce tilt: {config.enforce_tilt}")
    print(f"  Engine cutoff on ascent: {config.cutoff_on_ascent} "
          f"(v_z > {config.cutoff_vz_ms:.2f} m/s), "
          f"coast cap {config.coast_time:.1f} s")
    print()

    run_eval(config)


if __name__ == '__main__':
    main()
