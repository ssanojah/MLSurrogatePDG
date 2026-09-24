#!/usr/bin/env python3
"""
Single-IC DAgger diagnostic for the reactive Transformer.

Rolls the policy out on one IC, queries the IPOPT expert at every visited
state and plots policy against expert.

Usage:
    python diagnose_dagger_single.py --run_dir runs/<run> \\
        --data_dir data/<dataset> --ic_index 0
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import json
import time
import sys
from pathlib import Path

# =====================================================================
# Project imports
# =====================================================================

from reactive_model import ReactiveTransformer, ReactiveConfig
from expert import ExpertSolver
from ipopt_pdg import (
    build_dynamics_function, build_rk4_integrator,
    nx, nu, m_dry, m0,
    T_min_MN, T_max_MN,
    cos_tilt, theta_max, g0,
)
from dataset import (
    split_by_ic, NormalizationStats, load_trajectory,
)


# =====================================================================
# Thrust clipping and tilt enforcement
# =====================================================================

def clip_thrust(u_MN):
    """Direction-preserving thrust clipping to [T_min, T_max]."""
    T_mag = np.linalg.norm(u_MN)
    if T_mag < 1e-10:
        return np.array([0.0, 0.0, T_min_MN])
    direction = u_MN / T_mag
    T_clipped = np.clip(T_mag, T_min_MN, T_max_MN)
    return direction * T_clipped


def enforce_tilt(u_MN):
    """Enforce tilt constraint: T_z >= cos(theta_max) * ||T||."""
    T_mag = np.linalg.norm(u_MN)
    if T_mag < 1e-10:
        return u_MN
    if u_MN[2] >= cos_tilt * T_mag:
        return u_MN
    T_h_mag = np.sqrt(u_MN[0]**2 + u_MN[1]**2)
    sin_tilt = np.sin(theta_max)
    if sin_tilt < 1e-10:
        return np.array([0.0, 0.0, T_mag])
    T_z_new = cos_tilt * T_h_mag / sin_tilt
    u_new = u_MN.copy()
    u_new[2] = T_z_new
    return clip_thrust(u_new)


# =====================================================================
# Model loading
# =====================================================================

def load_reactive_model(run_dir):
    """Load ReactiveTransformer + norm_stats from a run directory."""
    run_dir = Path(run_dir)

    # --- Config ---
    config_path = run_dir / 'model_config.json'
    if config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)
    else:
        config_dict = {}

    # Build ReactiveConfig from whatever fields match
    valid_fields = {k for k in ReactiveConfig.__dataclass_fields__}
    filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
    config = ReactiveConfig(**filtered)

    # --- Norm stats ---
    for name in ['norm_stats.npz', 'best_norm_stats.npz']:
        p = run_dir / name
        if p.exists():
            norm_stats = NormalizationStats.load(str(p))
            break
    else:
        raise FileNotFoundError(f"No norm_stats.npz found in {run_dir}")

    # --- Checkpoint ---
    for name in ['best_model.pt', 'model.pt']:
        p = run_dir / name
        if p.exists():
            ckpt_path = p
            break
    else:
        raise FileNotFoundError(f"No model checkpoint found in {run_dir}")

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Handle both checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', '?')
        val_loss = checkpoint.get('val_loss', '?')
        print(f"  Checkpoint: epoch={epoch}, val_loss={val_loss}")
    else:
        state_dict = checkpoint

    # Build and load model
    model = ReactiveTransformer(config)
    model.load_state_dict(state_dict)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: ReactiveTransformer, {n_params:,} parameters")
    print(f"  Config: window={config.window_size}, d_model={config.d_model}, "
          f"heads={config.nhead}, layers={config.num_layers}")
    print(f"  Norm stats: state_dim={len(norm_stats.state_mean)}, "
          f"action_dim={len(norm_stats.action_mean)}")

    return model, config, norm_stats


# =====================================================================
# Reactive rollout
# =====================================================================

def rollout_reactive(model, config, norm_stats, x0_7, F_rk4,
                     dt=0.8, max_time=80.0, min_altitude=-5.0):
    """Roll out reactive sliding-window model step by step."""
    W = config.window_size
    state_dim = len(norm_stats.state_mean)  # 7 for reactive
    max_steps = int(max_time / dt)

    states = [x0_7.copy()]
    u_raw_list = []
    u_clipped_list = []
    history = [x0_7.astype(np.float32)]

    for k in range(max_steps):
        # Build (1, W, state_dim) window from recent history
        n_hist = len(history)
        w_start = max(0, n_hist - W)
        window_states = history[w_start:]
        pad_len = W - len(window_states)

        window = np.zeros((1, W, state_dim), dtype=np.float32)
        for i, s in enumerate(window_states):
            s_norm = (s[:state_dim] - norm_stats.state_mean) \
                     / norm_stats.state_std
            window[0, pad_len + i, :] = s_norm

        # Forward pass → (1, 3) single thrust
        with torch.no_grad():
            pred = model(torch.from_numpy(window))

        u_norm = pred[0].cpu().numpy()
        u_phys = u_norm * norm_stats.action_std + norm_stats.action_mean
        u_raw_list.append(u_phys.copy())

        u_clip = enforce_tilt(clip_thrust(u_phys))
        u_clipped_list.append(u_clip.copy())

        # Integrate dynamics
        x_next = np.array(F_rk4(states[-1], u_clip, dt)).flatten()

        # Safety checks
        if x_next[2] < min_altitude:
            states.append(x_next)
            break
        if x_next[6] < m_dry * 0.95:
            states.append(x_next)
            break
        if np.any(np.isnan(x_next)):
            states.append(x_next)
            break

        states.append(x_next)
        history.append(x_next.astype(np.float32))

    K = len(u_clipped_list)
    return {
        'x_traj': np.array(states),           # (K+1, 7)
        'u_raw': np.array(u_raw_list),         # (K, 3)
        'u_clipped': np.array(u_clipped_list), # (K, 3)
        'dt': dt,
        'n_steps': K,
        'times': np.arange(K) * dt,
        'times_state': np.arange(K + 1) * dt,
    }


# =====================================================================
# Expert queries at every policy-visited state
# =====================================================================

def query_expert_along_rollout(expert, rollout, t_f_original):
    """Query the expert at EVERY state in the policy rollout."""
    x_traj = rollout['x_traj']
    dt = rollout['dt']
    K = rollout['n_steps']

    u_expert = np.full((K, 3), np.nan)
    tf_opt = np.full(K, np.nan)
    feasible = np.zeros(K, dtype=bool)
    solve_time = np.zeros(K)

    tf_guess = t_f_original

    for k in range(K):
        x_k = x_traj[k]  # 7-dim physical state

        result = expert.query(x_k, tf_guess=tf_guess)

        solve_time[k] = result['solve_time']
        feasible[k] = result['feasible']

        if result['feasible']:
            u_expert[k] = result['u_opt']
            tf_opt[k] = result['tf_opt']
            tf_guess = max(result['tf_opt'] - dt, 3.0)
        else:
            tf_guess = max(tf_guess - dt, 3.0)

        # Progress every 10 steps
        if (k + 1) % 10 == 0 or k == K - 1:
            n_ok = int(np.sum(feasible[:k+1]))
            print(f"  Step {k+1:3d}/{K}: expert {n_ok}/{k+1} converged "
                  f"({100*n_ok/(k+1):.0f}%), "
                  f"solve={result['solve_time']:.2f}s, "
                  f"tf_guess={tf_guess:.1f}s"
                  + (f", tf*={result['tf_opt']:.1f}s" if result['feasible']
                     else ", FAILED"),
                  flush=True)

    return {
        'u_expert': u_expert,
        'tf_opt': tf_opt,
        'feasible': feasible,
        'solve_time': solve_time,
    }


# =====================================================================
# Plotting
# =====================================================================

def make_diagnostic_plots(rollout, expert_data, expert_traj, t_f_original,
                          ic_index, out_dir):
    """Generate the 6-panel DAgger diagnostic figure."""
    t_state = rollout['times_state']
    t_ctrl = rollout['times']
    x_traj = rollout['x_traj']
    u_clip = rollout['u_clipped']
    K = rollout['n_steps']

    u_exp = expert_data['u_expert']
    tf_opt = expert_data['tf_opt']
    feas = expert_data['feasible']
    st = expert_data['solve_time']

    # Original expert trajectory
    t_f_exp = expert_traj['t_f']
    x_exp = expert_traj['x_traj']
    u_exp_orig = expert_traj['u_traj']
    N_exp = len(u_exp_orig)
    t_exp_state = np.linspace(0, t_f_exp, len(x_exp))
    t_exp_ctrl = np.linspace(0, t_f_exp, N_exp)

    # Derived quantities
    T_mag_policy = np.linalg.norm(u_clip, axis=1) * 1e3     # kN
    T_mag_expert = np.linalg.norm(u_exp, axis=1) * 1e3       # kN, NaN if failed
    T_mag_exp_orig = np.linalg.norm(u_exp_orig, axis=1) * 1e3

    # DAgger correction magnitude
    correction = np.full(K, np.nan)
    for k in range(K):
        if feas[k]:
            correction[k] = np.linalg.norm(u_exp[k] - u_clip[k]) * 1e3  # kN

    r_z_final = x_traj[-1, 2]
    v_z_final = x_traj[-1, 5]
    n_feas = int(np.sum(feas))

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(
        f'DAgger Diagnostic — IC {ic_index}  |  '
        f'{K} steps, $r_z$={r_z_final:.1f} m, $v_z$={v_z_final:.1f} m/s  |  '
        f'Expert: {n_feas}/{K} converged ({100*n_feas/max(K,1):.0f}%)',
        fontsize=13, fontweight='bold',
    )

    t_feas = t_ctrl[feas]
    t_fail = t_ctrl[~feas]

    # ---- Panel 1: Altitude ----
    ax = axes[0, 0]
    ax.plot(t_state, x_traj[:, 2], 'b-', lw=2, label='Policy $r_z$')
    ax.plot(t_exp_state, x_exp[:, 2], 'g--', lw=1.5, alpha=0.7,
            label='Expert traj $r_z$')
    ax.axhline(0, color='r', ls=':', lw=1, alpha=0.5, label='Ground')
    ax.set_ylabel('Altitude [m]')
    ax.set_title('Altitude')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ---- Panel 2: Vertical velocity ----
    ax = axes[0, 1]
    ax.plot(t_state, x_traj[:, 5], 'b-', lw=2, label='Policy $v_z$')
    ax.plot(t_exp_state, x_exp[:, 5], 'g--', lw=1.5, alpha=0.7,
            label='Expert traj $v_z$')
    ax.axhline(0, color='k', ls=':', lw=0.5)
    ax.axhline(-2, color='orange', ls=':', lw=1, alpha=0.5,
               label='Soft: $|v_z|$≤2')
    ax.set_ylabel('Vertical velocity [m/s]')
    ax.set_title('Vertical Velocity (negative = descending)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ---- Panel 3: Thrust magnitude ----
    ax = axes[1, 0]
    ax.step(t_ctrl, T_mag_policy, 'b-', lw=1.5, where='post',
            label='Policy $\\|T\\|$')
    ax.plot(t_feas, T_mag_expert[feas], 'go', ms=4, alpha=0.7,
            label=f'Expert recommended ({n_feas} pts)')
    if len(t_fail) > 0:
        ax.plot(t_fail, np.full(len(t_fail), T_min_MN * 1e3),
                'rx', ms=6, label=f'Expert FAILED ({K - n_feas} pts)')
    ax.axhline(T_min_MN * 1e3, color='gray', ls='--', lw=1, alpha=0.5,
               label=f'$T_{{min}}$={T_min_MN*1e3:.0f} kN')
    ax.axhline(T_max_MN * 1e3, color='gray', ls='-.', lw=1, alpha=0.5,
               label=f'$T_{{max}}$={T_max_MN*1e3:.0f} kN')
    ax.step(t_exp_ctrl, T_mag_exp_orig, 'g--', lw=1, alpha=0.5,
            where='post', label='Expert traj $\\|T\\|$')
    ax.set_ylabel('Thrust magnitude [kN]')
    ax.set_title('Thrust Magnitude — Policy vs Expert Recommended')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(alpha=0.3)

    # ---- Panel 4: Vertical thrust T_z ----
    ax = axes[1, 1]
    ax.step(t_ctrl, u_clip[:, 2] * 1e3, 'b-', lw=1.5, where='post',
            label='Policy $T_z$')
    ax.plot(t_feas, u_exp[feas, 2] * 1e3, 'go', ms=4, alpha=0.7,
            label='Expert $T_z$')
    ax.step(t_exp_ctrl, u_exp_orig[:, 2] * 1e3, 'g--', lw=1, alpha=0.5,
            where='post', label='Expert traj $T_z$')
    weight_kN = x_traj[0, 6] * g0 / 1e3
    ax.axhline(weight_kN, color='orange', ls=':', lw=1, alpha=0.5,
               label=f'Weight ≈ {weight_kN:.0f} kN')
    ax.set_ylabel('Vertical thrust $T_z$ [kN]')
    ax.set_title('Vertical Thrust — Policy vs Expert Recommended')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ---- Panel 5: Expert solver diagnostics ----
    ax = axes[2, 0]
    if n_feas > 0:
        ax.plot(t_ctrl[feas], tf_opt[feas], 'g.-', lw=1.5, ms=4,
                label='Expert $t_f^*$ (remaining)')
    t_ideal = t_f_original - t_ctrl
    ax.plot(t_ctrl, t_ideal, 'k:', lw=1, alpha=0.5,
            label=f'Ideal: {t_f_original:.0f} − $t$')
    if len(t_fail) > 0:
        ax.plot(t_fail, np.zeros(len(t_fail)), 'rx', ms=6,
                label='Expert FAILED')
    ax.set_ylabel('Expert remaining time [s]')
    ax.set_xlabel('Time [s]')
    ax.set_title('Expert Solver: Remaining Time & Convergence')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Solve time on secondary axis
    if n_feas > 0:
        ax2 = ax.twinx()
        ax2.bar(t_ctrl[feas], st[feas], width=rollout['dt'] * 0.8,
                alpha=0.15, color='blue')
        ax2.set_ylabel('Solve time [s]', color='blue', alpha=0.5)
        ax2.tick_params(axis='y', labelcolor='blue')

    # ---- Panel 6: DAgger correction magnitude ----
    ax = axes[2, 1]
    colors = ['green' if f else 'red' for f in feas]
    bar_vals = np.where(np.isnan(correction), 0, correction)
    ax.bar(t_ctrl, bar_vals, width=rollout['dt'] * 0.8,
           color=colors, alpha=0.6)
    ax.set_ylabel('$\\|u_{expert} - u_{policy}\\|$ [kN]')
    ax.set_xlabel('Time [s]')
    ax.set_title('DAgger Correction Magnitude (green=converged, red=failed)')
    ax.grid(alpha=0.3)

    # Summary text box
    if n_feas > 0:
        corr_valid = correction[feas]
        max_corr = np.nanmax(corr_valid)
        max_idx = np.where(feas)[0][np.nanargmax(corr_valid)]
        mean_corr = np.nanmean(corr_valid)

        tf_valid = tf_opt[feas]
        n_increase = int(np.sum(np.diff(tf_valid) > 0.5))
        mono_str = "✓ monotonic" if n_increase == 0 \
            else f"✗ {n_increase} jumps"

        textstr = (
            f"Expert: {n_feas}/{K} converged ({100*n_feas/K:.0f}%)\n"
            f"Mean correction: {mean_corr:.1f} kN\n"
            f"Max correction: {max_corr:.1f} kN "
            f"(step {max_idx}, alt={x_traj[max_idx, 2]:.0f} m)\n"
            f"Mean solve time: {np.mean(st[feas]):.2f} s\n"
            f"$t_f^*$ trend: {mono_str}"
        )
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = Path(out_dir) / f'dagger_diag_ic{ic_index}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {out_path}")
    plt.close()
    return out_path


# =====================================================================
# Save detailed results
# =====================================================================

def save_results_json(rollout, expert_data, ic_index, out_dir):
    """Save per-step numerical data for later analysis."""
    K = rollout['n_steps']
    feas = expert_data['feasible']

    results = {
        'ic_index': ic_index,
        'n_steps': K,
        'dt': rollout['dt'],
        'r_z_final': float(rollout['x_traj'][-1, 2]),
        'v_z_final': float(rollout['x_traj'][-1, 5]),
        'expert_converged': int(np.sum(feas)),
        'expert_total': K,
        'expert_rate': float(np.mean(feas)),
    }

    if np.any(feas):
        correction = np.array([
            np.linalg.norm(expert_data['u_expert'][k] - rollout['u_clipped'][k])
            * 1e3 if feas[k] else np.nan for k in range(K)
        ])
        results['mean_correction_kN'] = float(np.nanmean(correction))
        results['max_correction_kN'] = float(np.nanmax(correction))
        max_k = int(np.nanargmax(correction))
        results['max_correction_step'] = max_k
        results['max_correction_alt_m'] = float(rollout['x_traj'][max_k, 2])
        results['mean_solve_time_s'] = float(
            np.mean(expert_data['solve_time'][feas]))

        tf_valid = expert_data['tf_opt'][feas]
        results['tf_opt_monotonic'] = bool(np.all(np.diff(tf_valid) <= 0.5))
        results['tf_opt_n_jumps'] = int(np.sum(np.diff(tf_valid) > 0.5))

    # Per-step data
    per_step = []
    for k in range(K):
        entry = {
            'step': k,
            'time': float(rollout['times'][k]),
            'r_z': float(rollout['x_traj'][k, 2]),
            'v_z': float(rollout['x_traj'][k, 5]),
            'T_policy_kN': float(np.linalg.norm(rollout['u_clipped'][k]) * 1e3),
            'feasible': bool(feas[k]),
            'solve_time': float(expert_data['solve_time'][k]),
        }
        if feas[k]:
            entry['T_expert_kN'] = float(
                np.linalg.norm(expert_data['u_expert'][k]) * 1e3)
            entry['tf_opt'] = float(expert_data['tf_opt'][k])
            entry['correction_kN'] = float(
                np.linalg.norm(
                    expert_data['u_expert'][k] - rollout['u_clipped'][k]
                ) * 1e3)
        per_step.append(entry)
    results['per_step'] = per_step

    out_path = Path(out_dir) / f'dagger_diag_ic{ic_index}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")
    return results


# =====================================================================
# Full expert trajectory visualization at selected steps
# =====================================================================

def solve_full_trajectories(expert, rollout, t_f_original, step_indices):
    """Solve the FULL OCP at selected policy-visited states and return complete
    expert trajectories (not just u_0).
    """
    from ipopt_pdg import N as N_IPOPT

    solver = expert._solver  # ParametricPDGSolver
    x_traj_policy = rollout['x_traj']
    dt = rollout['dt']

    results = {}
    tf_guess = t_f_original

    for k in step_indices:
        if k >= rollout['n_steps']:
            continue

        x_k = x_traj_policy[k]
        t_k = k * dt

        print(f"  Step {k:3d} (t={t_k:.1f}s, alt={x_k[2]:.0f}m, "
              f"v_z={x_k[5]:.1f}m/s): ", end='', flush=True)

        try:
            result = solver.solve(x_k, tf_guess=tf_guess)
            status = result['status']
            feasible = status in {'Solve_Succeeded',
                                  'Solved_To_Acceptable_Level'}

            if feasible:
                tf_opt = result['tf']
                tf_guess = max(tf_opt - dt, 3.0)
                # Build physical time arrays for this expert trajectory
                t_expert_state = t_k + np.linspace(0, tf_opt, N_IPOPT + 1)
                t_expert_ctrl = t_k + np.linspace(0, tf_opt, N_IPOPT)

                results[k] = {
                    'x_traj': result['x_traj'],    # (61, 7)
                    'u_traj': result['u_traj'],     # (60, 3)
                    'tf': tf_opt,
                    't_state': t_expert_state,
                    't_ctrl': t_expert_ctrl,
                    't_start': t_k,
                    'feasible': True,
                }
                print(f"OK  tf*={tf_opt:.1f}s, "
                      f"u0={np.linalg.norm(result['u_traj'][0])*1e3:.0f} kN, "
                      f"r_z_final={result['x_traj'][-1, 2]:.2f}m, "
                      f"v_z_final={result['x_traj'][-1, 5]:.2f}m/s")
            else:
                results[k] = {'feasible': False}
                tf_guess = max(tf_guess - dt, 3.0)
                print(f"FAILED ({status})")

        except Exception as e:
            results[k] = {'feasible': False}
            print(f"EXCEPTION: {e}")

    return results


def plot_expert_trajectories(rollout, full_trajs, expert_traj_bc,
                             ic_index, out_dir):
    """Plot full expert trajectories from selected policy-visited states."""
    from ipopt_pdg import N as N_IPOPT

    t_state = rollout['times_state']
    t_ctrl = rollout['times']
    x_policy = rollout['x_traj']
    u_policy = rollout['u_clipped']
    dt = rollout['dt']

    # Original expert trajectory for comparison
    t_f_bc = expert_traj_bc['t_f']
    x_bc = expert_traj_bc['x_traj']
    u_bc = expert_traj_bc['u_traj']
    N_bc = len(u_bc)
    t_bc_state = np.linspace(0, t_f_bc, len(x_bc))
    t_bc_ctrl = np.linspace(0, t_f_bc, N_bc)

    # Feasible trajectories only
    feasible_keys = sorted(k for k, v in full_trajs.items() if v['feasible'])
    if not feasible_keys:
        print("  No feasible expert trajectories to plot.")
        return

    # Color map: blue for early steps, red for late steps
    import matplotlib.cm as cm
    n_trajs = len(feasible_keys)
    colors = cm.coolwarm(np.linspace(0, 1, n_trajs))

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        f'Full Expert Trajectories — IC {ic_index}  |  '
        f'{n_trajs} expert solves from policy-visited states',
        fontsize=13, fontweight='bold',
    )

    # ---- Panel 1: Altitude ----
    ax = axes[0]
    ax.plot(t_state, x_policy[:, 2], 'k-', lw=2.5, alpha=0.4,
            label='Policy rollout', zorder=1)
    ax.plot(t_bc_state, x_bc[:, 2], 'k--', lw=1, alpha=0.3,
            label='BC expert traj')

    for i, k in enumerate(feasible_keys):
        tr = full_trajs[k]
        ax.plot(tr['t_state'], tr['x_traj'][:, 2],
                '-', color=colors[i], lw=1.5, alpha=0.8,
                label=f'Step {k} (t={k*dt:.0f}s)')
        # Mark the starting point
        ax.plot(tr['t_start'], tr['x_traj'][0, 2],
                'o', color=colors[i], ms=6, zorder=5)

    ax.axhline(0, color='r', ls=':', lw=1, alpha=0.5)
    ax.set_ylabel('Altitude $r_z$ [m]')
    ax.set_title('Altitude — each line is the expert\'s full planned '
                 'trajectory from that state')
    ax.legend(fontsize=7, ncol=3, loc='upper right')
    ax.grid(alpha=0.3)

    # ---- Panel 2: Vertical velocity ----
    ax = axes[1]
    ax.plot(t_state, x_policy[:, 5], 'k-', lw=2.5, alpha=0.4,
            label='Policy rollout')
    ax.plot(t_bc_state, x_bc[:, 5], 'k--', lw=1, alpha=0.3,
            label='BC expert traj')

    for i, k in enumerate(feasible_keys):
        tr = full_trajs[k]
        ax.plot(tr['t_state'], tr['x_traj'][:, 5],
                '-', color=colors[i], lw=1.5, alpha=0.8)
        ax.plot(tr['t_start'], tr['x_traj'][0, 5],
                'o', color=colors[i], ms=6, zorder=5)

    ax.axhline(0, color='k', ls=':', lw=0.5)
    ax.axhline(-2, color='orange', ls=':', lw=1, alpha=0.5,
               label='Soft: $|v_z|$≤2')
    ax.set_ylabel('Vertical velocity $v_z$ [m/s]')
    ax.set_title('Vertical Velocity — expert plans deceleration to v=0')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ---- Panel 3: Thrust magnitude ----
    ax = axes[2]
    ax.step(t_ctrl, np.linalg.norm(u_policy, axis=1) * 1e3,
            'k-', lw=2.5, alpha=0.4, where='post', label='Policy rollout')
    ax.step(t_bc_ctrl, np.linalg.norm(u_bc, axis=1) * 1e3,
            'k--', lw=1, alpha=0.3, where='post', label='BC expert traj')

    for i, k in enumerate(feasible_keys):
        tr = full_trajs[k]
        T_mag = np.linalg.norm(tr['u_traj'], axis=1) * 1e3  # kN
        ax.step(tr['t_ctrl'], T_mag,
                '-', color=colors[i], lw=1.5, alpha=0.8, where='post',
                label=f'Step {k} (tf*={tr["tf"]:.1f}s)')
        ax.plot(tr['t_start'], T_mag[0],
                'o', color=colors[i], ms=6, zorder=5)

    ax.axhline(T_min_MN * 1e3, color='gray', ls='--', lw=1, alpha=0.5,
               label=f'$T_{{min}}$={T_min_MN*1e3:.0f} kN')
    ax.axhline(T_max_MN * 1e3, color='gray', ls='-.', lw=1, alpha=0.5,
               label=f'$T_{{max}}$={T_max_MN*1e3:.0f} kN')
    ax.set_ylabel('Thrust magnitude [kN]')
    ax.set_xlabel('Time [s]')
    ax.set_title('Thrust — expert\'s full planned thrust profile '
                 'from each state')
    ax.legend(fontsize=6, ncol=3, loc='upper left')
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = Path(out_dir) / f'expert_trajs_ic{ic_index}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nFull trajectory plot saved to {out_path}")
    plt.close()


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Single-IC DAgger diagnostic for the reactive '
                    'sliding-window Transformer'
    )
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Path to trained reactive model run directory')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to BC trajectory data (for IC + expert traj)')
    parser.add_argument('--ic_index', type=int, default=0,
                        help='Index into the TEST set (default: 0)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory (default: results/dagger_diag/)')
    parser.add_argument('--dt', type=float, default=0.8,
                        help='Integration timestep [s] (default: 0.8)')
    parser.add_argument('--max_time', type=float, default=80.0,
                        help='Max rollout time [s] (default: 80)')
    parser.add_argument('--expert_verbose', action='store_true',
                        help='Print IPOPT output for each expert query')
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = 'results/dagger_diag'
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load reactive model
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 1: Loading reactive model")
    print("=" * 70)
    model, config, norm_stats = load_reactive_model(args.run_dir)

    # ------------------------------------------------------------------
    # 2. Select IC from test set
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: Selecting IC from test set")
    print("=" * 70)
    data_dir = Path(args.data_dir)
    all_files = sorted(data_dir.glob('traj_*.npz'))
    print(f"  Found {len(all_files)} trajectory files in {data_dir}")

    train_files, val_files, test_files = split_by_ic(
        all_files, train_frac=0.70, val_frac=0.15, seed=42)
    print(f"  Split: {len(train_files)} train, {len(val_files)} val, "
          f"{len(test_files)} test")

    if args.ic_index >= len(test_files):
        print(f"  ERROR: ic_index={args.ic_index} >= {len(test_files)} "
              f"test ICs")
        sys.exit(1)

    ic_file = test_files[args.ic_index]
    expert_traj = load_trajectory(ic_file)
    x0_7 = expert_traj['x_traj'][0, :7].copy()
    t_f_original = expert_traj['t_f']

    print(f"  IC {args.ic_index}: {Path(ic_file).name}")
    print(f"    r = [{x0_7[0]:.0f}, {x0_7[1]:.0f}, {x0_7[2]:.0f}] m")
    print(f"    v = [{x0_7[3]:.0f}, {x0_7[4]:.0f}, {x0_7[5]:.0f}] m/s")
    print(f"    m = {x0_7[6]:.0f} kg,  t_f = {t_f_original:.1f} s")

    # ------------------------------------------------------------------
    # 3. Build dynamics integrator
    # ------------------------------------------------------------------
    print("\n  Building CasADi RK4 integrator...")
    f_dyn = build_dynamics_function()
    F_rk4 = build_rk4_integrator(f_dyn)

    # ------------------------------------------------------------------
    # 4. Roll out policy
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: Rolling out reactive policy")
    print("=" * 70)

    t0 = time.time()
    rollout = rollout_reactive(
        model, config, norm_stats, x0_7, F_rk4,
        dt=args.dt, max_time=args.max_time,
    )
    t_roll = time.time() - t0
    K = rollout['n_steps']

    print(f"  Steps: {K}")
    print(f"  Duration: {K * rollout['dt']:.1f} s")
    print(f"  Final altitude: {rollout['x_traj'][-1, 2]:.1f} m")
    print(f"  Final v_z: {rollout['x_traj'][-1, 5]:.1f} m/s")
    print(f"  Rollout time: {t_roll:.2f} s")

    # ------------------------------------------------------------------
    # 5. Query expert at EVERY step
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"STEP 4: Querying expert at every step ({K} queries)")
    print("=" * 70)
    print(f"  Backend: IPOPT (parametric)")
    print(f"  Estimated time: ~{K * 5:.0f} s ({K * 5 / 60:.1f} min)")
    print()

    expert = ExpertSolver(
        verbose=args.expert_verbose, parametric=True,
        tf_min=3.0)

    t0 = time.time()
    expert_data = query_expert_along_rollout(
        expert, rollout, t_f_original)
    t_expert = time.time() - t0

    n_feas = int(np.sum(expert_data['feasible']))
    print(f"\n  Expert queries complete:")
    print(f"    Converged: {n_feas}/{K} ({100*n_feas/max(K,1):.0f}%)")
    print(f"    Total time: {t_expert:.1f} s ({t_expert/60:.1f} min)")
    if n_feas > 0:
        print(f"    Mean solve time: "
              f"{np.mean(expert_data['solve_time'][expert_data['feasible']]):.2f} s")

    # ------------------------------------------------------------------
    # 6. Generate plots
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: Generating diagnostic plots")
    print("=" * 70)

    make_diagnostic_plots(
        rollout, expert_data, expert_traj,
        t_f_original, args.ic_index, args.out_dir,
    )

    # ------------------------------------------------------------------
    # 7. Full expert trajectory solves at selected steps
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 6: Solving FULL expert trajectories at selected steps")
    print("=" * 70)

    K = rollout['n_steps']
    # Estimate where landing should occur from original expert t_f
    landing_step = int(t_f_original / rollout['dt'])
    landing_step = min(landing_step, K - 1)

    # 2 early, 2 middle, 10 near the landing region
    early = [5, 15]
    mid = [landing_step // 3, 2 * landing_step // 3]
    # 10 steps spanning the last ~15s before expected landing
    end_start = max(0, landing_step - 18)
    end_steps = list(range(end_start, landing_step + 1, 2))
    if landing_step not in end_steps:
        end_steps.append(landing_step)
    end_steps = end_steps[:10]

    selected = sorted(set(early + mid + end_steps))
    selected = [s for s in selected if s < K]

    print(f"  Selected steps: {selected}")
    print(f"  ({len(selected)} full OCP solves, ~{len(selected)*5:.0f}s)")
    print()

    full_trajs = solve_full_trajectories(
        expert, rollout, t_f_original, selected)

    n_ok = sum(1 for v in full_trajs.values() if v['feasible'])
    print(f"\n  {n_ok}/{len(selected)} full trajectories converged")

    plot_expert_trajectories(
        rollout, full_trajs, expert_traj,
        args.ic_index, args.out_dir)

    # ------------------------------------------------------------------
    # 8. Save JSON results
    # ------------------------------------------------------------------
    results = save_results_json(
        rollout, expert_data, args.ic_index, args.out_dir)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  IC index: {args.ic_index}")
    print(f"  Rollout: {K} steps, "
          f"r_z={rollout['x_traj'][-1, 2]:.1f} m, "
          f"v_z={rollout['x_traj'][-1, 5]:.1f} m/s")
    print(f"  Expert: {n_feas}/{K} converged ({100*n_feas/max(K,1):.0f}%)")

    if n_feas > 0:
        print(f"  Mean correction: {results['mean_correction_kN']:.1f} kN")
        print(f"  Max correction: {results['max_correction_kN']:.1f} kN "
              f"(step {results['max_correction_step']}, "
              f"alt={results['max_correction_alt_m']:.0f} m)")
        mono = results.get('tf_opt_monotonic', None)
        if mono is not None:
            print(f"  tf_opt monotonic: "
                  f"{'yes' if mono else 'NO (' + str(results['tf_opt_n_jumps']) + ' jumps)'}")

    print(f"\n  Output: {args.out_dir}/")
    print(f"  Total wall time: {t_roll + t_expert:.0f} s "
          f"({(t_roll + t_expert)/60:.1f} min)")


if __name__ == '__main__':
    main()
