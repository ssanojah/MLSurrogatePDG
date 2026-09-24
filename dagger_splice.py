#!/usr/bin/env python3
"""
DAgger splice builder for the TTG Transformer.

Rolls out the policy, queries the fixed-dt expert bank at the visited
states and saves prefix-suffix training sequences (states, actions, mask)
for ttg_train_v2.py --dagger_file.

Usage:
    python dagger_splice.py --run_dir runs/<run> --n_ics 3 --brute \\
        --diagnose --out_dir results/splice_diag

    python dagger_splice.py --run_dir runs/<run> --n_ics 800 \\
        --ic_source train --n_late 3 --n_early 3 --out_dir data/dagger_iter001
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import forcespro
import forcespro.nlp

from forces_pdg import m_dry, m_overhead, T_max, NVAR
from forces_pdg_fixeddt import solve_fixed_dt

from standalone_eval_v2 import (
    rollout_closed_loop, NormStats, PDGTransformer, TConfig,
    LandingThresholds, load_traj, split_by_ic,
    CATEGORIES, CAT_COLORS, CAT_LABELS,
    N as N_NODES, _F_rk4,
)

from test_t3_real import load_policy, pack_z0


T_MAX_MN = T_max / 1e6

TOL_SEAM_POS_M = 1e-6
TOL_SEAM_VEL_MS = 1e-6
TOL_COH_POS_M = 1e-3
TOL_COH_VEL_MS = 1e-4
TOL_COH_MASS_KG = 1e-2
TOL_TTG_S = 1e-9
TOL_TERM_POS_M = 1e-2
TOL_TERM_VEL_MS = 1e-3


# =====================================================================
# Warm-start chain
# =====================================================================

def shift_plan(res, tf_pin_next):
    """Warm start for k+1 from the plan solved at k."""
    u_next = res['u_traj'][1:]
    if u_next.shape[0] == 0:
        return None
    x_next = res['x_traj'][1:, :7]
    return pack_z0(x_next, u_next, tf_pin_next, x_next.shape[0])


# =====================================================================
# Splice assembly
# =====================================================================

def build_splice(x_pol, res, k, dt):
    """Assemble one (60, 8) / (60, 3) / (60,) training sequence."""
    states = np.zeros((N_NODES, 8), dtype=np.float64)
    actions = np.full((N_NODES, 3), np.nan, dtype=np.float64)
    mask = np.zeros(N_NODES, dtype=bool)

    # Prefix: policy-visited states, labels masked.
    states[:k, :7] = x_pol[:k]

    # Suffix: expert plan, nodes k … 59.
    x_exp = res['x_traj'][:, :7]
    states[k:, :7] = x_exp[:N_NODES - k]
    actions[k:] = res['u_traj']
    mask[k:] = True

    states[:, 7] = (N_NODES - np.arange(N_NODES)) * dt

    return states, actions, mask


def assert_splice(states, actions, mask, x_pol, res, k, dt, strict=True):
    """Run A1–A5 on one assembled splice."""
    x_exp = res['x_traj'][:, :7]
    out = {}

    # --- A1 seam: the expert plan must start FROM the policy state ---
    d_seam = np.abs(x_exp[0] - x_pol[k])
    out['seam_pos_m'] = float(d_seam[:3].max())
    out['seam_vel_ms'] = float(d_seam[3:6].max())
    out['seam_mass_kg'] = float(d_seam[6])

    # --- A2 TTG identity ---
    tf_pin = (N_NODES - k) * dt
    out['ttg_err_s'] = float(abs(states[k, 7] - tf_pin))
    out['ttg_monotone'] = bool(np.all(np.diff(states[:, 7]) < 0))

    # --- A3 coherence: the labelled actions must generate the stored states ---
    x = states[k, :7].copy()
    e_pos = e_vel = e_mass = 0.0
    for j in range(k, N_NODES):
        x = np.array(_F_rk4(x, actions[j], dt)).flatten()
        ref = (states[j + 1, :7] if j + 1 < N_NODES else x_exp[-1])
        d = np.abs(x - ref)
        e_pos = max(e_pos, d[:3].max())
        e_vel = max(e_vel, d[3:6].max())
        e_mass = max(e_mass, d[6])
    out['coh_pos_m'] = float(e_pos)
    out['coh_vel_ms'] = float(e_vel)
    out['coh_mass_kg'] = float(e_mass)

    # --- A4 terminal: the expert plan actually lands ---
    out['term_pos_m'] = float(np.abs(x_exp[-1, :3]).max())
    out['term_vel_ms'] = float(np.abs(x_exp[-1, 3:6]).max())

    # --- A5 finiteness and mask/NaN agreement ---
    out['states_finite'] = bool(np.all(np.isfinite(states)))
    out['mask_matches_nan'] = bool(
        np.all(np.isfinite(actions[mask])) and
        np.all(np.isnan(actions[~mask])))
    out['n_labels'] = int(mask.sum())

    if strict:
        assert out['seam_pos_m'] < TOL_SEAM_POS_M, \
            f"A1 seam position {out['seam_pos_m']:.3e} m at k={k}"
        assert out['seam_vel_ms'] < TOL_SEAM_VEL_MS, \
            f"A1 seam velocity {out['seam_vel_ms']:.3e} m/s at k={k}"
        assert out['ttg_err_s'] < TOL_TTG_S, \
            f"A2 TTG identity violated by {out['ttg_err_s']:.3e} s at k={k}"
        assert out['ttg_monotone'], f"A2 TTG not strictly decreasing at k={k}"
        assert out['coh_pos_m'] < TOL_COH_POS_M, \
            f"A3 coherence position {out['coh_pos_m']:.3e} m at k={k}"
        assert out['coh_vel_ms'] < TOL_COH_VEL_MS, \
            f"A3 coherence velocity {out['coh_vel_ms']:.3e} m/s at k={k}"
        assert out['coh_mass_kg'] < TOL_COH_MASS_KG, \
            f"A3 coherence mass {out['coh_mass_kg']:.3e} kg at k={k}"
        assert out['term_pos_m'] < TOL_TERM_POS_M, \
            f"A4 terminal position {out['term_pos_m']:.3e} m at k={k}"
        assert out['term_vel_ms'] < TOL_TERM_VEL_MS, \
            f"A4 terminal velocity {out['term_vel_ms']:.3e} m/s at k={k}"
        assert out['states_finite'], f"A5 non-finite state at k={k}"
        assert out['mask_matches_nan'], f"A5 mask/NaN mismatch at k={k}"

    return out


# =====================================================================
# Splice-point sampling
# =====================================================================

def select_splices(converged_ks, n_late, n_early, rng, brute=False):
    """Choose splice points from the converged queries."""
    ks = sorted(k for k in converged_ks if k >= 1)
    if not ks:
        return []
    if brute:
        return ks
    late = ks[-n_late:] if n_late > 0 else []
    pool = [k for k in ks if k not in late]
    if pool and n_early > 0:
        take = min(n_early, len(pool))
        early = rng.choice(np.asarray(pool), size=take,
                           replace=False).tolist()
    else:
        early = []
    return sorted(set(int(k) for k in early + late))


# =====================================================================
# Per-IC pass
# =====================================================================

def process_ic(model, norms, bank, expert, thres, device,
               n_late, n_early, rng, brute, warm=True, strict=False):
    """Roll the policy out on one IC, query the expert at every visited state,
    and assemble splices at the selected converged steps.
    """
    ic_7 = expert['ic'][:7]
    tf = expert['t_f']
    dt = tf / N_NODES

    roll = rollout_closed_loop(
        model, norms, ic_7, tf, thres,
        is_mlp=False, use_ttg=True, clip_tilt=True, device=device)

    x_pol = roll['x_traj']       # (K+1, 7)
    u_app = roll['u_traj']       # (K, 3) enforced, actually applied
    u_raw = roll['u_traj_raw']   # (K, 3) raw network output
    K = roll['steps_done']

    plans, records = {}, []
    prev = None

    for k in range(K):
        n_act = N_NODES - k
        if n_act < 1 or n_act not in bank:
            continue
        tf_pin = n_act * dt
        z0 = shift_plan(prev, tf_pin) if (warm and prev is not None) else None

        try:
            res = solve_fixed_dt(bank[n_act], x_pol[k], tf_pin,
                                 N_stages=n_act + 1, z0=z0)
            err = None
        except Exception as e:
            res, err = None, f'{type(e).__name__}: {e}'

        rec = {'k': k, 'n_act': n_act}
        if res is None or res['exitflag'] != 1:
            rec.update({
                'converged': False,
                'exitflag': -999 if res is None else int(res['exitflag']),
                'error': err,
            })
            prev = None
        else:
            u_star = res['u_traj'][0]
            rec.update({
                'converged': True, 'exitflag': 1,
                'it': int(res['it']),
                'solvetime_ms': res['solvetime'] * 1e3,
                'res_eq': res['res_eq'], 'res_ineq': res['res_ineq'],
                'm_f_kg': float(res['m_f']),
                'margin_kg': float(res['m_f'] - m_dry),
                # Against what was APPLIED (post-enforcement).
                'correction_kN': float(
                    np.linalg.norm(u_star - u_app[k]) * 1e3),
                # Against the network's RAW output, which is what the
                # training loss sees.
                'correction_raw_kN': float(
                    np.linalg.norm(u_star - u_raw[k]) * 1e3),
                'correction_raw_vec_kN': (
                    (u_star - u_raw[k]) * 1e3).tolist(),
                'correction_raw_sigma': float(np.linalg.norm(
                    (u_star - u_raw[k]) / norms.a_sig)),
            })
            plans[k] = res
            prev = res
        records.append(rec)

    fails = [r['k'] for r in records if not r['converged']]
    k_fail = fails[0] if fails else None

    chosen = select_splices(plans.keys(), n_late, n_early, rng, brute=brute)

    splices = []
    for k in chosen:
        res = plans[k]
        states, actions, mask = build_splice(x_pol, res, k, dt)
        checks = assert_splice(states, actions, mask, x_pol, res, k, dt,
                               strict=strict)
        rec = next(r for r in records if r['k'] == k)
        splices.append({
            'states': states.astype(np.float32),
            'actions': actions.astype(np.float32),
            'mask': mask,
            'meta': {
                'ic': expert['name'], 'k': k, 'n_act': N_NODES - k, 'dt': dt,
                't_f': tf, 'k_fail': k_fail,
                'post_kfail': bool(k_fail is not None and k > k_fail),
                'exitflag': rec['exitflag'], 'it': rec['it'],
                'solvetime_ms': rec['solvetime_ms'],
                'res_eq': rec['res_eq'], 'res_ineq': rec['res_ineq'],
                'm_f_kg': rec['m_f_kg'], 'margin_kg': rec['margin_kg'],
                'correction_kN': rec['correction_kN'],
                'correction_raw_kN': rec['correction_raw_kN'],
                'correction_raw_vec_kN': rec['correction_raw_vec_kN'],
                'correction_raw_sigma': rec['correction_raw_sigma'],
                'n_labels': int(mask.sum()),
                'checks': checks,
            },
        })

    summary = {
        'name': expert['name'], 't_f': tf, 'dt': dt,
        'm_f_stored_kg': expert['m_f'],
        'category': roll['category'],
        'reached_ground': bool(roll['reached_ground']),
        'steps_done': K, 'term_reason': roll['term_reason'],
        'n_queries': len(records),
        'n_converged': sum(1 for r in records if r['converged']),
        'k_fail': k_fail,
        'n_splices': len(splices),
        'n_labels': int(sum(s['meta']['n_labels'] for s in splices)),
    }
    return summary, splices, records, roll, plans


# =====================================================================
# Diagnostic figure
# =====================================================================

def diagnose_ic(summary, splices, records, roll, plans, out_dir, tag):
    """Six panels, x-axis = NODE INDEX not time."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dt = summary['dt']
    x_pol = roll['x_traj']
    K = summary['steps_done']
    k_fail = summary['k_fail']
    ks = sorted(plans.keys())
    chosen = [s['meta']['k'] for s in splices]
    cmap = plt.cm.viridis
    cnorm = plt.Normalize(vmin=0, vmax=max(N_NODES - 1, 1))

    fig, ax = plt.subplots(2, 3, figsize=(19, 10))
    fig.suptitle(
        f"Splice diagnostic — {summary['name']} — "
        f"{CAT_LABELS.get(summary['category'], summary['category'])} — "
        f"{summary['n_splices']} splices, {summary['n_labels']} labels",
        fontweight='bold')

    def mark_kfail(a):
        if k_fail is not None:
            a.axvline(k_fail, color='k', ls=':', lw=1.2,
                      label=r'$k_{fail}$')

    # (0,0) altitude: policy spine + fan of spliced expert recovery plans
    a = ax[0, 0]
    for k in chosen:
        xe = plans[k]['x_traj'][:, :7]
        a.plot(np.arange(k, N_NODES + 1), xe[:, 2],
               color=cmap(cnorm(k)), lw=0.9, alpha=0.75)
    a.plot(np.arange(K + 1), x_pol[:, 2], 'k-', lw=2.2, label='policy')
    a.axhline(0, color='k', lw=0.5)
    mark_kfail(a)
    a.set_xlabel('node index'); a.set_ylabel(r'$r_z$ [m]')
    a.set_title('Altitude — policy spine, spliced expert plans')
    a.legend(fontsize=8)

    # (0,1) vertical velocity
    a = ax[0, 1]
    for k in chosen:
        xe = plans[k]['x_traj'][:, :7]
        a.plot(np.arange(k, N_NODES + 1), xe[:, 5],
               color=cmap(cnorm(k)), lw=0.9, alpha=0.75)
    a.plot(np.arange(K + 1), x_pol[:, 5], 'k-', lw=2.2, label='policy')
    a.axhline(0, color='k', lw=0.5)
    mark_kfail(a)
    a.set_xlabel('node index'); a.set_ylabel(r'$v_z$ [m/s]')
    a.set_title('Vertical velocity')
    a.legend(fontsize=8)

    # (0,2) thrust: raw vs applied vs expert plans, with the rails
    a = ax[0, 2]
    for k in chosen:
        ue = plans[k]['u_traj']
        a.plot(np.arange(k, N_NODES), np.linalg.norm(ue, axis=1) * 1e3,
               color=cmap(cnorm(k)), lw=0.8, alpha=0.6)
    a.plot(np.arange(K), roll['T_mag_kN'], 'k-', lw=2.0, label='applied')
    a.plot(np.arange(K), roll['T_mag_raw_kN'], color='tab:red', lw=1.0,
           alpha=0.7, label='raw NN output')
    a.axhline(472, color='gray', ls='--', lw=1, label=r'$T_{min}$')
    a.axhline(1179, color='gray', ls='-.', lw=1, label=r'$T_{max}$')
    mark_kfail(a)
    a.set_xlabel('node index'); a.set_ylabel('thrust [kN]')
    a.set_title('Thrust magnitude')
    a.legend(fontsize=7, ncol=2)

    a = ax[1, 0]
    conv = [r for r in records if r['converged']]
    a.plot([r['k'] for r in conv], [r['correction_kN'] for r in conv],
           'o-', ms=3, lw=1, color='tab:red',
           label=r'$\|u^*-u_{applied}\|$')
    a.plot([r['k'] for r in conv], [r['correction_raw_kN'] for r in conv],
           's--', ms=3, lw=1, color='tab:blue',
           label=r'$\|u^*-u_{raw}\|$  (what the loss sees)')
    mark_kfail(a)
    a.set_xlabel('rollout step k'); a.set_ylabel('correction [kN]')
    a.set_title('DAgger correction magnitude')
    a.legend(fontsize=8)

    # (1,1) expert health vs k
    a = ax[1, 1]
    a.plot([r['k'] for r in conv], [r['it'] for r in conv],
           'o-', ms=3, lw=1, color='tab:green', label='iterations')
    a.set_xlabel('rollout step k'); a.set_ylabel('iterations',
                                                 color='tab:green')
    a2 = a.twinx()
    a2.semilogy([r['k'] for r in conv],
                [max(r['res_ineq'], 1e-16) for r in conv],
                's--', ms=3, lw=1, color='tab:purple')
    a2.set_ylabel(r'res$_{ineq}$', color='tab:purple')
    bad = [r['k'] for r in records if not r['converged']]
    for kb in bad:
        a.axvline(kb, color='tab:red', alpha=0.25, lw=1)
    mark_kfail(a)
    a.set_title('Expert health (red lines: non-convergence)')

    # (1,2) THE TRAINING TENSOR, RENDERED.
    # Catches plumbing bugs no scalar metric will.
    a = ax[1, 2]
    if splices:
        s = splices[len(splices) // 2]
        S = s['states'].astype(float).T          # (8, 60)
        mu, sd = S.mean(axis=1, keepdims=True), S.std(axis=1,
                                                      keepdims=True) + 1e-9
        im = a.imshow((S - mu) / sd, aspect='auto', cmap='RdBu_r',
                      vmin=-2.5, vmax=2.5, interpolation='nearest')
        a.axvline(s['meta']['k'] - 0.5, color='k', lw=2)
        a.set_yticks(range(8))
        a.set_yticklabels(['r_x', 'r_y', 'r_z', 'v_x', 'v_y', 'v_z',
                           'm', 'TTG'], fontsize=8)
        a.set_xlabel('node index')
        a.set_title(f"Training tensor at k={s['meta']['k']} "
                    f"(z-scored; black line = splice seam,\n"
                    f"left of it = masked prefix)")
        fig.colorbar(im, ax=a, fraction=0.04)
    else:
        a.text(0.5, 0.5, 'no splices', ha='center', va='center',
               transform=a.transAxes)

    for axis in ax.flat:
        axis.grid(alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = Path(out_dir) / f'splice_diag_{tag}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# =====================================================================
# Main
# =====================================================================

def run(args):
    print("=" * 78)
    print("DAgger SPLICE BUILDER" + ("  [DIAGNOSTIC MODE]" if args.diagnose
                                     else "  [CAMPAIGN MODE]"))
    print("=" * 78)

    device = torch.device('cpu')
    thres = LandingThresholds()
    rng = np.random.default_rng(args.sample_seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1] Policy")
    model, norms, cfg = load_policy(args.run_dir, args.nhead, device)

    print(f"\n[2] Solver bank")
    bank = {}
    for entry in sorted(Path(args.bank_dir).iterdir()):
        if entry.is_dir() and entry.name.startswith('pdg_N'):
            try:
                n_int = int(entry.name.replace('pdg_N', ''))
            except ValueError:
                continue
            bank[n_int] = forcespro.nlp.Solver.from_directory(str(entry))
    print(f"    {len(bank)} solvers, n_int in [{min(bank)}, {max(bank)}]")

    print(f"\n[3] ICs")
    all_files = sorted(Path(args.data_dir).glob('traj_*.npz'))
    tr, va, te = split_by_ic(all_files, seed=args.seed)
    pool = {'train': tr, 'val': va, 'test': te, 'all': all_files}[
        args.ic_source]
    print(f"    source: {args.ic_source} split of split_by_ic("
          f"seed={args.seed}) -> {len(pool)} files")
    if args.ic_source == 'test':
        print(f"    WARNING: test-split ICs are for EVALUATION. The campaign "
              f"must draw\n             from train (handoff 19 section 9).")

    order = rng.permutation(len(pool))
    experts = []
    for i in order:
        if len(experts) >= args.n_ics:
            break
        e = load_traj(pool[i])
        if e['status'] != 0:
            continue
        e['name'] = Path(pool[i]).name
        experts.append(e)
    print(f"    selected {len(experts)} converged ICs")
    if args.require != 'any':
        print(f"    filtering to '{args.require}' rollouts (this runs the "
              f"policy twice per candidate)")
        kept = []
        for e in experts:
            r = rollout_closed_loop(model, norms, e['ic'][:7], e['t_f'],
                                    thres, is_mlp=False, use_ttg=True,
                                    clip_tilt=True, device=device)
            hit = r['reached_ground']
            if (args.require == 'ground') == hit:
                kept.append(e)
        experts = kept
        print(f"    {len(experts)} ICs match")
        if not experts:
            print(f"    none found — widen --n_ics")
            return 1

    print(f"\n[4] Mass floor policy")
    print(f"    NO post-solve filter. model.lb pins m >= m_dry = "
          f"{m_dry/1e3:.1f} t at every stage,")
    print(f"    so every converged label satisfies it by construction — the "
          f"same criterion")
    print(f"    the BC dataset was generated under. m_overhead "
          f"({m_overhead:.0f} kg) is LOGGED, not enforced.")

    print(f"\n[5] Rollout, query, splice"
          + ("  (strict assertions on EVERY splice)" if args.diagnose
             else "  (spot-check assertions)"))
    all_splices, summaries = [], []
    t0 = time.time()
    for i, e in enumerate(experts):
        strict = args.diagnose or (rng.random() < args.spotcheck_frac)
        s, sp, recs, roll, plans = process_ic(
            model, norms, bank, e, thres, device,
            args.n_late, args.n_early, rng, args.brute,
            warm=not args.no_warm, strict=strict)
        summaries.append(s)
        all_splices.extend(sp)

        if args.diagnose:
            p = diagnose_ic(s, sp, recs, roll, plans, out_dir,
                            Path(s['name']).stem)
            mrg = [x['meta']['margin_kg'] for x in sp]
            print(f"    {s['name']}: {s['category']:>15s}  K={s['steps_done']}"
                  f"  k_fail={s['k_fail']}  splices={s['n_splices']}"
                  f"  labels={s['n_labels']}  -> {p.name}")
            print(f"        m_f expert={s['m_f_stored_kg']-m_dry:.0f} kg "
                  f"above m_dry | replan margin median "
                  f"{np.median(mrg):.0f} kg, min {min(mrg):.0f} kg")
            worst = {}
            for x in sp:
                for key in ('seam_pos_m', 'coh_pos_m', 'coh_vel_ms',
                            'term_pos_m', 'ttg_err_s'):
                    worst[key] = max(worst.get(key, 0.0),
                                     x['meta']['checks'][key])
            print("        worst residuals: " + "  ".join(
                f"{k}={v:.2e}" for k, v in worst.items()))
        elif (i + 1) % 25 == 0 or i == len(experts) - 1:
            print(f"    IC {i+1:5d}/{len(experts)}  "
                  f"{len(all_splices)} splices, "
                  f"{sum(x['meta']['n_labels'] for x in all_splices)} labels",
                  flush=True)
    wall = time.time() - t0

    if not all_splices:
        print("\nNo splices produced — nothing to save.")
        return 1

    # ---- Campaign statistics --------------------------------------------
    print(f"\n[6] Yield")
    n_lab = sum(x['meta']['n_labels'] for x in all_splices)
    per_ic = np.array([s['n_splices'] for s in summaries])
    print(f"    wall time      : {wall:.1f} s "
          f"({wall/max(len(experts),1):.2f} s per IC)")
    print(f"    sequences      : {len(all_splices)}")
    print(f"    labels         : {n_lab:,}")
    print(f"    splices per IC : median {np.median(per_ic):.0f}, "
          f"mean {per_ic.mean():.1f}, min {per_ic.min()}, "
          f"max {per_ic.max()}")
    bc_labels = 9406 * 60          # corrected split, post-filter train
    print(f"    DAgger fraction of aggregate: "
          f"{n_lab/(n_lab+bc_labels):.1%}  (target 15-20%)")
    n_pol = len(all_splices)
    print(f"    policy-visited (state, expert-action) pairs: {n_pol:,}")
    print(f"    strict policy-distribution fraction: "
          f"{n_pol/(n_pol+bc_labels):.2%}")
    print(f"    (handoff 13 diagnosed 0.19% as arithmetically insufficient; "
          f"the previous campaign")
    print(f"     died there. Below ~1% expect the same outcome.)")

    # ---- Label-count-per-position histogram ------------------------------
    print(f"\n[7] Labels per output position "
          f"(set loss weighting FROM this, do not inherit it)")
    counts = np.zeros(N_NODES, dtype=int)
    for x in all_splices:
        counts += x['mask']
    for lo in range(0, N_NODES, 10):
        seg = counts[lo:lo + 10]
        bar = '#' * int(40 * seg.mean() / max(counts.max(), 1))
        print(f"    pos {lo:2d}-{lo+9:2d}: mean {seg.mean():8.0f}  {bar}")
    print(f"    ratio last-decile / first-decile: "
          f"{counts[50:].mean()/max(counts[:10].mean(),1e-9):.1f}x")

    # ---- Mass margin -----------------------------------------------------
    print(f"\n[8] Mass margin above m_dry (logged, not filtered)")
    marg = np.array([x['meta']['margin_kg'] for x in all_splices])
    print(f"    median {np.median(marg):.0f} kg, p10 "
          f"{np.percentile(marg,10):.0f}, min {marg.min():.0f}")
    print(f"    below m_overhead ({m_overhead:.0f} kg): "
          f"{np.mean(marg < m_overhead):.1%} of labels "
          f"— these would have been discarded under the tighter floor")
    print(f"    {'k range':>10}{'n':>7}{'median margin':>16}"
          f"{'vs expert IC':>15}")
    stored = {s['name']: s['m_f_stored_kg'] - m_dry for s in summaries}
    for lo in range(0, N_NODES, 10):
        sel = [x['meta'] for x in all_splices if lo <= x['meta']['k'] < lo+10]
        if not sel:
            continue
        rel = [m['margin_kg'] - stored[m['ic']] for m in sel]
        print(f"    {f'{lo}-{lo+9}':>10}{len(sel):>7}"
              f"{np.median([m['margin_kg'] for m in sel]):>16.0f}"
              f"{np.median(rel):>15.0f}")

    # ---- Correction ------------------------------------------------------
    print(f"\n[9] Correction magnitude by splice point")
    print(f"    kN columns are physical; the SIGMA column is the same "
          f"correction in\n    normalised action units — the only one that "
          f"predicts gradient signal.")
    print(f"    {'k range':>10}{'n':>7}{'vs applied':>13}{'vs raw':>10}"
          f"{'sigma':>9}{'raw dTx':>10}{'raw dTy':>10}{'raw dTz':>10}")
    for lo in range(0, N_NODES, 10):
        sel = [x['meta'] for x in all_splices if lo <= x['meta']['k'] < lo+10]
        if not sel:
            continue
        vec = np.abs(np.array([m['correction_raw_vec_kN'] for m in sel]))
        print(f"    {f'{lo}-{lo+9}':>10}{len(sel):>7}"
              f"{np.median([m['correction_kN'] for m in sel]):>13.1f}"
              f"{np.median([m['correction_raw_kN'] for m in sel]):>10.1f}"
              f"{np.median([m['correction_raw_sigma'] for m in sel]):>9.3f}"
              + "".join(f"{v:>10.1f}" for v in np.median(vec, axis=0)))
    print(f"    action_std = "
          f"{np.array2string(norms.a_sig*1e3, precision=1)} kN")

    # ---- k_fail ----------------------------------------------------------
    kf = [s['k_fail'] for s in summaries if s['k_fail'] is not None]
    if kf:
        print(f"\n[10] k_fail (primary progress metric — pre-registered as "
              f"saturation at ~52)")
        print(f"     median {np.median(kf):.0f}, p10 "
              f"{np.percentile(kf,10):.0f}, p90 {np.percentile(kf,90):.0f}")
    n_post = sum(1 for x in all_splices if x['meta']['post_kfail'])
    print(f"     splices after k_fail: {n_post} "
          f"({100*n_post/len(all_splices):.1f}%) — kept and flagged")

    # ---- Save ------------------------------------------------------------
    states = np.stack([x['states'] for x in all_splices])
    actions = np.stack([x['actions'] for x in all_splices])
    mask = np.stack([x['mask'] for x in all_splices])
    meta = [x['meta'] for x in all_splices]

    npz_path = out_dir / f'{args.tag}.npz'
    np.savez_compressed(
        npz_path,
        states=states, actions=actions, mask=mask,
        k=np.array([m['k'] for m in meta], dtype=np.int16),
        n_act=np.array([m['n_act'] for m in meta], dtype=np.int16),
        dt=np.array([m['dt'] for m in meta], dtype=np.float32),
        t_f=np.array([m['t_f'] for m in meta], dtype=np.float32),
        m_f_kg=np.array([m['m_f_kg'] for m in meta], dtype=np.float32),
        correction_kN=np.array([m['correction_kN'] for m in meta],
                               dtype=np.float32),
        correction_raw_kN=np.array([m['correction_raw_kN'] for m in meta],
                                   dtype=np.float32),
        correction_raw_sigma=np.array(
            [m['correction_raw_sigma'] for m in meta], dtype=np.float32),
        correction_raw_vec_kN=np.array(
            [m['correction_raw_vec_kN'] for m in meta], dtype=np.float32),
        post_kfail=np.array([m['post_kfail'] for m in meta], dtype=bool),
        ic=np.array([m['ic'] for m in meta]),
    )
    print(f"\n[11] Wrote {npz_path}  "
          f"({npz_path.stat().st_size/1e6:.1f} MB, "
          f"states {states.shape})")

    with open(out_dir / f'{args.tag}_summary.json', 'w') as fh:
        json.dump({
            'run_dir': str(args.run_dir), 'ic_source': args.ic_source,
            'seed': args.seed, 'n_ics': len(experts),
            'n_sequences': len(all_splices), 'n_labels': int(n_lab),
            'brute': args.brute, 'n_late': args.n_late,
            'n_early': args.n_early, 'wall_s': wall,
            'labels_per_position': counts.tolist(),
            'mass_floor': 'm_dry (no post-solve filter)',
            'per_ic': summaries,
            'splice_meta': meta,
        }, fh, indent=2, default=float)
    print(f"     Wrote {out_dir / (args.tag + '_summary.json')}")

    print("\n" + "=" * 78)
    print("ALL ASSERTIONS PASSED" if args.diagnose else "BUILD COMPLETE")
    print("=" * 78)
    return 0


def main():
    p = argparse.ArgumentParser(
        description='DAgger prefix-suffix splice builder (60-node TTG)')
    p.add_argument('--run_dir', required=True)
    p.add_argument('--bank_dir', default='pdg_bank')
    p.add_argument('--data_dir', default='data/batch003')
    p.add_argument('--ic_source', choices=['train', 'val', 'test', 'all'],
                   default='train')
    p.add_argument('--n_ics', type=int, default=800)
    p.add_argument('--seed', type=int, default=42,
                   help='SPLIT seed — MUST match ttg_train_v2 --split_seed '
                        'and standalone_eval_v2 --seed. Never vary this.')
    p.add_argument('--sample_seed', type=int, default=1,
                   help='IC draw / splice selection. Vary this per DAgger '
                        'iteration to get fresh ICs.')
    p.add_argument('--nhead', type=int, default=4,
                   help='not recoverable from the checkpoint; assumed')
    p.add_argument('--n_late', type=int, default=3,
                   help='splices taken from the last converged steps')
    p.add_argument('--n_early', type=int, default=3,
                   help='splices sampled uniformly from earlier steps')
    p.add_argument('--brute', action='store_true',
                   help='keep EVERY converged splice (handoff 19 fallback)')
    p.add_argument('--no_warm', action='store_true')
    p.add_argument('--spotcheck_frac', type=float, default=0.05,
                   help='fraction of ICs given strict assertions in '
                        'campaign mode')
    p.add_argument('--require', choices=['any', 'ground', 'airborne'],
                   default='any',
                   help="restrict to ICs whose rollout does/does not reach "
                        "the ground. Use 'ground' to exercise the K<60 "
                        "branch, which no run has touched yet.")
    p.add_argument('--diagnose', action='store_true',
                   help='strict assertions on every splice + per-IC figure')
    p.add_argument('--out_dir', default='data/dagger_iter001')
    p.add_argument('--tag', default='dagger_splices')
    return run(p.parse_args())


if __name__ == '__main__':
    sys.exit(main())
