#!/usr/bin/env python3
"""
Expert convergence test on policy-visited states.

Rolls out a trained policy and queries the fixed-dt expert bank at every
visited state. Reports convergence against rollout step, the first failed
step (k_fail), recovery after failure and the size of the expert
corrections.

Usage:
    python test_t3_real.py --run_dir runs/<run> --bank_dir pdg_bank \\
        --data_dir data/<dataset> --n_ics 50 --plot
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

from forces_pdg import (
    m_dry, m_overhead, T_max,
    ndf, ndp, ndv, ndm, ndt,
    NVAR,
)
from forces_pdg_fixeddt import solve_fixed_dt

from standalone_eval_v2 import (
    rollout_closed_loop, NormStats, PDGTransformer, TConfig,
    LandingThresholds, load_traj, split_by_ic,
    CATEGORIES, CAT_COLORS, CAT_LABELS,
    N as N_INTERVALS,
)


# =====================================================================
# Gates — declared before running
# =====================================================================

GATE_K_MAX = 45                  # i.e. n_act >= 15
GATE_CONV_RATE = 0.90

# Failure is ABSORBING if, after the first failure, essentially nothing
# converges again.
ABSORBING_THRESHOLD = 0.05       # post-failure convergence below this

M_FLOOR_KG = m_dry + m_overhead
T_MAX_MN = T_max / 1e6
FUEL_MARGIN_SPLIT_KG = 60_500.0  # matches T1 / T3-synthetic base filter


# =====================================================================
# Model loading with architecture inference
# =====================================================================

def infer_config_from_state_dict(sd, nhead_assumed=4):
    """Recover the PDGTransformer architecture from checkpoint tensor shapes."""
    d_model, state_dim = sd['input_proj.weight'].shape
    max_seq_len = sd['pos_embedding.weight'].shape[0]
    action_dim = sd['output_head.weight'].shape[0]

    layer_ids = set()
    d_ff = None
    for key in sd:
        if key.startswith('transformer_encoder.layers.'):
            layer_ids.add(int(key.split('.')[2]))
            if key.endswith('linear1.weight'):
                d_ff = sd[key].shape[0]
    num_layers = max(layer_ids) + 1 if layer_ids else 0

    return TConfig(
        state_dim=int(state_dim), action_dim=int(action_dim),
        d_model=int(d_model), nhead=int(nhead_assumed),
        num_layers=int(num_layers), d_ff=int(d_ff),
        dropout=0.0,                       # irrelevant in eval()
        max_seq_len=int(max_seq_len),
    )


def load_policy(run_dir, nhead_assumed, device):
    """Load the TTG transformer and its TTG normalisation statistics."""
    run_dir = Path(run_dir)

    ckpt_path = None
    for name in ('model_ttg.pt', 'best_model.pt', 'model.pt'):
        if (run_dir / name).exists():
            ckpt_path = run_dir / name
            break
    if ckpt_path is None:
        raise FileNotFoundError(f"No TTG checkpoint found in {run_dir}")

    ns_path = run_dir / 'norm_ttg.npz'
    if not ns_path.exists():
        raise FileNotFoundError(
            f"{ns_path} not found. This script requires the TTG "
            f"normalisation; norm_const.npz is NOT an acceptable substitute.")

    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = raw['model_state_dict'] if (isinstance(raw, dict)
                                     and 'model_state_dict' in raw) else raw

    cfg = infer_config_from_state_dict(sd, nhead_assumed)
    model = PDGTransformer(cfg).to(device)
    model.load_state_dict(sd)
    model.eval()
    norms = NormStats.load(ns_path)

    print(f"  checkpoint : {ckpt_path.name}")
    print(f"  norms      : {ns_path.name}")
    print(f"  inferred   : state_dim={cfg.state_dim} d_model={cfg.d_model} "
          f"layers={cfg.num_layers} d_ff={cfg.d_ff} "
          f"max_seq_len={cfg.max_seq_len}")
    print(f"  ASSUMED    : nhead={cfg.nhead}  "
          f"(not recoverable from tensor shapes)")
    print(f"  parameters : {model.count_parameters():,}")
    print(f"  action_std : [{norms.a_sig[0]:.4f}, {norms.a_sig[1]:.4f}, "
          f"{norms.a_sig[2]:.4f}] MN  "
          f"(denominator of the loss-space correction)")

    cfg_json = run_dir / 'config.json'
    if cfg_json.exists():
        with open(cfg_json) as fh:
            tc = json.load(fh)
        print(f"  training   : late_weight={tc.get('late_weight')} "
              f"tz_weight={tc.get('tz_weight')} seed={tc.get('seed')} "
              f"out_dir={tc.get('out_dir')}")

    return model, norms, cfg


# =====================================================================
# Stage packing for the warm-start chain
# =====================================================================

def pack_z0(x_phys, u_MN, tf_pin, n_stages):
    """Flat, stage-major, scaled guess. z = [F(3) | r(3) v(3) m | t_f]."""
    Z = np.zeros((n_stages, NVAR))
    Z[:n_stages - 1, 0:3] = u_MN * 1e6 / ndf
    Z[n_stages - 1, 0:3] = u_MN[-1] * 1e6 / ndf
    Z[:, 3:6] = x_phys[:, 0:3] / ndp
    Z[:, 6:9] = x_phys[:, 3:6] / ndv
    Z[:, 9] = x_phys[:, 6] / ndm
    Z[:, 10] = tf_pin / ndt
    return Z.reshape(-1)


def shift_plan(res, tf_pin_next):
    """Warm start for k+1 from the plan solved at k."""
    x_next = res['x_traj'][1:, :7]
    u_next = res['u_traj'][1:]
    return pack_z0(x_next, u_next, tf_pin_next, x_next.shape[0])


# =====================================================================
# Per-IC pass: rollout, then query every visited state
# =====================================================================

def process_ic(model, norms, bank, expert, thres, device, warm=True):
    """Roll the policy out on one IC and query the expert at every visited
    state.
    """
    ic_7 = expert['ic'][:7]
    tf = expert['t_f']
    dt = tf / N_INTERVALS

    roll = rollout_closed_loop(
        model, norms, ic_7, tf, thres,
        is_mlp=False, use_ttg=True, clip_tilt=True, device=device)

    x_pol = roll['x_traj']            # (K+1, 7), trimmed at termination
    u_app = roll['u_traj']            # (K, 3) enforced, actually applied
    u_raw = roll['u_traj_raw']        # (K, 3) raw network output
    K = roll['steps_done']

    x_exp = expert['x_traj'][:, :7]   # (61, 7)

    records = []
    prev = None

    for k in range(K):
        n_act = N_INTERVALS - k
        if n_act < 1 or n_act not in bank:
            continue
        solver = bank[n_act]
        tf_pin = n_act * dt
        x_k = x_pol[k]

        z0 = shift_plan(prev, tf_pin) if (warm and prev is not None) else None

        try:
            res = solve_fixed_dt(solver, x_k, tf_pin,
                                 N_stages=n_act + 1, z0=z0)
            err = None
        except Exception as e:
            res, err = None, f'{type(e).__name__}: {e}'

        d = x_k - x_exp[k]
        rec = {
            'k': k, 'n_act': n_act, 'dt': dt,
            'alt_m': float(x_k[2]), 'vz_ms': float(x_k[5]),
            'mass_kg': float(x_k[6]),
            'dev_rz_m': float(d[2]),
            'dev_vz_ms': float(d[5]),
            'dev_vh_ms': float(np.linalg.norm(d[3:5])),
            'dev_rh_m': float(np.linalg.norm(d[0:2])),
            'dev_m_kg': float(d[6]),
        }

        if res is None:
            rec.update({'converged': False, 'exitflag': -999, 'error': err})
            prev = None
        elif res['exitflag'] != 1:
            rec.update({'converged': False, 'exitflag': int(res['exitflag']),
                        'it': int(res['it']),
                        'solvetime_ms': res['solvetime'] * 1e3})
            prev = None
        else:
            u_star = res['u_traj'][0]
            d_raw = u_star - u_raw[k]        # what the LOSS sees
            d_app = u_star - u_app[k]        # what the VEHICLE saw

            rec.update({
                'converged': True, 'exitflag': 1,
                'it': int(res['it']),
                'solvetime_ms': res['solvetime'] * 1e3,
                'res_eq': res['res_eq'], 'res_ineq': res['res_ineq'],
                'm_f_kg': float(res['m_f']),
                'passes_mass_filter': bool(res['m_f'] >= M_FLOOR_KG),

                # --- Correction measures ---
                'correction_raw_kN': float(np.linalg.norm(d_raw) * 1e3),
                'correction_raw_rel': float(np.linalg.norm(d_raw) / T_MAX_MN),

                'correction_kN': float(np.linalg.norm(d_app) * 1e3),
                'correction_rel': float(np.linalg.norm(d_app) / T_MAX_MN),

                'correction_loss_space': float(
                    np.linalg.norm(d_raw / norms.a_sig)),

                # How much work the enforcement layer did at this step.
                'enforced_kN': float(
                    np.linalg.norm(u_app[k] - u_raw[k]) * 1e3),
            })
            prev = res

        records.append(rec)

    fails = [r['k'] for r in records if not r['converged']]
    k_fail = fails[0] if fails else None

    summary = {
        'name': expert['name'],
        't_f': tf, 'dt': dt,
        'm_f_stored_kg': expert['m_f'],
        'category': roll['category'],
        'reached_ground': bool(roll['reached_ground']),
        'steps_done': K,
        'term_reason': roll['term_reason'],
        'n_queries': len(records),
        'n_converged': sum(1 for r in records if r['converged']),
        'k_fail': k_fail,
        'n_clip_mag': roll['n_clip_mag'],
        'n_clip_tilt': roll['n_clip_tilt'],
    }
    return summary, records


# =====================================================================
# Reporting helpers
# =====================================================================

def _med(rs, key):
    """Median of one key over converged records; nan when empty."""
    v = [r[key] for r in rs if r.get('converged') and key in r]
    return float(np.median(v)) if v else float('nan')


def _series(recs, ks, key, stat):
    """Per-k statistic of one key over converged records."""
    out = []
    for k in ks:
        v = [r[key] for r in recs
             if r['k'] == k and r.get('converged') and key in r]
        out.append(stat(v) if v else np.nan)
    return np.array(out, dtype=float)


# =====================================================================
# Main
# =====================================================================

def run(run_dir, bank_dir, data_dir, n_ics, seed, nhead, ic_source,
        do_warm, t3_json, out_dir, make_plot):
    print("=" * 78)
    print("T3-real — EXPERT CONVERGENCE ON POLICY-VISITED STATES")
    print("=" * 78)

    device = torch.device('cpu')
    thres = LandingThresholds()

    print(f"\n[1] Loading policy from {run_dir}")
    model, norms, cfg = load_policy(run_dir, nhead, device)

    print(f"\n[2] Loading solver bank")
    bank_dir = Path(bank_dir)
    bank = {}
    for entry in sorted(bank_dir.iterdir()):
        if entry.is_dir() and entry.name.startswith('pdg_N'):
            try:
                n_int = int(entry.name.replace('pdg_N', ''))
            except ValueError:
                continue
            bank[n_int] = forcespro.nlp.Solver.from_directory(str(entry))
    if not bank:
        raise FileNotFoundError(f"No pdg_N* solvers in {bank_dir}")
    print(f"    {len(bank)} solvers, n_int in [{min(bank)}, {max(bank)}]")

    # ---- ICs (no fuel-margin filter) ----
    print(f"\n[3] Sampling ICs (no fuel-margin filter — the campaign faces "
          f"the real distribution)")
    data_dir = Path(data_dir)
    all_files = sorted(data_dir.glob('traj_*.npz'))
    if ic_source == 'test':
        _, _, pool = split_by_ic(all_files, seed=seed)
        print(f"    source: test split of split_by_ic(seed={seed}) "
              f"-> {len(pool)} files")
        print(f"    NOTE: the HPC trainer's split may differ (handoff 18 "
              f"section 7), so\n          some of these may have been seen "
              f"in training. Acceptable here:\n          this measures expert "
              f"convergence on policy-visited states, not\n          policy "
              f"generalisation.")
    else:
        pool = all_files
        print(f"    source: all {len(pool)} files")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pool))
    experts = []
    for i in order:
        if len(experts) >= n_ics:
            break
        e = load_traj(pool[i])
        if e['status'] != 0:
            continue
        e['name'] = Path(pool[i]).name
        experts.append(e)
    print(f"    selected {len(experts)} converged ICs")
    mf = np.array([e['m_f'] for e in experts])
    print(f"    m_f_stored: median {np.median(mf)/1e3:.2f} t, "
          f"{np.mean(mf < FUEL_MARGIN_SPLIT_KG):.0%} below "
          f"{FUEL_MARGIN_SPLIT_KG/1e3:.1f} t")

    # ---- Rollout + query -------------------------------------------------
    print(f"\n[4] Rolling out and querying every visited state")
    print(f"    (queries continue PAST the first failure, so the absorbing "
          f"question\n     can be answered — see [8])")
    summaries, all_recs = [], []
    t0 = time.time()
    for i, e in enumerate(experts):
        s, recs = process_ic(model, norms, bank, e, thres, device,
                             warm=do_warm)
        summaries.append(s)
        for r in recs:
            r['ic'] = s['name']
            r['category'] = s['category']
            r['m_f_stored_kg'] = s['m_f_stored_kg']
        all_recs.extend(recs)
        if (i + 1) % 10 == 0 or i == len(experts) - 1:
            c = sum(r['converged'] for r in all_recs)
            print(f"    IC {i+1:4d}/{len(experts)}  "
                  f"{len(all_recs)} queries, {c} converged "
                  f"({100*c/max(len(all_recs),1):.0f}%)", flush=True)
    wall = time.time() - t0
    print(f"    {len(all_recs)} queries in {wall:.1f} s "
          f"({1e3*wall/max(len(all_recs),1):.1f} ms each)")

    conv_all = [r for r in all_recs if r['converged']]

    # ---- Policy outcomes -------------------------------------------------
    print(f"\n[5] Policy outcomes (the model under test)")
    for c in CATEGORIES:
        n = sum(1 for s in summaries if s['category'] == c)
        if n:
            print(f"    {CAT_LABELS[c]:>20s}: {n:4d} / {len(summaries)} "
                  f"({100*n/len(summaries):5.1f}%)")

    # ---- Convergence vs k: the point-of-no-return curve ------------------
    print(f"\n[6] Expert convergence and correction vs rollout step")
    print(f"    'raw'     = ||u* - u_raw||      the DAgger gradient signal")
    print(f"    'appl'    = ||u* - u_applied||  physical tracking error")
    print(f"    'loss-sp' = ||(u* - u_raw)/sigma_a||  comparable to val MSE")
    print(f"    'enforc'  = ||u_applied - u_raw||     work done by the clipper")
    bins = list(range(0, 60, 5))
    print(f"\n    {'k range':>10}{'n_act':>10}{'n':>6}{'conv':>8}"
          f"{'post-filt':>11}{'raw kN':>10}{'appl kN':>10}"
          f"{'loss-sp':>10}{'enforc kN':>11}")
    print("    " + "-" * 82)
    for lo in bins:
        hi = lo + 4
        sel = [r for r in all_recs if lo <= r['k'] <= hi]
        if not sel:
            continue
        conv = [r for r in sel if r['converged']]
        pf = [r for r in conv if r.get('passes_mass_filter')]
        print(f"    {f'{lo}-{hi}':>10}{f'{60-lo}-{60-hi}':>10}{len(sel):>6}"
              f"{len(conv)/len(sel):>8.0%}{len(pf)/len(sel):>11.0%}"
              f"{_med(conv,'correction_raw_kN'):>10.1f}"
              f"{_med(conv,'correction_kN'):>10.1f}"
              f"{_med(conv,'correction_loss_space'):>10.3f}"
              f"{_med(conv,'enforced_kN'):>11.1f}")

    if conv_all:
        raw_med = _med(conv_all, 'correction_raw_kN')
        app_med = _med(conv_all, 'correction_kN')
        enf_med = _med(conv_all, 'enforced_kN')
        ls_med = _med(conv_all, 'correction_loss_space')
        ratio = raw_med / app_med if app_med > 1e-12 else float('inf')
        print(f"\n    overall medians: raw {raw_med:.1f} kN, "
              f"applied {app_med:.1f} kN, enforced {enf_med:.1f} kN, "
              f"loss-space {ls_med:.3f}")
        print(f"    raw / applied ratio: {ratio:.2f}")
        if ratio > 2.0:
            print(f"    -> the enforcement layer is masking an error "
                  f"substantially larger than\n       the applied-units "
                  f"correction suggests. The DAgger gradient signal is\n"
                  f"       bigger than physical tracking error implies.")
        elif ratio < 1.2:
            print(f"    -> raw and applied corrections agree; the clipper is "
                  f"not materially\n       changing the picture, and either "
                  f"measure reads the same way.")
        else:
            print(f"    -> moderate divergence; quote the raw measure when "
                  f"discussing learning\n       signal and the applied "
                  f"measure when discussing tracking.")

    gate_sel = [r for r in all_recs if r['k'] <= GATE_K_MAX]
    gate_rate = (np.mean([r['converged'] for r in gate_sel])
                 if gate_sel else 0.0)

    # ---- k_fail ----------------------------------------------------------
    print(f"\n[7] k_fail distribution")
    kf = [s['k_fail'] for s in summaries if s['k_fail'] is not None]
    n_never = sum(1 for s in summaries if s['k_fail'] is None)
    if kf:
        kf = np.array(kf)
        print(f"    ICs with a failure : {len(kf)}/{len(summaries)}")
        print(f"    never failed       : {n_never}")
        print(f"    k_fail  median {np.median(kf):.0f}, "
              f"p10 {np.percentile(kf,10):.0f}, "
              f"p90 {np.percentile(kf,90):.0f}, "
              f"min {kf.min()}, max {kf.max()}")
        early = int(np.sum(kf <= 5))
        print(f"    k_fail <= 5        : {early} "
              f"({100*early/len(summaries):.0f}%)  "
              f"[expect ~5% from ICs on the fuel floor]")
    else:
        print(f"    no IC ever produced an expert failure")

    # ---- Is failure absorbing? ------------------------------------------
    print(f"\n[8] Is expert failure ABSORBING? "
          f"(decides handoff 19, Decision 4)")
    post_total = post_conv = 0
    n_reentry = 0
    for s in summaries:
        if s['k_fail'] is None:
            continue
        after = [r for r in all_recs
                 if r['ic'] == s['name'] and r['k'] > s['k_fail']]
        if not after:
            continue
        post_total += len(after)
        c = sum(1 for r in after if r['converged'])
        post_conv += c
        if c > 0:
            n_reentry += 1
    if post_total:
        rate = post_conv / post_total
        print(f"    queries after the first failure : {post_total}")
        print(f"    of which converged              : {post_conv} "
              f"({rate:.1%})")
        print(f"    ICs that re-entered the recoverable set : {n_reentry}")
        if rate < ABSORBING_THRESHOLD:
            print(f"    -> FAILURE IS ABSORBING. Decision 4 confirmed: "
                  f"abandon on first\n       failure discards nothing "
                  f"usable.")
        else:
            print(f"    -> FAILURE IS NOT ABSORBING. Abandon-on-first-failure "
                  f"would discard\n       {post_conv} usable labels "
                  f"({rate:.0%} of post-failure queries). Either add a\n"
                  f"       patience parameter, or drop the rule and simply "
                  f"mask failed queries.")
    else:
        print(f"    no post-failure queries to assess")

    # ---- Labels available ------------------------------------------------
    print(f"\n[9] Label yield per IC (campaign sizing)")
    nconv = np.array([s['n_converged'] for s in summaries])
    print(f"    converged queries per IC: median {np.median(nconv):.0f}, "
          f"mean {nconv.mean():.1f}, min {nconv.min()}, max {nconv.max()}")
    print(f"    total per IC (queries)  : "
          f"mean {np.mean([s['n_queries'] for s in summaries]):.1f}")
    bc_labels = 9419 * 60
    for frac in (0.15, 0.20):
        need = frac / (1 - frac) * bc_labels
        n_ic_needed = need / max(nconv.mean(), 1)
        secs = n_ic_needed * wall / max(len(experts), 1)
        print(f"    for {frac:.0%} DAgger fraction: ~{need/1e3:.0f}k labels "
              f"-> ~{n_ic_needed:,.0f} ICs -> ~{secs/60:.0f} min of "
              f"rollout+solve")
    print(f"    (assumes ~{bc_labels/1e3:.0f}k BC labels and one label per "
          f"converged query;\n     splicing multiplies this — see handoff 19 "
          f"section 6)")

    # ---- Fuel margin split ----------------------------------------------
    print(f"\n[10] Split by stored fuel margin")
    for label, sel in (
            (f'm_f >= {FUEL_MARGIN_SPLIT_KG/1e3:.1f} t',
             [s for s in summaries
              if s['m_f_stored_kg'] >= FUEL_MARGIN_SPLIT_KG]),
            (f'm_f <  {FUEL_MARGIN_SPLIT_KG/1e3:.1f} t',
             [s for s in summaries
              if s['m_f_stored_kg'] < FUEL_MARGIN_SPLIT_KG])):
        if not sel:
            continue
        names = {s['name'] for s in sel}
        rs = [r for r in all_recs if r['ic'] in names]
        kfs = [s['k_fail'] for s in sel if s['k_fail'] is not None]
        print(f"    {label:<22} n={len(sel):4d}  "
              f"conv {np.mean([r['converged'] for r in rs]):.0%}  "
              f"median k_fail "
              f"{np.median(kfs) if kfs else float('nan'):.0f}")

    # ---- Cross-validation against T3-synthetic --------------------------
    if t3_json:
        print(f"\n[11] Cross-validation against T3-synthetic")
        try:
            with open(t3_json) as fh:
                syn = json.load(fh)
            n_grid = syn['n_act_grid']
            print(f"     Observed deviation at k_fail vs the synthetic "
                  f"recoverable radius\n     for the corresponding horizon. "
                  f"Agreement means the radius table is\n     predictive and "
                  f"can be quoted as a tolerance budget.")
            print(f"     {'n_act':>7}{'n':>5}{'|dv_z| obs':>12}"
                  f"{'vz_up rad':>11}{'|dr_z| obs':>12}{'rz_up rad':>11}")
            fails = [(s, [r for r in all_recs
                          if r['ic'] == s['name'] and r['k'] == s['k_fail']])
                     for s in summaries if s['k_fail'] is not None]
            for na in n_grid:
                sel = [rs[0] for s, rs in fails
                       if rs and rs[0]['n_act'] == na]
                if not sel:
                    continue
                j = n_grid.index(na)
                rv = syn['radius_raw']['vz_up'][j]
                rr = syn['radius_raw']['rz_up'][j]
                print(f"     {na:>7}{len(sel):>5}"
                      f"{np.median([abs(r['dev_vz_ms']) for r in sel]):>12.2f}"
                      f"{('—' if rv is None else f'{rv:.0f}'):>11}"
                      f"{np.median([abs(r['dev_rz_m']) for r in sel]):>12.1f}"
                      f"{('—' if rr is None else f'{rr:.0f}'):>11}")
            print(f"\n     CAVEAT: if the observed deviations fall BELOW the "
                  f"smallest tested\n     synthetic magnitude, the grid "
                  f"cannot resolve the radius in the region\n     that "
                  f"matters. Re-run T3-synthetic with finer magnitudes "
                  f"(0.1 / 0.25 / 0.5\n     m/s) at n_act <= 15 before "
                  f"quoting the table as predictive.")
        except Exception as e:
            print(f"     could not read {t3_json}: {e}")

    # ---- Verdict ---------------------------------------------------------
    passed = gate_rate >= GATE_CONV_RATE
    print("\n" + "=" * 78)
    print(f"T3-real VERDICT: {'PASS' if passed else 'REVIEW'}")
    print("=" * 78)
    print(f"  expert convergence for k <= {GATE_K_MAX} "
          f"(n_act >= {60-GATE_K_MAX}): {gate_rate:.1%} "
          f"(gate {GATE_CONV_RATE:.0%})")
    if passed:
        print("  The expert can label the states this policy visits, over the")
        print("  region where it has authority. The campaign is viable as")
        print("  specified. Next: validate the DAgger splice construction —")
        print("  nothing so far has tested the label form itself.")
    else:
        print("  Expert convergence is below the gate on policy-visited "
              "states.")
        print("  Before softening the failure rule, check [6] against [11]: "
              "if the")
        print("  failures track the synthetic radius, the policy is simply "
              "too far")
        print("  off-schedule and the answer is a better starting policy, not "
              "a more")
        print("  permissive expert.")

    # ---- Persist ---------------------------------------------------------
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        'verdict': 'PASS' if passed else 'REVIEW',
        'run_dir': str(run_dir),
        'gate': {'k_max': GATE_K_MAX, 'rate': GATE_CONV_RATE},
        'gate_conv_rate': float(gate_rate),
        'n_ics': len(summaries),
        'ic_source': ic_source,
        'seed': seed,
        'nhead_assumed': nhead,
        'warm': do_warm,
        'wall_s': wall,
        'action_std': [float(v) for v in norms.a_sig],
        'correction_medians': ({
            'raw_kN': _med(conv_all, 'correction_raw_kN'),
            'applied_kN': _med(conv_all, 'correction_kN'),
            'loss_space': _med(conv_all, 'correction_loss_space'),
            'enforced_kN': _med(conv_all, 'enforced_kN'),
        } if conv_all else None),
        'policy_categories': {c: sum(1 for s in summaries
                                     if s['category'] == c)
                              for c in CATEGORIES},
        'k_fail': [s['k_fail'] for s in summaries],
        'per_ic': summaries,
        'records': all_recs,
    }
    jpath = out_dir / 't3_real.json'
    with open(jpath, 'w') as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\nSummary written to {jpath}")

    if make_plot:
        p = make_plots(all_recs, summaries, out_dir)
        print(f"Plot written to {p}")

    return 0 if passed else 1


# =====================================================================
# Plots
# =====================================================================

def make_plots(recs, summaries, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('T3-real — expert convergence on policy-visited states',
                 fontweight='bold')

    ks = sorted({r['k'] for r in recs})

    # (0,0) convergence vs k
    a = ax[0, 0]
    rate = [np.mean([r['converged'] for r in recs if r['k'] == k])
            for k in ks]
    pf = [np.mean([r['converged'] and r.get('passes_mass_filter', False)
                   for r in recs if r['k'] == k]) for k in ks]
    a.plot(ks, rate, 'o-', ms=3, label='converged')
    a.plot(ks, pf, 's--', ms=3, label='after mass filter')
    a.axvline(GATE_K_MAX, color='r', ls=':', lw=1, label=f'k={GATE_K_MAX}')
    a.set_xlabel('rollout step k')
    a.set_ylabel('expert convergence rate')
    a.set_title('Point-of-no-return curve')
    a.legend(fontsize=8)

    # (0,1) k_fail histogram by category
    a = ax[0, 1]
    data, colors, labels = [], [], []
    for c in CATEGORIES:
        v = [s['k_fail'] for s in summaries
             if s['category'] == c and s['k_fail'] is not None]
        if v:
            data.append(v)
            colors.append(CAT_COLORS[c])
            labels.append(CAT_LABELS[c])
    if data:
        a.hist(data, bins=20, stacked=True, color=colors, label=labels)
        a.legend(fontsize=7)
    a.set_xlabel(r'$k_{fail}$')
    a.set_ylabel('count')
    a.set_title('First unrecoverable failure, by policy outcome')

    a = ax[0, 2]
    raw_med = _series(recs, ks, 'correction_raw_kN', np.median)
    raw_p25 = _series(recs, ks, 'correction_raw_kN',
                      lambda v: np.percentile(v, 25))
    raw_p75 = _series(recs, ks, 'correction_raw_kN',
                      lambda v: np.percentile(v, 75))
    app_med = _series(recs, ks, 'correction_kN', np.median)
    enf_med = _series(recs, ks, 'enforced_kN', np.median)

    a.plot(ks, raw_med, 'o-', ms=3, color='tab:red',
           label=r'raw $\|u^*-u_{raw}\|$ (loss sees this)')
    a.fill_between(ks, raw_p25, raw_p75, alpha=0.20, color='tab:red')
    a.plot(ks, app_med, 's--', ms=3, color='tab:blue',
           label=r'applied $\|u^*-u_{app}\|$ (vehicle saw this)')
    a.plot(ks, enf_med, '^:', ms=3, color='tab:gray',
           label=r'enforced $\|u_{app}-u_{raw}\|$ (clipper)')
    a.set_xlabel('rollout step k')
    a.set_ylabel('correction [kN]')
    a.set_title('DAgger correction — raw vs applied\n'
                '(median; band is raw IQR)')
    a.legend(fontsize=7)

    # (1,0) deviation channels vs k
    a = ax[1, 0]
    for key, lab in (('dev_vz_ms', r'$|\Delta v_z|$ [m/s]'),
                     ('dev_vh_ms', r'$|\Delta v_h|$ [m/s]')):
        a.plot(ks, [np.median([abs(r[key]) for r in recs if r['k'] == k])
                    for k in ks], 'o-', ms=3, label=lab)
    a.set_xlabel('rollout step k')
    a.set_ylabel('median deviation')
    a.set_title('Covariate shift: velocity deviation from expert')
    a.legend(fontsize=8)

    a = ax[1, 1]
    ls_med = _series(recs, ks, 'correction_loss_space', np.median)
    ls_p25 = _series(recs, ks, 'correction_loss_space',
                     lambda v: np.percentile(v, 25))
    ls_p75 = _series(recs, ks, 'correction_loss_space',
                     lambda v: np.percentile(v, 75))
    a.plot(ks, ls_med, 'o-', ms=3, color='tab:green')
    a.fill_between(ks, ls_p25, ls_p75, alpha=0.20, color='tab:green')
    a.set_xlabel('rollout step k')
    a.set_ylabel(r'$\|(u^*-u_{raw})/\sigma_a\|$')
    a.set_title('Correction in loss space (median, IQR)\n'
                'comparable to logged validation MSE')

    # (1,2) failure vs deviation
    a = ax[1, 2]
    okc = [r for r in recs if r['converged']]
    bad = [r for r in recs if not r['converged']]
    a.scatter([r['k'] for r in okc], [r['dev_vz_ms'] for r in okc],
              s=5, alpha=0.25, color='tab:green', label='expert converged')
    a.scatter([r['k'] for r in bad], [r['dev_vz_ms'] for r in bad],
              s=8, alpha=0.5, color='tab:red', label='expert failed')
    a.axhline(0, color='k', lw=0.5)
    a.set_xlabel('rollout step k')
    a.set_ylabel(r'$\Delta v_z$ [m/s]  (policy $-$ expert)')
    a.set_title('Where failure lives in deviation space\n'
                r'(positive $\Delta v_z$ = policy rising relative to expert)')
    a.legend(fontsize=8)

    for axis in ax.flat:
        axis.grid(alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path = Path(out_dir) / 't3_real.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# =====================================================================
# CLI
# =====================================================================

def main():
    p = argparse.ArgumentParser(
        description='T3-real — expert convergence on policy-visited states')
    p.add_argument('--run_dir', required=True,
                   help='e.g. runs_hpc/ttg_late_w3.0_seed42')
    p.add_argument('--bank_dir', default='pdg_bank')
    p.add_argument('--data_dir', default='data/batch003')
    p.add_argument('--n_ics', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--nhead', type=int, default=4,
                   help='not recoverable from the checkpoint; assumed')
    p.add_argument('--ic_source', choices=['test', 'all'], default='test')
    p.add_argument('--no_warm', action='store_true')
    p.add_argument('--t3_synthetic_json', default=None,
                   help='results/t3_synthetic/t3_synthetic.json, for the '
                        'cross-validation table')
    p.add_argument('--out_dir', default='results/t3_real')
    p.add_argument('--plot', action='store_true')
    args = p.parse_args()

    return run(args.run_dir, args.bank_dir, args.data_dir, args.n_ics,
               args.seed, args.nhead, args.ic_source, not args.no_warm,
               args.t3_synthetic_json, args.out_dir, args.plot)


if __name__ == '__main__':
    sys.exit(main())