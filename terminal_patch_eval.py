#!/usr/bin/env python3
"""
Hybrid guidance evaluation with a triggered terminal law (r = v = 0 policy).

The network flies until the trigger condition fires (altitude and/or
radius), then the analytic terminal law flies to touchdown at --dt_term.
Baseline (network only) and hybrid are run on the same ICs.

Usage:
    python terminal_patch_eval.py --run_dir runs/<run> \\
        --data_dir data/<dataset> --ttg --trigger radius --r_switch 10 \\
        --dt_term 0.1 --n_ics 300
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from standalone_eval_v2 import (
    N, nx, nu, g0, m_dry, T_min_MN, T_max_MN, theta_max, _F_rk4,
    enforce_thrust, tilt_deg, classify_landing, LandingThresholds,
    load_traj, split_by_ic, load_model_from_dir, _state_8,
    rollout_closed_loop, CATEGORIES,
)

EPS = 1e-6


# =====================================================================
# 1. The analytic terminal law
# =====================================================================

def terminal_command(x7, cfg):
    """Return (u_MN, diag) for one terminal step."""
    r_h = x7[:2].astype(float)
    v_h = x7[3:5].astype(float)
    r_z = float(x7[2])
    v_z = float(x7[5])
    m = float(x7[6])

    h_eff = r_z - cfg["h_c"]

    # --- vertical ---
    if h_eff > cfg["eps_rz"]:
        if v_z < 0.0:
            a_req = (v_z ** 2 - cfg["v_td"] ** 2) / (2.0 * h_eff)
        else:
            a_req = 0.0                      # ascending: T_z = m g (hover), no braking
    else:
        # capture phase: regulate v_z toward -v_td with a first-order law
        a_req = (-cfg["v_td"] - v_z) / cfg["tau_capture"]
    a_req = float(np.clip(a_req, 0.0, cfg["a_cap"]))

    # --- horizontal: ZEM/ZEV to r_h = 0, v_h = 0 ---
    if v_z < -cfg["eps_v"] and h_eff > cfg["eps_rz"]:
        t_go = 2.0 * h_eff / abs(v_z)
    else:
        t_go = cfg["t_go_max"]
    t_go = float(np.clip(t_go, cfg["t_go_min"], cfg["t_go_max"]))
    a_hor = -6.0 * r_h / t_go ** 2 - 4.0 * v_h / t_go
    if cfg["no_horizontal"]:
        a_hor = np.zeros(2)

    # --- assemble, convert to MN, enforce admissible set ---
    a_des = np.array([a_hor[0], a_hor[1], a_req + g0])
    u_raw = m * a_des / 1e6
    u = enforce_thrust(u_raw, clip_tilt=True)

    a_max = T_max_MN * 1e6 * np.cos(theta_max) / m - g0
    diag = dict(a_req=a_req, a_max=a_max,
                authority=a_req / max(a_max, EPS),
                clipped=bool(abs(np.linalg.norm(u) -
                                 np.linalg.norm(u_raw)) > 1e-10),
                mag_raw_MN=float(np.linalg.norm(u_raw)),
                below_Tmin=bool(np.linalg.norm(u_raw) < T_min_MN),
                a_h_req=float(np.linalg.norm(a_hor)),
                a_h_max=float(T_max_MN * 1e6 * np.sin(theta_max) / m))
    return u, diag


def should_switch(x7, cfg):
    r_z = float(x7[2])
    r_h = float(np.linalg.norm(x7[:2]))
    alt_ok = r_z <= cfg["h_switch"]
    rad_ok = r_h <= cfg["r_switch"]
    if cfg["trigger"] == "altitude":
        return alt_ok
    if cfg["trigger"] == "radius":
        return rad_ok
    if cfg["trigger"] == "both":
        return alt_ok and rad_ok
    return alt_ok or rad_ok                                  # "either"


# =====================================================================
# 2. Hybrid rollout
# =====================================================================

def rollout_hybrid(model, norms, ic_7, tf, thres, cfg, use_ttg=True,
                   is_mlp=False, device=torch.device("cpu")):
    """Network on the 60-node grid until the trigger fires, then the analytic
    law at cfg['dt_term'] until ground contact.
    """
    dt_nn = tf / N
    x = ic_7.astype(float).copy()
    xs, us, dts = [x.copy()], [], []
    switched = False
    handoff = None
    term_reason = ""
    diags = []

    # ---------- phase 1: network ----------
    model.eval()
    if not is_mlp:
        buf = np.zeros((1, N, 8), dtype=np.float32)
    with torch.no_grad():
        for k in range(N):
            if should_switch(x, cfg):
                switched = True
                break
            s8 = _state_8(x, tf, k, use_ttg)
            if is_mlp:
                inp = torch.from_numpy(
                    norms.norm_state(s8.reshape(1, -1))).float().to(device)
                a_n = model(inp)[0].cpu().numpy()
            else:
                buf[0, k, :] = s8
                inp = torch.from_numpy(
                    norms.norm_state(buf.reshape(-1, 8)).reshape(1, N, 8)
                ).float().to(device)
                a_n = model(inp)[0, k, :].cpu().numpy()
            u = enforce_thrust(
                norms.unnorm_action(a_n.reshape(1, -1)).flatten(),
                clip_tilt=True)
            x_next = np.array(_F_rk4(x, u, dt_nn)).flatten()
            us.append(u); dts.append(dt_nn); xs.append(x_next.copy())
            if np.any(~np.isfinite(x_next)):
                term_reason = "NaN in network phase"; x = x_next; break
            if x_next[2] < 0.0:
                term_reason = "ground during network phase"; x = x_next; break
            if x_next[6] < m_dry:
                term_reason = "fuel exhausted (network phase)"
                x = x_next; break
            x = x_next

    if switched:
        handoff = dict(
            r_z=float(x[2]), v_z=float(x[5]),
            r_h=float(np.linalg.norm(x[:2])),
            v_h=float(np.linalg.norm(x[3:5])),
            m=float(x[6]), mass_margin_kg=float(x[6] - m_dry),
            node=len(us),
        )
        h_eff = max(x[2] - cfg["h_c"], cfg["eps_rz"])
        a0 = (x[5] ** 2 - cfg["v_td"] ** 2) / (2 * h_eff) if x[5] < 0 else 0.0
        a_max0 = T_max_MN * 1e6 * np.cos(theta_max) / x[6] - g0
        handoff["a_req"] = float(max(a0, 0.0))
        handoff["authority"] = float(max(a0, 0.0) / max(a_max0, EPS))

        # ---------- phase 2: analytic law ----------
        dt_t = cfg["dt_term"]
        n_max = int(np.ceil(cfg["t_term_max"] / dt_t))
        for _ in range(n_max):
            u, d = terminal_command(x, cfg)
            diags.append(d)
            x_next = np.array(_F_rk4(x, u, dt_t)).flatten()
            us.append(u); dts.append(dt_t); xs.append(x_next.copy())
            if np.any(~np.isfinite(x_next)):
                term_reason = "NaN in terminal phase"; x = x_next; break
            if x_next[2] < 0.0:
                term_reason = "ground contact"; x = x_next; break
            if x_next[6] < m_dry:
                term_reason = "fuel exhausted (terminal phase)"
                x = x_next; break
            x = x_next
        else:
            term_reason = f"terminal phase timed out ({cfg['t_term_max']} s)"

    x_traj = np.array(xs)
    u_traj = np.array(us) if us else np.zeros((0, nu))

    # ---------- touchdown ----------
    td_state = None
    if len(x_traj) >= 2 and x_traj[-1, 2] < 0.0 <= x_traj[-2, 2]:
        lo, hi = 0.0, dts[-1]
        x_lo = x_traj[-2]
        for _ in range(thres.bisect_max_iter):
            mid = 0.5 * (lo + hi)
            xm = np.array(_F_rk4(x_lo, u_traj[-1], mid)).flatten()
            if xm[2] > 0:
                lo = mid
            else:
                hi = mid
        td_state = np.array(_F_rk4(x_lo, u_traj[-1], 0.5 * (lo + hi))).flatten()
    elif 0.0 <= x_traj[-1, 2] <= thres.grazing_alt_m:
        td_state = x_traj[-1].copy()

    if td_state is not None:
        category, details = classify_landing(td_state, thres)
    elif "fuel" in term_reason:
        category, details = "fuel_exhausted", {}
    elif "NaN" in term_reason:
        category, details = "diverged", {}
    else:
        category, details = "airborne", {}

    return dict(category=category, details=details, td_state=td_state,
                x_traj=x_traj, u_traj=u_traj, switched=switched,
                handoff=handoff, term_reason=term_reason,
                reached_ground=td_state is not None,
                m0=float(x_traj[0, 6]),
                m_handoff=float(handoff["m"]) if handoff else None,
                m_final=float(td_state[6]) if td_state is not None
                else float(x_traj[-1, 6]),
                 n_below_Tmin=sum(d["below_Tmin"] for d in diags),
                n_terminal_steps=len(diags),
                min_mag_raw_MN=float(min((d["mag_raw_MN"] for d in diags),
                                        default=float("nan"))),
                frac_h_saturated=float(np.mean(
                    [d["a_h_req"] > d["a_h_max"] for d in diags])) if diags
                    else float("nan"),
                max_authority=float(max((d["authority"] for d in diags),
                                default=float("nan"))),
                t_terminal=float(sum(dts[handoff["node"]:])) if handoff else 0.0)


# =====================================================================
# 3. Main
# =====================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", default="results/terminal_patch")
    p.add_argument("--n_ics", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ttg", action="store_true", default=False)
    p.add_argument("--model_type", default="auto")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--norms", default=None)
    # trigger
    p.add_argument("--trigger", default="radius",
                   choices=["altitude", "radius", "both", "either"])
    p.add_argument("--h_switch", type=float, default=150.0)
    p.add_argument("--r_switch", type=float, default=10.0)
    # law
    p.add_argument("--dt_term", type=float, default=0.1)
    p.add_argument("--h_c", type=float, default=0.0,
                   help="capture altitude [m]; 0 = pure arc to the pad")
    p.add_argument("--v_td", type=float, default=0.0,
                   help="target descent speed at/below h_c [m/s]")
    p.add_argument("--tau_capture", type=float, default=0.3)
    p.add_argument("--t_term_max", type=float, default=60.0)
    p.add_argument("--no_horizontal", action="store_true", default=False)
    p.add_argument("--skip_baseline", action="store_true", default=False)
    a = p.parse_args()

    cfg = dict(trigger=a.trigger, h_switch=a.h_switch, r_switch=a.r_switch,
               dt_term=a.dt_term, h_c=a.h_c, v_td=a.v_td,
               tau_capture=a.tau_capture, t_term_max=a.t_term_max,
               no_horizontal=a.no_horizontal,
               eps_rz=0.01, eps_v=0.01, t_go_min=0.2, t_go_max=60.0,
               a_cap=50.0)

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    thres = LandingThresholds()

    model, norms, is_mlp = load_model_from_dir(
        a.run_dir, device, model_type=a.model_type,
        checkpoint_file=a.checkpoint, norm_file=a.norms)

    files = sorted(Path(a.data_dir).glob("traj_*.npz"))
    _, _, test_files = split_by_ic(files, seed=a.seed)
    print(f"\ntest split {len(test_files)} | running {a.n_ics}")
    print(f"trigger={a.trigger}  h_switch={a.h_switch}  "
          f"r_switch={a.r_switch}  dt_term={a.dt_term}  "
          f"h_c={a.h_c}  v_td={a.v_td}")

    hyb, base, hand,experts = [], [], [],[]
    n = 0
    for f in test_files:
        if n >= a.n_ics:
            break
        e = load_traj(f)
        if e["status"] != 0:
            continue
        n += 1
        ic, tf = e["ic"][:7], float(e["t_f"])
        r = rollout_hybrid(model, norms, ic, tf, thres, cfg,
                           use_ttg=a.ttg, is_mlp=is_mlp, device=device)
        hyb.append(r)
        experts.append(e)
        if r["handoff"]:
            hand.append(r["handoff"])
        if not a.skip_baseline:
            base.append(rollout_closed_loop(
                model, norms, ic, tf, thres, is_mlp=is_mlp,
                use_ttg=a.ttg, clip_tilt=True, device=device))

    def counts(rs):
        return {c: sum(1 for r in rs if r["category"] == c)
                for c in CATEGORIES}

    ch = counts(hyb)
    cb = counts(base) if base else None
    n_sw = sum(1 for r in hyb if r["switched"])

    print(f"\n{'category':>16s} {'baseline':>10s} {'hybrid':>10s} {'delta':>8s}")
    for c in CATEGORIES:
        b = cb[c] if cb else 0
        print(f"{c:>16s} {b:>10d} {ch[c]:>10d} {ch[c]-b:>+8d}")
    print(f"{'switched':>16s} {'-':>10s} {n_sw:>10d}")

    import csv
    with open(out / "failures.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cat", "h_r_z", "h_v_z", "h_r_h", "h_v_h", "h_auth",
                    "frac_h_sat", "td_r_h", "td_vz", "td_vh",
                    "fail_vz", "fail_vh"])
        for r in hyb:
            if r["category"] not in ("crash", "hard"):
                continue
            h, d = r["handoff"] or {}, r["details"]
            w.writerow([r["category"], h.get("r_z"), h.get("v_z"),
                        h.get("r_h"), h.get("v_h"), h.get("authority"),
                        r["frac_h_saturated"], d.get("r_h"),
                        d.get("vz_signed"), d.get("vh"),
                        abs(d.get("vz_signed", 0)) > thres.hard_vz_ms,
                        d.get("vh", 0) > thres.hard_vh_norm_ms])
    with open(out / "handoff_margins.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "r_z", "v_z", "r_h", "mass_margin_kg"])
        for r in hyb:
            h = r["handoff"]
            if h:
                w.writerow([r["category"], h["r_z"], h["v_z"],
                            h["r_h"], h["mass_margin_kg"]])

    tmin_hits = sum(r["n_below_Tmin"] for r in hyb)
    tmin_steps = sum(r["n_terminal_steps"] for r in hyb)
    print(f"\n  T_min floor: {tmin_hits}/{tmin_steps} terminal steps "
          f"({100*tmin_hits/max(tmin_steps,1):.2f}%), "
          f"min raw = {min(r['min_mag_raw_MN'] for r in hyb)*1e3:.0f} kN "
          f"(T_min = {T_min_MN*1e3:.0f} kN)")
    print(f"  horizontal saturated: median "
          f"{np.median([r['frac_h_saturated'] for r in hyb if r['switched']]):.2f} "
          f"of terminal steps")

    def q(v):
        v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
        if v.size == 0:
            return (float("nan"),) * 3
        return (float(np.median(v)), float(np.percentile(v, 25)),
                float(np.percentile(v, 75)))

    summary = dict(
        n=n, n_switched=n_sw, config={k: cfg[k] for k in sorted(cfg)},
        categories_hybrid=ch, categories_baseline=cb,
        soft_rate_hybrid=ch["soft"] / max(n, 1),
        soft_rate_baseline=(cb["soft"] / max(n, 1)) if cb else None,
    )
    landed = [(h, b, e) for h, b, e in zip(hyb, base or hyb, experts)
                  if h["category"] in ("soft", "hard", "crash")]
    if landed:
        prop_h = np.array([h["m0"] - h["m_final"] for h, _, _ in landed])
        prop_e = np.array([float(e["ic"][6]) - float(e["m_f"])
                            for _, _, e in landed])
        prop_nn = np.array([h["m0"] - h["m_handoff"] for h, _, _ in landed
                            if h["m_handoff"] is not None])
        prop_t = np.array([h["m_handoff"] - h["m_final"] for h, _, _ in landed
                            if h["m_handoff"] is not None])
        summary["fuel"] = {
            "n_landed": len(landed),
            "expert_propellant_kg": q(prop_e),
            "hybrid_propellant_kg": q(prop_h),
            "excess_vs_expert_kg": q(prop_h - prop_e),
            "excess_vs_expert_pct": q(100 * (prop_h - prop_e) / prop_e),
            "network_phase_kg": q(prop_nn),
            "terminal_phase_kg": q(prop_t),
        }
        print("\n--- fuel (landed ICs only) ---")
        for k, v in summary["fuel"].items():
            if isinstance(v, tuple):
                print(f"  {k:28s}: {v[0]:9.1f}  [{v[1]:.1f}, {v[2]:.1f}]")
    if hand:
        summary["handoff"] = {
            "altitude_m": q([h["r_z"] for h in hand]),
            "v_z_ms": q([h["v_z"] for h in hand]),
            "r_h_m": q([h["r_h"] for h in hand]),
            "v_h_ms": q([h["v_h"] for h in hand]),
            "a_req_ms2": q([h["a_req"] for h in hand]),
            "authority_a_req_over_a_max": q([h["authority"] for h in hand]),
            "frac_infeasible_at_handoff":
                float(np.mean([h["authority"] > 1.0 for h in hand])),
            "mass_margin_kg": q([h["mass_margin_kg"] for h in hand]),
            "node": q([h["node"] for h in hand]),
        }
        summary["terminal_phase_duration_s"] = q(
            [r["t_terminal"] for r in hyb if r["switched"]])
        summary["max_authority_during_terminal"] = q(
            [r["max_authority"] for r in hyb if r["switched"]])
        print("\n--- handoff (median [q25, q75]) ---")
        for k, v in summary["handoff"].items():
            if isinstance(v, tuple):
                print(f"  {k:34s}: {v[0]:9.3f}  [{v[1]:.3f}, {v[2]:.3f}]")
            else:
                print(f"  {k:34s}: {v}")
        fi = summary["handoff"]["frac_infeasible_at_handoff"]
        if fi > 0.02:
            print(f"\n  WARNING: {fi:.1%} of handoffs need more deceleration "
                  f"than T_max can give.\n  Trigger is firing too late — "
                  f"switch earlier or raise h_switch.")

    with open(out / "terminal_patch.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved {out/'terminal_patch.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())