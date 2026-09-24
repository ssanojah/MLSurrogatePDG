#!/usr/bin/env python3
"""
Hybrid guidance evaluation: gate policy followed by the analytic terminal law.

The network flies its 60-node trajectory to the gate (20 m, -16 m/s), then
the analytic terminal law flies to touchdown at --dt_term. Writes
gate_patch.json, gate_arrivals.csv, failures.csv, ic_outcomes.csv and figures.

Usage:
    python gate_patch_eval.py --run_dir runs/<gate_run> \\
        --data_dir data/<gate_dataset> --opt_data_dir data/<rv0_dataset> \\
        --ttg --dt_term 0.05 --n_ics 1000 --out_dir results/gate20
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from standalone_eval_v2 import (
    # physics / constants
    N, nu, g0, m_dry, rho, S_D, C_D,
    T_min_MN, T_max_MN, theta_max, _F_rk4,
    # helpers
    enforce_thrust, tilt_deg, classify_landing, LandingThresholds,
    load_traj, split_by_ic, load_model_from_dir, _state_8,
    rollout_closed_loop, CATEGORIES, CAT_COLORS, CAT_LABELS,
    # plotting reused unchanged (no time axis inside)
    plot_batch_summary, plot_3d_trajectories, _stacked_hist,
    # figure helpers: one file per figure, vector output, no in-figure title
    _save, _cat_legend,
)

# Use the evaluator's write_ic_csv when available, otherwise a local copy.
try:
    from standalone_eval_v2 import write_ic_csv
except ImportError:
    def write_ic_csv(results, experts, out_dir):
        """One row per IC: initial condition -> landing category."""
        with open(Path(out_dir) / "ic_outcomes.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["r_x0", "r_y0", "r_z0", "v_x0", "v_y0", "v_z0",
                        "category"])
            for r, e in zip(results, experts):
                w.writerow([*e["x_traj"][0, :6], r["category"]])
        print(f"  Saved: ic_outcomes.csv  ({len(results)} rows)")

EPS = 1e-6


# =====================================================================
# 0. Pairing with the fuel-optimal (r = v = 0) expert
# =====================================================================

def _ic_key(ic7, dec=2):
    """Hashable key from a 7-vector IC, rounded to absorb float32 storage
    error.
    """
    return tuple(np.round(np.asarray(ic7, dtype=np.float64)[:7], dec))


def load_optimal_experts(opt_dir, dec=2):
    """r = v = 0 expert solves keyed by IC VALUE -> propellant burned [kg]."""
    out, n_bad = {}, 0
    files = sorted(Path(opt_dir).glob("traj_*.npz"))
    for f in files:
        d = np.load(f, allow_pickle=False)
        if int(d["status"]) != 0:
            n_bad += 1
            continue
        ic = d["ic"]
        out[_ic_key(ic, dec)] = dict(
            prop_kg=float(ic[6]) - float(d["m_f"]),
            m0=float(ic[6]), m_f=float(d["m_f"]), t_f=float(d["t_f"]))
    print(f"  optimal (r=v=0) experts: {len(out)} converged of {len(files)} "
          f"files ({n_bad} non-converged skipped)")
    return out


# =====================================================================
# 1. The analytic terminal law
# =====================================================================

def terminal_command(x7, cfg):
    """Return (u_MN, u_raw_MN, diag) for one terminal step."""
    r_h = x7[:2].astype(float)
    v_h = x7[3:5].astype(float)
    r_z = float(x7[2])
    v_z = float(x7[5])
    m = float(x7[6])
    v_norm = float(np.linalg.norm(x7[3:6]))

    h_eff = r_z - cfg["h_c"]
    branch = "arc"

    # ---- vertical ------------------------------------------------------
    if h_eff > cfg["eps_rz"]:
        if v_z < 0.0:
            a_req = (v_z ** 2 - cfg["v_td"] ** 2) / (2.0 * h_eff)
        else:
            a_req = 0.0
            branch = "vz_nonneg"
    else:
        # capture phase: regulate v_z toward -v_td, first-order
        a_req = (-cfg["v_td"] - v_z) / cfg["tau_capture"]
        branch = "capture"
    a_req = float(np.clip(a_req, 0.0, cfg["a_cap"]))

    # ---- horizontal: ZEM/ZEV to r_h = 0, v_h = 0 -----------------------
    if v_z < -cfg["eps_v"] and h_eff > cfg["eps_rz"]:
        t_go = 2.0 * h_eff / abs(v_z)
    else:
        t_go = cfg["t_go_max"]
    t_go = float(np.clip(t_go, cfg["t_go_min"], cfg["t_go_max"]))
    a_hor = -6.0 * r_h / t_go ** 2 - 4.0 * v_h / t_go
    if cfg["no_horizontal"]:
        a_hor = np.zeros(2)

    a_des = np.array([a_hor[0], a_hor[1], a_req + g0])
    u = m * a_des / 1e6

    u_raw = np.asarray(u, dtype=np.float64)
    u = enforce_thrust(u_raw, clip_tilt=True)

    # ---- authority bookkeeping -----------------------------------------
    a_max = T_max_MN * 1e6 * np.cos(theta_max) / m - g0          # pessimistic
    tilt_cmd = np.deg2rad(tilt_deg(u))
    D_z = -0.5 * rho * S_D * C_D * v_norm * v_z                  # up if falling
    a_max_eff = (T_max_MN * 1e6 * np.cos(tilt_cmd) + D_z) / m - g0

    diag = dict(
        branch=branch,
        a_req=a_req,
        a_max=a_max,
        a_max_eff=a_max_eff,
        authority=a_req / max(a_max, EPS),
        authority_eff=a_req / max(a_max_eff, EPS),
        clipped=bool(np.linalg.norm(u - u_raw) > 1e-10),
        mag_raw_MN=float(np.linalg.norm(u_raw)),
        below_Tmin=bool(np.linalg.norm(u_raw) < T_min_MN),
        above_Tmax=bool(np.linalg.norm(u_raw) > T_max_MN),
        tilt_raw_deg=tilt_deg(u_raw),
        tilt_deg=tilt_deg(u),
        a_h_req=float(np.linalg.norm(a_hor)),
        a_h_max=float(T_max_MN * 1e6 * np.sin(theta_max) / m),
    )
    return u, u_raw, diag


# =====================================================================
# 2. Hybrid rollout
# =====================================================================

def rollout_hybrid(model, norms, ic_7, tf, thres, cfg, use_ttg=True,
                   is_mlp=False, device=torch.device("cpu")):
    """Network on its 60-node grid to the gate, then the analytic terminal
    law at cfg['dt_term'] until ground contact.
    """
    dt_nn = tf / N
    x = ic_7.astype(float).copy()

    xs = [x.copy()]
    us, us_raw, dts, phase = [], [], [], []
    tilt_a, tilt_r, mag_a, mag_r = [], [], [], []
    n_clip_mag = n_clip_tilt = 0

    switched = False
    term_reason = ""
    diags = []

    # ---------------- phase 1: network ----------------
    model.eval()
    if not is_mlp:
        buf = np.zeros((1, N, 8), dtype=np.float32)

    with torch.no_grad():
        for k in range(N):
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

            u_raw = np.asarray(
                norms.unnorm_action(a_n.reshape(1, -1)).flatten(),
                dtype=np.float64)
            u = enforce_thrust(u_raw, clip_tilt=True)

            if abs(np.linalg.norm(u_raw) - np.linalg.norm(u)) > 1e-9:
                n_clip_mag += 1
            if tilt_deg(u_raw) > np.degrees(theta_max) + 1e-9:
                n_clip_tilt += 1

            x_next = np.array(_F_rk4(x, u, dt_nn)).flatten()

            us.append(u); us_raw.append(u_raw); dts.append(dt_nn)
            phase.append(0); xs.append(x_next.copy())
            tilt_a.append(tilt_deg(u)); tilt_r.append(tilt_deg(u_raw))
            mag_a.append(np.linalg.norm(u) * 1e3)
            mag_r.append(np.linalg.norm(u_raw) * 1e3)

            if np.any(~np.isfinite(x_next)):
                term_reason = "NaN in network phase"; x = x_next; break
            if x_next[2] < 0.0:
                term_reason = "ground during network phase"; x = x_next; break
            if x_next[6] < m_dry:
                term_reason = "fuel exhausted (network phase)"
                x = x_next; break
            x = x_next
        else:
            # all 60 nodes flown and still airborne -> this IS the gate
            switched = True

    # ---------------- gate / handoff record ----------------
    gate = None
    if switched:
        h_eff = max(x[2] - cfg["h_c"], cfg["eps_rz"])
        a0 = (x[5] ** 2 - cfg["v_td"] ** 2) / (2 * h_eff) if x[5] < 0 else 0.0
        a0 = max(a0, 0.0)
        a_max0 = T_max_MN * 1e6 * np.cos(theta_max) / x[6] - g0
        v_n0 = float(np.linalg.norm(x[3:6]))
        D_z0 = -0.5 * rho * S_D * C_D * v_n0 * float(x[5])
        a_max_eff0 = (T_max_MN * 1e6 + D_z0) / x[6] - g0   # tilt ~ 0 at gate
        gate = dict(
            node=len(us),
            r_z=float(x[2]), v_z=float(x[5]),
            r_h=float(np.linalg.norm(x[:2])),
            v_h=float(np.linalg.norm(x[3:5])),
            v_x=float(x[3]), v_y=float(x[4]),
            m=float(x[6]), mass_margin_kg=float(x[6] - m_dry),
            tilt_cmd_deg=float(tilt_r[-1]) if tilt_r else float("nan"),
            a_req=float(a0),
            authority=float(a0 / max(a_max0, EPS)),
            authority_eff=float(a0 / max(a_max_eff0, EPS)),
        )

        # ---------------- phase 2: analytic law ----------------
        dt_t = cfg["dt_term"]
        n_max = int(np.ceil(cfg["t_term_max"] / dt_t))
        for _ in range(n_max):
            u, u_raw, d = terminal_command(x, cfg)
            diags.append(d)
            x_next = np.array(_F_rk4(x, u, dt_t)).flatten()

            us.append(u); us_raw.append(u_raw); dts.append(dt_t)
            phase.append(1); xs.append(x_next.copy())
            tilt_a.append(d["tilt_deg"]); tilt_r.append(d["tilt_raw_deg"])
            mag_a.append(np.linalg.norm(u) * 1e3)
            mag_r.append(d["mag_raw_MN"] * 1e3)
            if d["clipped"]:
                n_clip_mag += 1
            if d["tilt_raw_deg"] > np.degrees(theta_max) + 1e-9:
                n_clip_tilt += 1

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
    u_traj_raw = np.array(us_raw) if us_raw else np.zeros((0, nu))
    dts = np.array(dts)
    phase = np.array(phase, dtype=int)
    t_state = np.concatenate([[0.0], np.cumsum(dts)]) if len(dts) else \
        np.zeros(1)
    t_ctrl = t_state[:-1] if len(dts) else np.zeros(0)

    # ---------------- touchdown (variable dt: bisect the LAST step) ------
    td_state = td_time = None
    td_type = "none"
    if len(x_traj) >= 2 and x_traj[-1, 2] < 0.0 <= x_traj[-2, 2]:
        lo, hi = 0.0, float(dts[-1])
        x_lo = x_traj[-2]
        for _ in range(thres.bisect_max_iter):
            mid = 0.5 * (lo + hi)
            xm = np.array(_F_rk4(x_lo, u_traj[-1], mid)).flatten()
            if xm[2] > 0:
                lo = mid
            else:
                hi = mid
        tau = 0.5 * (lo + hi)
        td_state = np.array(_F_rk4(x_lo, u_traj[-1], tau)).flatten()
        td_time = float(t_state[-2] + tau)
        td_type = "bisection"
    elif 0.0 <= x_traj[-1, 2] <= thres.grazing_alt_m:
        td_state = x_traj[-1].copy()
        td_time = float(t_state[-1])
        td_type = "grazing"

    if td_state is not None:
        category, details = classify_landing(td_state, thres)
    elif "fuel" in term_reason:
        category, details = "fuel_exhausted", {}
    elif "NaN" in term_reason:
        category, details = "diverged", {}
    else:
        category, details = "airborne", {}

    n_term = int(phase.sum()) if len(phase) else 0
    t_terminal = float(dts[phase == 1].sum()) if n_term else 0.0

    def _agg(key, fn, default=float("nan")):
        v = [d[key] for d in diags]
        return float(fn(v)) if v else default

    return dict(
        # --- keys read by the standalone_eval_v2 plotting functions ---
        x_traj=x_traj, u_traj=u_traj, u_traj_raw=u_traj_raw,
        T_mag_kN=np.array(mag_a), T_mag_raw_kN=np.array(mag_r),
        tilt_deg=np.array(tilt_a), tilt_raw_deg=np.array(tilt_r),
        completed=("NaN" not in term_reason),
        reached_ground=td_state is not None,
        term_reason=term_reason, steps_done=len(us),
        n_clip_mag=n_clip_mag, n_clip_tilt=n_clip_tilt,
        td_state=td_state, td_time=td_time, td_type=td_type,
        category=category, details=details, tf=tf,
        # --- additions ---
        t_state=t_state, t_ctrl=t_ctrl, dts=dts, phase=phase,
        switched=switched, gate=gate, handoff=gate,   # 'handoff' is an alias
        n_network_steps=int((phase == 0).sum()) if len(phase) else 0,
        n_terminal_steps=n_term, t_terminal=t_terminal,
        t_handoff=float(t_state[gate["node"]]) if gate else None,
        m0=float(x_traj[0, 6]),
        m_handoff=float(gate["m"]) if gate else None,
        m_final=float(td_state[6]) if td_state is not None
        else float(x_traj[-1, 6]),
        n_below_Tmin=sum(d["below_Tmin"] for d in diags),
        n_above_Tmax=sum(d["above_Tmax"] for d in diags),
        min_mag_raw_MN=_agg("mag_raw_MN", np.min),
        max_authority=_agg("authority", np.max),
        max_authority_eff=_agg("authority_eff", np.max),
        frac_h_saturated=float(np.mean(
            [d["a_h_req"] > d["a_h_max"] for d in diags])) if diags
        else float("nan"),
        n_hover_steps=sum(1 for d in diags
                          if d["branch"] == "vz_nonneg"),
        n_capture_steps=sum(1 for d in diags if d["branch"] == "capture"),
    )


# =====================================================================
# 3. Plotting — time-correct versions of the evaluator figures that
# =====================================================================

def plot_single_comparison_hybrid(expert, r, out_dir, tag=""):
    """Expert-vs-hybrid comparison for one IC, written as six independent
    figures:
    """
    tf = expert["t_f"]
    sfx = f"_{tag}" if tag else ""
    t_s_exp = np.linspace(0, tf, expert["x_traj"].shape[0])
    t_c_exp = np.linspace(0, tf * (N - 1) / N, N)
    x_exp = expert["x_traj"][:, :7]
    u_exp = expert["u_traj"]

    t_s = r["t_state"]
    t_c = r["t_ctrl"]
    t_h = r["t_handoff"]

    s_exp = dict(color="tab:blue", lw=2, label="Expert (to gate)")
    s_pol = dict(color="tab:red", lw=2, ls="--", label="Hybrid")

    def vline(ax):
        if t_h is not None:
            ax.axvline(t_h, color="k", ls=":", lw=1.2, label="Handoff (gate)")

    cat = r["category"]
    print(f"  [figure set{sfx}] {CAT_LABELS.get(cat, cat).upper()}, "
          f"t_f = {tf:.1f} s + {r['t_terminal']:.2f} s terminal")

    # --- Altitude ---
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(t_s_exp, x_exp[:, 2], **s_exp)
    ax.plot(t_s, r["x_traj"][:, 2], **s_pol)
    ax.axhline(0, color="k", lw=0.5)
    vline(ax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Altitude $r_z$ [m]")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    _save(fig, out_dir, f"altitude{sfx}")

    # --- Vertical velocity ---
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(t_s_exp, x_exp[:, 5], **s_exp)
    ax.plot(t_s, r["x_traj"][:, 5], **s_pol)
    ax.axhline(0, color="k", lw=0.5)
    vline(ax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("$v_z$ [m/s]")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    _save(fig, out_dir, f"vz{sfx}")

    # --- Horizontal position norm ---
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(t_s_exp, np.linalg.norm(x_exp[:, :2], axis=1), **s_exp)
    ax.plot(t_s, np.linalg.norm(r["x_traj"][:, :2], axis=1), **s_pol)
    ax.axhline(0, color="k", lw=0.5)
    vline(ax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\|r_h\|$ [m]")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    _save(fig, out_dir, f"rh{sfx}")

    # --- Thrust magnitude ---
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.step(t_c_exp, np.linalg.norm(u_exp, axis=1) * 1e3, where="post",
            **s_exp)
    if len(t_c):
        ax.step(t_c, r["T_mag_kN"], where="post", **s_pol)
        ax.step(t_c, r["T_mag_raw_kN"], where="post", color="tab:red",
                alpha=0.35, lw=1, label="Hybrid raw (pre-clip)")
    ax.axhline(T_min_MN * 1e3, color="gray", ls="--", lw=1,
               label=f"$T_{{min}}$={T_min_MN*1e3:.0f} kN")
    ax.axhline(T_max_MN * 1e3, color="gray", ls="-.", lw=1,
               label=f"$T_{{max}}$={T_max_MN*1e3:.0f} kN")
    vline(ax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Thrust [kN]")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.grid(alpha=0.3)
    _save(fig, out_dir, f"thrust{sfx}")

    # --- Tilt angle: enforced AND raw ---
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.step(t_c_exp, [tilt_deg(u_exp[k]) for k in range(N)],
            where="post", **s_exp)
    if len(t_c):
        ax.step(t_c, r["tilt_deg"], where="post", **s_pol)
        ax.step(t_c, r["tilt_raw_deg"], where="post", color="tab:red",
                alpha=0.35, lw=1, label="Hybrid raw (pre-clip)")
    ax.axhline(np.degrees(theta_max), color="red", ls="--", lw=1,
               label=rf"$\theta_{{max}}$={np.degrees(theta_max):.0f}$\degree$")
    vline(ax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Tilt [deg]")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.3)
    _save(fig, out_dir, f"tilt{sfx}")

    # --- 2-D trajectory, East-Up, with near-pad inset ---
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.plot(x_exp[:, 0], x_exp[:, 2], **s_exp)
    ax.plot(r["x_traj"][:, 0], r["x_traj"][:, 2], **s_pol)
    ax.plot(x_exp[0, 0], x_exp[0, 2], "go", ms=10, label="Start")
    ax.plot(0, 0, "k*", ms=15, label="Target")
    if r["gate"]:
        n = r["gate"]["node"]
        ax.plot(r["x_traj"][n, 0], r["x_traj"][n, 2], "ks", ms=8,
                mfc="none", mew=1.5, label="Handoff (gate)")
    xf = r["x_traj"][-1]
    ax.plot(xf[0], xf[2], "rx", ms=12, mew=2, label="End")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("East $r_x$ [m]")
    ax.set_ylabel("Up $r_z$ [m]")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    if r["gate"]:
        gz = r["gate"]["r_z"]
        span = max(20.0, 2.0 * r["gate"]["r_h"], 1.5 * gz)
        axins = ax.inset_axes([0.1, 0.08, 0.40, 0.40])
        axins.plot(x_exp[:, 0], x_exp[:, 2], color="tab:blue", lw=1.5)
        axins.plot(r["x_traj"][:, 0], r["x_traj"][:, 2], color="tab:red",
                   lw=1.5, ls="--")
        n = r["gate"]["node"]
        axins.plot(r["x_traj"][n, 0], r["x_traj"][n, 2], "ks", ms=6,
                   mfc="none", mew=1.4)
        axins.plot(0, 0, "k*", ms=10)
        axins.axhline(0, color="k", lw=0.5)
        axins.set_xlim(-span, span)
        axins.set_ylim(-0.05 * span, 1.6 * gz + 1)
        axins.xaxis.set_major_locator(MaxNLocator(nbins=4))
        axins.yaxis.set_major_locator(MaxNLocator(nbins=4))
        axins.tick_params(labelsize=6, pad=1, length=2)
        axins.grid(alpha=0.20, lw=0.5)
        axins.patch.set_alpha(0.92)

    _save(fig, out_dir, f"traj_eastup{sfx}")


def plot_gate_arrival(results, out_dir, h_gate, v_gate, thres):
    """State AT THE GATE (i.e. at handoff), stacked by the FINAL outcome,
    written as six independent figures:
    """
    out_dir = Path(out_dir)
    sw = [r for r in results if r["gate"] is not None]
    if not sw:
        print("  (no handoffs — gate arrival figures skipped)")
        return

    print(f"  [gate arrival figures] {len(sw)} ICs reached the gate; "
          f"nominal gate r_z={h_gate:g} m, v_z=-{v_gate:g} m/s")

    def by_cat(fn):
        out = {c: [] for c in CATEGORIES}
        for r in sw:
            out[r["category"]].append(fn(r))
        return {c: np.asarray(v, dtype=float) for c, v in out.items()}

    g = lambda k: by_cat(lambda r: r["gate"][k])
    figsize = (6.0, 4.2)

    # --- Gate altitude ---
    fig, ax = plt.subplots(figsize=figsize)
    c = _stacked_hist(ax, g("r_z"), clip_lo_pct=1.0,
                      xlabel=r"$r_z$ at gate [m]",
                      vlines=[(h_gate, "--", "nominal")])
    _cat_legend(ax, c)
    _save(fig, out_dir, "gate_rz")

    # --- Gate vertical velocity (signed) ---
    fig, ax = plt.subplots(figsize=figsize)
    c = _stacked_hist(ax, g("v_z"), clip_lo_pct=1.0,
                      xlabel=r"$v_z$ at gate [m/s]",
                      vlines=[(-v_gate, "--", "nominal"), (0.0, "-", "0")])
    _cat_legend(ax, c, loc="upper left")
    _save(fig, out_dir, "gate_vz")

    # --- Gate horizontal position ---
    fig, ax = plt.subplots(figsize=figsize)
    c = _stacked_hist(ax, g("r_h"),
                      xlabel=r"$\|r_h\|$ at gate [m]",
                      vlines=[(thres.soft_rh_m, "--", "soft limit")])
    _cat_legend(ax, c)
    _save(fig, out_dir, "gate_rh")

    # --- Gate horizontal velocity ---
    fig, ax = plt.subplots(figsize=figsize)
    c = _stacked_hist(ax, g("v_h"), xlabel=r"$\|v_h\|$ at gate [m/s]")
    _cat_legend(ax, c)
    _save(fig, out_dir, "gate_vh")

    # --- Terminal-law authority at the gate ---
    fig, ax = plt.subplots(figsize=figsize)
    c = _stacked_hist(ax, g("authority"),
                      xlabel=r"$a_{req}/a_{max}$ at gate",
                      vlines=[(1.0, "--", "infeasible")])
    _cat_legend(ax, c)
    _save(fig, out_dir, "gate_authority")

    # --- Propellant margin at the gate ---
    fig, ax = plt.subplots(figsize=figsize)
    c = _stacked_hist(ax,
                      by_cat(lambda r: r["gate"]["mass_margin_kg"] / 1e3),
                      xlabel=r"$m - m_{dry}$ at gate [t]")
    _cat_legend(ax, c)
    _save(fig, out_dir, "gate_mass_margin")


def plot_bundles_hybrid(results, experts, out_dir, max_plot=100):
    """Overlay bundles on true time axes, one figure each: altitude, v_z,
    thrust.
    """
    out_dir = Path(out_dir)

    fig_alt, ax_alt = plt.subplots(figsize=(6.5, 4.4))
    fig_vz, ax_vz = plt.subplots(figsize=(6.5, 4.4))
    fig_T, ax_T = plt.subplots(figsize=(6.5, 4.4))

    for r, e in list(zip(results, experts))[:max_plot]:
        col = CAT_COLORS.get(r["category"], "tab:gray")
        t_s_ex = np.linspace(0, e["t_f"], e["x_traj"].shape[0])
        t_c_ex = np.linspace(0, e["t_f"] * (N - 1) / N, N)
        ax_alt.plot(r["t_state"], r["x_traj"][:, 2], color=col,
                    alpha=0.25, lw=0.7)
        ax_alt.plot(t_s_ex, e["x_traj"][:, 2], color="tab:blue",
                    alpha=0.08, lw=0.5)
        ax_vz.plot(r["t_state"], r["x_traj"][:, 5], color=col,
                   alpha=0.25, lw=0.7)
        ax_vz.plot(t_s_ex, e["x_traj"][:, 5], color="tab:blue",
                   alpha=0.08, lw=0.5)
        if len(r["t_ctrl"]):
            ax_T.plot(r["t_ctrl"], r["T_mag_kN"], color=col,
                      alpha=0.25, lw=0.7)
        ax_T.plot(t_c_ex, np.linalg.norm(e["u_traj"], axis=1) * 1e3,
                  color="tab:blue", alpha=0.08, lw=0.5)

    proxies = [Line2D([0], [0], color="tab:blue", lw=2, label="Expert")]
    proxies += [Line2D([0], [0], color=CAT_COLORS[c], lw=2,
                       label=CAT_LABELS[c]) for c in CATEGORIES]

    ax_alt.axhline(0, color="k", lw=0.5)
    ax_alt.set_xlabel("Time [s]")
    ax_alt.set_ylabel(r"Altitude $r_z$ [m]")
    ax_alt.grid(alpha=0.3)
    ax_alt.legend(handles=proxies, fontsize=8, ncol=2, frameon=False)
    _save(fig_alt, out_dir, "bundles_altitude")

    ax_vz.axhline(0, color="k", lw=0.5)
    ax_vz.set_xlabel("Time [s]")
    ax_vz.set_ylabel("$v_z$ [m/s]")
    ax_vz.grid(alpha=0.3)
    ax_vz.legend(handles=proxies, fontsize=8, ncol=2, frameon=False)
    _save(fig_vz, out_dir, "bundles_vz")

    ax_T.axhline(T_min_MN * 1e3, color="gray", ls="--", lw=1)
    ax_T.axhline(T_max_MN * 1e3, color="gray", ls="-.", lw=1)
    ax_T.set_xlabel("Time [s]")
    ax_T.set_ylabel("Thrust [kN]")
    ax_T.grid(alpha=0.3)
    ax_T.legend(handles=proxies, fontsize=8, ncol=2, frameon=False)
    _save(fig_T, out_dir, "bundles_thrust")


# =====================================================================
# 4. Statistics helpers
# =====================================================================

def q(v):
    """median, q25, q75 over finite entries."""
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return (float("nan"),) * 3
    return (float(np.median(v)), float(np.percentile(v, 25)),
            float(np.percentile(v, 75)))


def fuel_block(pairs, opt=None):
    """Propellant statistics for the hybrid rollouts.

    pairs : list of (hybrid_result, gate_expert_dict)
    opt   : dict from load_optimal_experts(), or None
    """
    if not pairs:
        return None
    prop_tot = np.array([r["m0"] - r["m_final"] for r, _ in pairs])
    prop_e = np.array([float(e["ic"][6]) - float(e["m_f"]) for _, e in pairs])
    out = dict(n=len(pairs),
               expert_to_gate_kg=q(prop_e),
               total_hybrid_kg=q(prop_tot),
               total_vs_expert_to_gate_kg=q(prop_tot - prop_e))

    # ---- subset that reached the gate: network / terminal split --------
    idx = [i for i, (r, _) in enumerate(pairs) if r["m_handoff"] is not None]
    if idx:
        nn = np.array([pairs[i][0]["m0"] - pairs[i][0]["m_handoff"]
                       for i in idx])
        te = np.array([pairs[i][0]["m_handoff"] - pairs[i][0]["m_final"]
                       for i in idx])
        pe = prop_e[idx]
        out.update(n_reached_gate=len(idx),
                   network_phase_kg=q(nn),
                   terminal_phase_kg=q(te),
                   network_excess_vs_expert_kg=q(nn - pe),
                   network_excess_vs_expert_pct=q(100 * (nn - pe) / pe))

    # ---- against the TRUE optimum (r=v=0 expert, same ICs) -------------
    if opt:
        oi, op = [], []
        for i, (_, e) in enumerate(pairs):
            k = _ic_key(e["ic"])
            if k in opt:
                oi.append(i)
                op.append(opt[k]["prop_kg"])
        if oi:
            op = np.array(op)
            pt = prop_tot[oi]
            out.update(n_paired_optimal=len(oi),
                       optimal_expert_kg=q(op),
                       excess_vs_optimal_kg=q(pt - op),
                       excess_vs_optimal_pct=q(100 * (pt - op) / op))
            # Attribution, on ICs with both an optimal pair and a gate arrival
            j = [i for i in oi if pairs[i][0]["m_handoff"] is not None]
            if j:
                tot = np.array([pairs[i][0]["m0"] - pairs[i][0]["m_final"]
                                for i in j])
                net = np.array([pairs[i][0]["m0"] - pairs[i][0]["m_handoff"]
                                for i in j])
                gex = np.array([float(pairs[i][1]["ic"][6])
                                - float(pairs[i][1]["m_f"]) for i in j])
                oex = np.array([opt[_ic_key(pairs[i][1]["ic"])]["prop_kg"]
                                for i in j])
                out.update(
                    n_attributed=len(j),
                    excess_from_network_imitation_kg=q(net - gex),
                    excess_from_gate_and_patch_kg=q((tot - oex) - (net - gex)))
    return out


def print_block(name, blk):
    if not blk:
        return
    print(f"\n--- {name} (median [q25, q75]) ---")
    for k, v in blk.items():
        if isinstance(v, tuple):
            print(f"  {k:34s}: {v[0]:10.3f}  [{v[1]:.3f}, {v[2]:.3f}]")
        else:
            print(f"  {k:34s}: {v}")


# =====================================================================
# 5. Main
# =====================================================================

def main():
    p = argparse.ArgumentParser(
        description="Gate-policy hybrid evaluation (network -> analytic law)")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--data_dir", required=True,
                   help="GATE expert dataset (terminates at the gate)")
    p.add_argument("--opt_data_dir", default=None,
                   help="r=v=0 (fuel-optimal landing) expert solves. Enables "
                        "the total-propellant comparison against the true "
                        "optimum. Paired by IC VALUE, so the two datasets need "
                        "not have the same file count or ordering.")
    p.add_argument("--out_dir", default="results/gate_patch")
    p.add_argument("--n_ics", type=int, default=300)
    p.add_argument("--seed", type=int, default=42,
                   help="split seed — MUST match training")
    p.add_argument("--ttg", action="store_true", default=False)
    p.add_argument("--model_type", default="auto")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--norms", default=None)

    p.add_argument("--h_gate", type=float, default=20.0)
    p.add_argument("--v_gate", type=float, default=16.0)

    # law
    p.add_argument("--dt_term", type=float, default=0.05)
    p.add_argument("--h_c", type=float, default=0.0,
                   help="capture altitude [m]; 0 = pure arc to the pad")
    p.add_argument("--v_td", type=float, default=0.0,
                   help="target descent speed at/below h_c [m/s]")
    p.add_argument("--tau_capture", type=float, default=0.3)
    p.add_argument("--t_term_max", type=float, default=60.0)
    p.add_argument("--no_horizontal", action="store_true", default=False)

    # baseline / figures
    p.add_argument("--skip_baseline", action="store_true", default=False,
                   help="baseline = pure network to node 60. For a gate "
                        "policy it is airborne by construction; it is a "
                        "sanity check, not a competitor.")
    p.add_argument("--max_3d", type=int, default=0)
    p.add_argument("--plot_bundles", action="store_true", default=False)
    p.add_argument("--no_plots", action="store_true", default=False)
    p.add_argument("--quiet", action="store_true", default=False)
    a = p.parse_args()

    cfg = dict(dt_term=a.dt_term, h_c=a.h_c, v_td=a.v_td,
               tau_capture=a.tau_capture, t_term_max=a.t_term_max,
               no_horizontal=a.no_horizontal,
               eps_rz=0.01, eps_v=0.01,
               t_go_min=0.2, t_go_max=60.0, a_cap=50.0)

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    thres = LandingThresholds()

    model, norms, is_mlp = load_model_from_dir(
        a.run_dir, device, model_type=a.model_type,
        checkpoint_file=a.checkpoint, norm_file=a.norms)

    opt = load_optimal_experts(a.opt_data_dir) if a.opt_data_dir else None
    if opt is None:
        print("  (no --opt_data_dir: the comparison against the true "
              "fuel-optimal landing is skipped)")

    files = sorted(Path(a.data_dir).glob("traj_*.npz"))
    _, _, test_files = split_by_ic(files, seed=a.seed)
    print(f"\ntest split {len(test_files)} | running up to {a.n_ics}")
    print(f"handoff: after node {N}   dt_term={a.dt_term}  "
          f"h_c={a.h_c}  v_td={a.v_td}")
    print(f"nominal gate (from flags): r_z={a.h_gate:g} m, "
          f"v_z=-{a.v_gate:g} m/s   — checked against the data below")

    hyb, base, experts = [], [], []
    n = 0
    if not a.quiet:
        print(f"\n{'#':>5s}  {'Cat':>15s}  {'gate_rz':>8s}  {'gate_vz':>8s}  "
              f"{'td_rh':>7s}  {'td_vz':>7s}  {'t_term':>6s}  Notes")
        print("-" * 88)

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
        if not a.skip_baseline:
            base.append(rollout_closed_loop(
                model, norms, ic, tf, thres, is_mlp=is_mlp,
                use_ttg=a.ttg, clip_tilt=True, device=device))
        if not a.quiet:
            gt = r["gate"] or {}
            d = r["details"]
            print(f"{n-1:5d}  {r['category']:>15s}  "
                  f"{gt.get('r_z', float('nan')):8.2f}  "
                  f"{gt.get('v_z', float('nan')):8.2f}  "
                  f"{d.get('r_h', float('nan')):7.2f}  "
                  f"{d.get('vz_signed', float('nan')):7.2f}  "
                  f"{r['t_terminal']:6.2f}  {r['term_reason'][:28]}")

    if n == 0:
        print("No converged expert trajectories in the test split — nothing "
              "to evaluate. Check --data_dir and --seed.")
        return 1

    # ---------------- nominal gate READ FROM THE DATA ----------------
    h_det = float(np.median([e["x_traj"][-1, 2] for e in experts]))
    v_det = float(-np.median([e["x_traj"][-1, 5] for e in experts]))
    if abs(h_det - a.h_gate) > 0.5 or abs(v_det - a.v_gate) > 0.5:
        print(f"\n  WARNING: --h_gate/--v_gate = {a.h_gate:g}/{a.v_gate:g}, "
              f"but the expert data terminates at {h_det:.2f} m / "
              f"{v_det:.2f} m/s.\n  Using the DATA values for all gate-error "
              f"reporting and plot guide lines.")
        a.h_gate, a.v_gate = h_det, v_det
    print(f"\n  gate from data: r_z = {h_det:.2f} m, v_z = -{v_det:.2f} m/s"
          f"  -> nominal a_req = {v_det**2/(2*max(h_det,EPS)):.3f} m/s^2")

    # ---------------- category table ----------------
    def counts(rs):
        return {c: sum(1 for r in rs if r["category"] == c)
                for c in CATEGORIES}

    ch = counts(hyb)
    cb = counts(base) if base else None
    print(f"\n{'category':>16s} {'baseline':>10s} {'hybrid':>10s} "
          f"{'delta':>8s}")
    for c in CATEGORIES:
        b = cb[c] if cb else 0
        print(f"{c:>16s} {b:>10d} {ch[c]:>10d} {ch[c]-b:>+8d}")
    print(f"{'n':>16s} {n:>10d} {n:>10d}")
    print("\n  NOTE: for a gate policy the baseline is airborne by "
          "construction\n  (the network stops at the gate). Sanity check "
          "only — do not quote it.")

    n_no_handoff = sum(1 for r in hyb if r["gate"] is None)
    if n_no_handoff:
        print(f"\n  {n_no_handoff}/{n} ICs never reached the gate (network "
              f"phase ended early — see term_reason in gate_arrivals.csv)")

    # ---------------- enforcement / law health ----------------
    tmin_hits = sum(r["n_below_Tmin"] for r in hyb)
    tmax_hits = sum(r["n_above_Tmax"] for r in hyb)
    tsteps = sum(r["n_terminal_steps"] for r in hyb)
    # guard: all-NaN reduction when no IC reached the gate
    _mr = [r["min_mag_raw_MN"] for r in hyb
           if np.isfinite(r["min_mag_raw_MN"])]
    min_raw = float(np.min(_mr)) if _mr else float("nan")
    print(f"\n  terminal steps: {tsteps}")
    print(f"  T_min floor: {tmin_hits}/{max(tsteps,1)} "
          f"({100*tmin_hits/max(tsteps,1):.2f}%), min raw = "
          f"{min_raw*1e3:.0f} kN (T_min = {T_min_MN*1e3:.0f} kN)")
    print(f"  T_max ceiling: {tmax_hits}/{max(tsteps,1)} "
          f"({100*tmax_hits/max(tsteps,1):.2f}%)")
    fh = [r["frac_h_saturated"] for r in hyb if r["switched"]]
    if fh:
        print(f"  horizontal saturated: median "
              f"{np.nanmedian(fh):.2f} of terminal steps")
    hov = sum(r["n_hover_steps"] for r in hyb)
    n_hov_ic = sum(1 for r in hyb if r["n_hover_steps"] > 0)
    print(f"  steps with v_z >= 0 (should be 0 — the gate pins v_z < 0): "
          f"{hov} on {n_hov_ic} ICs")

    # ---------------- gate arrival ----------------
    sw = [r for r in hyb if r["gate"] is not None]
    summary = dict(
        n=n, n_switched=len(sw),
        config={k: cfg[k] for k in sorted(cfg)},
        nominal_gate=dict(h_gate=a.h_gate, v_gate=a.v_gate,
                          source="expert data (median terminal state)",
                          a_req_nominal=a.v_gate ** 2 / (2 * max(a.h_gate,
                                                                 EPS))),
        categories_hybrid=ch, categories_baseline=cb,
        soft_rate_hybrid=ch["soft"] / n,
        soft_rate_baseline=(cb["soft"] / n) if cb else None,
        baseline_note="pure network to node 60; airborne by construction "
                      "for a gate policy — sanity check, not a competitor",
        terminal_law_health=dict(
            terminal_steps=tsteps, below_Tmin=tmin_hits,
            above_Tmax=tmax_hits, min_raw_kN=float(min_raw * 1e3),
            vz_nonneg_steps=hov, vz_nonneg_ics=n_hov_ic,
            frac_h_saturated_median=float(np.nanmedian(fh)) if fh else None,
        ),
    )

    if sw:
        gate_blk = {
            "r_z_m": q([r["gate"]["r_z"] for r in sw]),
            "v_z_ms": q([r["gate"]["v_z"] for r in sw]),
            "r_h_m": q([r["gate"]["r_h"] for r in sw]),
            "v_h_ms": q([r["gate"]["v_h"] for r in sw]),
            "tilt_cmd_deg": q([r["gate"]["tilt_cmd_deg"] for r in sw]),
            "mass_margin_kg": q([r["gate"]["mass_margin_kg"] for r in sw]),
            "a_req_ms2": q([r["gate"]["a_req"] for r in sw]),
            "authority": q([r["gate"]["authority"] for r in sw]),
            "authority_eff": q([r["gate"]["authority_eff"] for r in sw]),
            "node": q([r["gate"]["node"] for r in sw]),
            "frac_infeasible_at_gate": float(np.mean(
                [r["gate"]["authority"] > 1.0 for r in sw])),
        }
        err_nom = {
            "d_r_z_m": q([r["gate"]["r_z"] - a.h_gate for r in sw]),
            "d_v_z_ms": q([r["gate"]["v_z"] + a.v_gate for r in sw]),
            "r_h_m": q([r["gate"]["r_h"] for r in sw]),
            "v_h_ms": q([r["gate"]["v_h"] for r in sw]),
        }
        pairs = [(r, e) for r, e in zip(hyb, experts) if r["gate"] is not None]
        err_exp = {
            "d_r_z_m": q([r["gate"]["r_z"] - float(e["x_traj"][-1, 2])
                          for r, e in pairs]),
            "d_v_z_ms": q([r["gate"]["v_z"] - float(e["x_traj"][-1, 5])
                           for r, e in pairs]),
            "d_r_h_m": q([r["gate"]["r_h"]
                          - float(np.linalg.norm(e["x_traj"][-1, :2]))
                          for r, e in pairs]),
            "d_v_h_ms": q([r["gate"]["v_h"]
                           - float(np.linalg.norm(e["x_traj"][-1, 3:5]))
                           for r, e in pairs]),
            "d_m_kg": q([r["gate"]["m"] - float(e["x_traj"][-1, 6])
                         for r, e in pairs]),
        }
        summary["gate"] = gate_blk
        summary["gate_error_vs_nominal"] = err_nom
        summary["gate_error_vs_expert_terminal"] = err_exp
        summary["terminal_phase_duration_s"] = q(
            [r["t_terminal"] for r in sw])
        summary["max_authority_during_terminal"] = q(
            [r["max_authority"] for r in sw])
        summary["max_authority_eff_during_terminal"] = q(
            [r["max_authority_eff"] for r in sw])

        print_block("gate arrival", gate_blk)
        print_block("gate error vs nominal", err_nom)
        print_block("gate error vs expert terminal state", err_exp)
        if gate_blk["frac_infeasible_at_gate"] > 0.02:
            print(f"\n  WARNING: {gate_blk['frac_infeasible_at_gate']:.1%} of "
                  f"gates need more deceleration than T_max can give. For a "
                  f"gate policy this means the network is not delivering the "
                  f"gate condition — check gate_arrival.png before blaming "
                  f"the law.")

    # ---------------- fuel ----------------
    soft = [(r, e) for r, e in zip(hyb, experts) if r["category"] == "soft"]
    landed = [(r, e) for r, e in zip(hyb, experts)
              if r["category"] in ("soft", "hard", "crash")]
    summary["fuel_soft_only"] = fuel_block(soft, opt)
    summary["fuel_all_landed"] = fuel_block(landed, opt)
    summary["fuel_note"] = (
        "expert_to_gate_kg is the GATE expert's propellant (flight to the gate "
        "only), so network_phase_kg vs expert_to_gate_kg is the like-for-like "
        "imitation comparison and terminal_phase_kg is the added cost of the "
        "analytic patch. optimal_expert_kg is the r=v=0 fuel-optimal expert on "
        "the SAME IC (paired by IC value); excess_vs_optimal_kg is the "
        "headline. The attribution splits that excess into the network's "
        "imitation error and the cost of the gate architecture itself.")
    print_block("fuel — SOFT landings only (primary)",
                summary["fuel_soft_only"])
    print_block("fuel — all landed ICs (secondary)",
                summary["fuel_all_landed"])

    if opt:
        fs = summary["fuel_soft_only"] or {}
        if "n_paired_optimal" in fs and soft:
            frac = fs["n_paired_optimal"] / len(soft)
            if frac < 0.9:
                print(f"\n  WARNING: only {fs['n_paired_optimal']}/{len(soft)} "
                      f"({100*frac:.0f}%) of soft landings paired with an "
                      f"optimal-expert solve. The two datasets may not share "
                      f"ICs; the comparison is valid only on the overlap.")
        elif "n_paired_optimal" not in fs:
            print("\n  WARNING: no ICs paired with the optimal-expert "
                  "dataset. Check that --opt_data_dir covers the same IC "
                  "sweep as --data_dir.")

    # ---------------- failures.csv ----------------
    with open(out / "failures.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cat", "g_r_z", "g_v_z", "g_r_h", "g_v_h", "g_auth",
                    "g_auth_eff", "g_mass_margin", "frac_h_sat",
                    "td_r_h", "td_vz", "td_vh", "fail_vz", "fail_vh",
                    "t_term", "term_reason"])
        for r in hyb:
            if r["category"] not in ("crash", "hard"):
                continue
            g, d = r["gate"] or {}, r["details"]
            w.writerow([r["category"], g.get("r_z"), g.get("v_z"),
                        g.get("r_h"), g.get("v_h"), g.get("authority"),
                        g.get("authority_eff"), g.get("mass_margin_kg"),
                        r["frac_h_saturated"], d.get("r_h"),
                        d.get("vz_signed"), d.get("vh"),
                        abs(d.get("vz_signed", 0.0)) > thres.hard_vz_ms,
                        d.get("vh", 0.0) > thres.hard_vh_norm_ms,
                        r["t_terminal"], r["term_reason"]])
    print("\n  Saved: failures.csv")

    # ---------------- gate_arrivals.csv (ALL ICs, uncensored) ----------
    with open(out / "gate_arrivals.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "category", "reached_gate",
                    "g_r_z", "g_v_z", "g_r_h", "g_v_h",
                    "g_m", "mass_margin_kg", "g_authority",
                    "d_r_z_nominal", "d_v_z_nominal",
                    "e_r_z", "e_v_z",
                    "d_r_z_expert", "d_v_z_expert",
                    "terminal_phase_kg", "t_terminal",
                    "opt_expert_kg", "total_hybrid_kg",
                    "excess_vs_optimal_kg", "term_reason"])
        for i, (r, e) in enumerate(zip(hyb, experts)):
            g = r["gate"]
            if g is None:
                w.writerow([i, r["category"], 0] + [""] * 18
                           + [r["term_reason"]])
                continue
            xe = e["x_traj"][-1]
            o = opt.get(_ic_key(e["ic"]), {}).get("prop_kg") if opt else None
            tot = r["m0"] - r["m_final"]
            w.writerow([
                i, r["category"], 1,
                g["r_z"], g["v_z"], g["r_h"], g["v_h"],
                g["m"], g["mass_margin_kg"], g["authority"],
                g["r_z"] - a.h_gate, g["v_z"] + a.v_gate,
                float(xe[2]), float(xe[5]),
                g["r_z"] - float(xe[2]), g["v_z"] - float(xe[5]),
                r["m_handoff"] - r["m_final"], r["t_terminal"],
                o if o is not None else "", tot,
                (tot - o) if o is not None else "",
                r["term_reason"],
            ])
    n_gate = sum(1 for r in hyb if r["gate"] is not None)
    print(f"  Saved: gate_arrivals.csv  ({n_gate}/{n} reached the gate)")

    # ---------------- figures ----------------
    if not a.no_plots:
        print("\n--- figures ---")
        plot_batch_summary(hyb, out, thres)
        plot_3d_trajectories(hyb, out, max_plot=a.max_3d)
        plot_gate_arrival(hyb, out, a.h_gate, a.v_gate, thres)
        if a.plot_bundles:
            plot_bundles_hybrid(hyb, experts, out,
                                max_plot=min(len(hyb), 100))
        seen = set()
        for i, r in enumerate(hyb):
            if r["category"] in seen:
                continue
            seen.add(r["category"])
            plot_single_comparison_hybrid(
                experts[i], r, out, tag=f"example_{r['category']}_{i:04d}")
        write_ic_csv(hyb, experts, out)

    with open(out / "gate_patch.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out/'gate_patch.json'}")

    # ---------------- headline ----------------
    print(f"\n{'='*68}\n  HEADLINE\n{'='*68}")
    print(f"  soft rate (hybrid)       : {ch['soft']}/{n} = "
          f"{100*ch['soft']/n:.1f}%")
    fs = summary.get("fuel_soft_only") or {}
    if "terminal_phase_kg" in fs:
        print(f"  terminal-phase burn      : "
              f"{fs['terminal_phase_kg'][0]:.0f} kg   "
              f"(gate_mc predicted ~856 for 20/-16)")
    if "excess_vs_optimal_kg" in fs:
        e_tot, e_pct = fs["excess_vs_optimal_kg"], fs["excess_vs_optimal_pct"]
        print(f"  excess vs r=v=0 optimum  : {e_tot[0]:+.0f} kg "
              f"({e_pct[0]:+.2f}%)   [q25 {e_tot[1]:+.0f}, "
              f"q75 {e_tot[2]:+.0f}]   n = {fs['n_paired_optimal']}")
        if "excess_from_network_imitation_kg" in fs:
            print(f"    from network imitation : "
                  f"{fs['excess_from_network_imitation_kg'][0]:+.0f} kg")
            print(f"    from gate + patch      : "
                  f"{fs['excess_from_gate_and_patch_kg'][0]:+.0f} kg   "
                  f"(gate_mc point model predicted ~+247)")
    gen = summary.get("gate_error_vs_nominal")
    if gen:
        print(f"  gate arrival d_r_z       : {gen['d_r_z_m'][0]:+.2f} m     "
              f"(15/-5 campaign: +7.15 m)")
        print(f"  gate arrival d_v_z       : {gen['d_v_z_ms'][0]:+.2f} m/s   "
              f"(15/-5 campaign: +0.40 m/s)")
    print(f"  v_z >= 0 branch fired    : {hov} steps on {n_hov_ic} ICs "
          f"(expected 0)")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
