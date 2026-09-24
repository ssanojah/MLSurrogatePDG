#!/usr/bin/env python3
"""
Monte Carlo of the analytic terminal law over candidate gate conditions.

Flies the terminal law from a dispersed set of gate arrival states for each
candidate (h_gate, v_gate) and reports landing outcomes, propellant use and
control authority. No network or solver is used.

Usage:
    python gate_mc.py --gates 15,-14 20,-16 --n 2000
    python gate_mc.py --gates 20,-16 --prop_csv gate_arrivals.csv \\
        --prop_col mass_margin_kg
    python gate_mc.py --gates 15,-14 --grid_dh=-6:12:19 --grid_dv=-4:4:17
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------
# 0. BACKEND LOADING — physics and law are SEPARATE roles
# ---------------------------------------------------------------------

NEED = ["g0", "Isp", "m_dry", "T_min_MN", "T_max_MN", "theta_max",
        "rho", "S_D", "C_D", "_F_rk4", "enforce_thrust",
        "classify_landing", "LandingThresholds"]

_PHYS_SRC = None
_LAW_SRC = None
_law_fn_ext = None


def _import(name):
    """Import by module name or by path to a .py file."""
    import importlib
    p = Path(name)
    if p.suffix == ".py":
        if not p.exists():
            raise FileNotFoundError(f"no such file: {p}")
        sys.path.insert(0, str(p.resolve().parent))
        name = p.stem
    return importlib.import_module(name)


def load_backend(phys_module, law_module, law_fn):
    """Populate module globals from --phys_module; resolve the law."""
    global _PHYS_SRC, _LAW_SRC, _law_fn_ext
    global g0, Isp, m_dry, T_min_MN, T_max_MN, theta_max
    global rho, S_D, C_D, _F_rk4, enforce_thrust, classify_landing
    global LandingThresholds

    mod = _import(phys_module)
    miss = [a for a in NEED if not hasattr(mod, a)]
    if miss:
        raise AttributeError(
            f"--phys_module '{phys_module}' is missing {miss}.\n"
            f"Pass the module that DEFINES the physics (standalone_eval_v2 "
            f"or standalone_eval_v3), not one that re-imports a subset of "
            f"it. terminal_patch_eval is not valid here: its import list "
            f"omits Isp, rho, S_D and C_D.")
    for a in NEED:
        globals()[a] = getattr(mod, a)
    _PHYS_SRC = mod.__name__

    if law_module in (None, "", "local"):
        _law_fn_ext, _LAW_SRC = None, "LOCAL reference re-implementation"
        return

    lmod = _import(law_module)
    fn = getattr(lmod, law_fn, None)
    if fn is None:
        cands = [n for n in dir(lmod) if any(k in n.lower() for k in
                 ("term", "patch", "arc", "brake", "guid"))]
        raise RuntimeError(
            f"'{law_module}' has no '{law_fn}'. Candidates: "
            f"{cands or '(none found)'} — pass --law_fn.")
    _law_fn_ext, _LAW_SRC = fn, f"{lmod.__name__}.{law_fn}"


# ---------------------------------------------------------------------
# 1. REFERENCE TERMINAL LAW  (used only with --law_module local)
# ---------------------------------------------------------------------

def terminal_command_local(x7, cfg):
    """Reference implementation, mirroring terminal_patch_eval.terminal_command
    INCLUDING the t_go clamp.
    """
    r_h = np.asarray(x7[:2], float)
    v_h = np.asarray(x7[3:5], float)
    r_z, v_z, m = float(x7[2]), float(x7[5]), float(x7[6])
    h_eff = r_z - cfg["h_c"]

    if h_eff > cfg["eps_rz"]:
        if v_z < 0.0:
            a_req = (v_z ** 2 - cfg["v_td"] ** 2) / (2.0 * h_eff)
        else:
            return np.array([0.0, 0.0, T_min_MN]), dict(a_req=0.0)
    else:
        a_req = (-cfg["v_td"] - v_z) / cfg["tau_capture"]
    a_req = float(np.clip(a_req, 0.0, cfg["a_cap"]))

    if v_z < -cfg["eps_v"] and h_eff > cfg["eps_rz"]:
        t_go = 2.0 * h_eff / abs(v_z)
    else:
        t_go = cfg["t_go_max"]
    t_go = float(np.clip(t_go, cfg["t_go_min"], cfg["t_go_max"]))

    a_hor = -6.0 * r_h / t_go ** 2 - 4.0 * v_h / t_go
    if cfg["no_horizontal"]:
        a_hor = np.zeros(2)

    u_raw = m * np.array([a_hor[0], a_hor[1], a_req + g0]) / 1e6
    return enforce_thrust(u_raw, clip_tilt=True), dict(a_req=a_req)


def make_tc(cfg: "RunCfg"):
    """Return tc(x) -> u_MN, adapting whichever law was resolved."""
    fn = _law_fn_ext or terminal_command_local
    law_cfg = cfg.law_cfg()

    def tc(x):
        out = fn(np.asarray(x, float), law_cfg)
        return np.asarray(out[0] if isinstance(out, tuple) else out, float)
    return tc


# ---------------------------------------------------------------------
# 2. DIAGNOSTICS
# ---------------------------------------------------------------------

def a_required(x, cfg):
    h = max(x[2] - cfg.h_c, cfg.eps_rz)
    return max((x[5] ** 2 - cfg.v_td ** 2) / (2.0 * h), 0.0)


def a_available(x, u_cmd, cfg):
    """Vertical deceleration headroom."""
    m = x[6]
    if cfg.auth_def == "cone":
        return T_max_MN * 1e6 * np.cos(theta_max) / m - g0
    n = u_cmd / max(np.linalg.norm(u_cmd), 1e-12)
    v = x[3:6]
    D_z = -0.5 * rho * S_D * C_D * np.linalg.norm(v) * v[2]
    return (T_max_MN * 1e6 * n[2] + D_z) / m - g0


def switching_speed(h, m):
    """v* = sqrt(2 a_max h): the max-thrust braking arc."""
    a_max = T_max_MN * 1e6 / m - g0
    return float(np.sqrt(2.0 * a_max * h)), float(a_max)


# ---------------------------------------------------------------------
# 3. ROLLOUT
# ---------------------------------------------------------------------

@dataclass
class RunCfg:
    # integration / diagnostics
    dt: float = 0.05
    t_max: float = 60.0
    freefall_on_exhaust: bool = True
    auth_def: str = "cone"
    auth_min_alt: float = 0.5
    h_c: float = 0.0
    v_td: float = 0.0
    tau_capture: float = 0.3
    a_cap: float = 50.0
    eps_rz: float = 0.01
    eps_v: float = 0.01
    t_go_min: float = 0.2
    t_go_max: float = 60.0
    no_horizontal: bool = False

    def law_cfg(self):
        return dict(h_c=self.h_c, v_td=self.v_td,
                    tau_capture=self.tau_capture, a_cap=self.a_cap,
                    eps_rz=self.eps_rz, eps_v=self.eps_v,
                    t_go_min=self.t_go_min, t_go_max=self.t_go_max,
                    no_horizontal=self.no_horizontal)


def _bisect(x_lo, u, dt, iters=60):
    lo, hi = 0.0, dt
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if np.array(_F_rk4(x_lo, u, mid)).flatten()[2] > 0:
            lo = mid
        else:
            hi = mid
    return np.array(_F_rk4(x_lo, u, 0.5 * (lo + hi))).flatten()


def _freefall(x, cfg, thres):
    """Engine out: coast to the pad and classify the impact."""
    u0 = np.zeros(3)
    xx, t = np.asarray(x, float).copy(), 0.0
    while t < 20.0:
        prev = xx.copy()
        xx = np.array(_F_rk4(xx, u0, cfg.dt)).flatten()
        t += cfg.dt
        if xx[2] < 0.0:
            cat, det = classify_landing(_bisect(prev, u0, cfg.dt), thres)
            return dict(cat=cat, v_z=float(det["vz_signed"]),
                        vh=float(det["vh"]), t_fall=t)
    return dict(cat="none", v_z=float("nan"), vh=float("nan"), t_fall=t)


def run_patch(x0, cfg: RunCfg, thres, tc):
    """Integrate the terminal law from an arrival state to termination."""
    x = np.asarray(x0, float).copy()
    m0 = x[6]
    t, n = 0.0, 0
    n_Tmax = n_Tmin = n_tilt = n_infeas = 0
    auth_peak, a_avail_min = 0.0, np.inf
    ascent_alt, ascent_events = np.nan, 0
    cone = theta_max - 1e-9

    while True:
        if x[6] <= m_dry:
            out = dict(outcome="fuel_exhausted", t=t,
                       r_z=float(x[2]), v_z=float(x[5]))
            if cfg.freefall_on_exhaust:
                out["impact"] = _freefall(x, cfg, thres)
            break
        if t > cfg.t_max:
            out = dict(outcome="timeout", t=t,
                       r_z=float(x[2]), v_z=float(x[5]))
            break

        u = tc(x)

        if x[5] < 0.0 and x[2] > cfg.auth_min_alt:
            a_req = a_required(x, cfg)
            a_av = a_available(x, u, cfg)
            a_avail_min = min(a_avail_min, a_av)
            if a_av > 1e-6:
                auth = a_req / a_av
                auth_peak = max(auth_peak, auth)
                if auth > 1.0:
                    n_infeas += 1

        mag = np.linalg.norm(u)
        if mag >= T_max_MN * (1 - 1e-9):
            n_Tmax += 1
        if mag <= T_min_MN * (1 + 1e-9):
            n_Tmin += 1
        if np.arctan2(np.linalg.norm(u[:2]), u[2]) >= cone:
            n_tilt += 1

        prev = x.copy()
        x = np.array(_F_rk4(x, u, cfg.dt)).flatten()
        t += cfg.dt
        n += 1

        if not np.all(np.isfinite(x)):
            out = dict(outcome="diverged", t=t)
            break
        if x[5] >= 0.0 and x[2] > 0.05:          # hover / ascent branch
            ascent_events += 1
            if np.isnan(ascent_alt):
                ascent_alt = float(x[2])
        if x[2] < 0.0:                           # ground crossing
            cat, det = classify_landing(_bisect(prev, u, cfg.dt), thres)
            out = dict(outcome=cat, t=t,
                       **{k: float(v) for k, v in det.items()})
            break

    out.update(prop_used=float(m0 - x[6]), n_steps=n,
               auth_peak=float(auth_peak),
               a_avail_min=float(a_avail_min if np.isfinite(a_avail_min)
                                 else np.nan),
               frac_Tmax=n_Tmax / max(n, 1),
               frac_Tmin=n_Tmin / max(n, 1),
               frac_tilt=n_tilt / max(n, 1),
               frac_infeasible=n_infeas / max(n, 1),
               ascent_alt=float(ascent_alt), ascent_events=ascent_events)
    return out


# ---------------------------------------------------------------------
# 4. ARRIVAL DISPERSION MODEL
# ---------------------------------------------------------------------

@dataclass
class Arrival:
    """Commanded gate (h_gate, v_gate) -> sampled arrival state."""
    dh_mode: str = "abs"
    dh_bias: float = 6.0
    dh_rel: float = 0.40
    dh_per_v: float = 1.2
    dh_sigma: float = 3.0
    dv_frac: float = 0.94
    dv_sigma: float = 0.5
    rh_mean: float = 1.7
    rh_sigma: float = 1.0
    vh_mean: float = 0.17
    vh_sigma: float = 0.15

    def bias_h(self, h_gate, v_gate):
        if self.dh_mode == "rel":
            return self.dh_rel * h_gate
        if self.dh_mode == "vel":
            return self.dh_per_v * abs(v_gate)
        return self.dh_bias

    def sample(self, h_gate, v_gate, m, rng):
        rz = max(h_gate + self.bias_h(h_gate, v_gate)
                 + rng.normal(0, self.dh_sigma), 0.5)
        vz = min(self.dv_frac * v_gate + rng.normal(0, self.dv_sigma), -0.1)
        rh = max(rng.normal(self.rh_mean, self.rh_sigma), 0.0)
        vh = max(rng.normal(self.vh_mean, self.vh_sigma), 0.0)
        th, tv = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
        return np.array([rh * np.cos(th), rh * np.sin(th), rz,
                         vh * np.cos(tv), vh * np.sin(tv), vz, m])

    def nominal(self, h_gate, v_gate, m, dh=None, dv=None):
        rz = h_gate + (self.bias_h(h_gate, v_gate) if dh is None else dh)
        vz = (self.dv_frac * v_gate) if dv is None else (v_gate + dv)
        return np.array([self.rh_mean, 0.0, max(rz, 0.5),
                         self.vh_mean, 0.0, min(vz, -0.1), m])


# ---------------------------------------------------------------------
# 5. SWEEPS
# ---------------------------------------------------------------------

def mc_gate(h_gate, v_gate, arr, cfg, thres, tc, n, margin, rng, tag):
    rows = []
    for _ in range(n):
        m = m_dry + (margin() if callable(margin) else margin)
        x0 = arr.sample(h_gate, v_gate, m, rng)
        r = run_patch(x0, cfg, thres, tc)
        r.update(pass_=tag, h_gate=h_gate, v_gate=v_gate,
                 arr_rz=float(x0[2]), arr_vz=float(x0[5]),
                 arr_rh=float(np.linalg.norm(x0[:2])), arr_m=float(x0[6]))
        rows.append(r)
    return rows


def grid_gate(h_gate, v_gate, arr, cfg, thres, tc, dh_vals, dv_vals, margin):
    """Deterministic (d_r_z, d_v_z) map at nominal mass. d_v_z is ADDED to the
    commanded v_gate, so d_v_z < 0 means arriving faster.
    """
    out, m = [], m_dry + margin
    for dh in dh_vals:
        for dv in dv_vals:
            x0 = arr.nominal(h_gate, v_gate, m, dh=dh, dv=dv)
            r = run_patch(x0, cfg, thres, tc)
            out.append(dict(h_gate=h_gate, v_gate=v_gate,
                            d_rz=float(dh), d_vz=float(dv),
                            arr_rz=float(x0[2]), arr_vz=float(x0[5]),
                            outcome=r["outcome"], auth_peak=r["auth_peak"],
                            frac_Tmax=r["frac_Tmax"],
                            ascent_alt=r["ascent_alt"],
                            prop_used=r["prop_used"]))
    return out


# ---------------------------------------------------------------------
# 6. REPORTING
# ---------------------------------------------------------------------

CATS = ["soft", "hard", "crash", "fuel_exhausted", "timeout", "diverged"]


def summarise(rows):
    n = len(rows)
    cnt = {c: sum(1 for r in rows if r["outcome"] == c) for c in CATS}
    landed = [r for r in rows if r["outcome"] in ("soft", "hard", "crash")]
    prop = np.array([r["prop_used"] for r in landed]) if landed \
        else np.array([np.nan])
    auth = np.array([r["auth_peak"] for r in rows])
    fe = [r for r in rows if r["outcome"] == "fuel_exhausted"]
    fe_alt = np.array([r["r_z"] for r in fe]) if fe else np.array([np.nan])
    fe_surv = sum(1 for r in fe
                  if r.get("impact", {}).get("cat") in ("soft", "hard"))

    def pc(a, q):
        return float(np.nanpercentile(a, q)) if np.any(np.isfinite(a)) \
            else float("nan")

    return dict(
        n=n, **{f"n_{c}": cnt[c] for c in CATS},
        soft_pct=100.0 * cnt["soft"] / n,
        survivable_pct=100.0 * (cnt["soft"] + cnt["hard"] + fe_surv) / n,
        prop_p50=pc(prop, 50), prop_p90=pc(prop, 90), prop_p99=pc(prop, 99),
        prop_max=float(np.nanmax(prop)) if np.any(np.isfinite(prop))
        else float("nan"),
        auth_p50=pc(auth, 50), auth_p99=pc(auth, 99),
        auth_max=float(np.nanmax(auth)),
        frac_infeas_runs=float(np.mean([r["frac_infeasible"] > 0
                                        for r in rows])),
        frac_Tmax_steps=float(np.mean([r["frac_Tmax"] for r in rows])),
        frac_Tmin_steps=float(np.mean([r["frac_Tmin"] for r in rows])),
        frac_tilt_steps=float(np.mean([r["frac_tilt"] for r in rows])),
        n_ascent=sum(1 for r in rows if r["ascent_events"] > 0),
        fe_alt_p50=pc(fe_alt, 50), fe_survivable=fe_surv,
    )


def print_table(title, note, summ):
    print(f"\n{title}")
    if note:
        print(f"  {note}")
    hdr = (f"{'gate':>10} {'v*':>6} {'soft%':>6} {'surv%':>6} {'auth50':>7} "
           f"{'auth99':>7} {'infeas':>7} {'Tmax':>6} {'tilt':>6} {'asc':>5} "
           f"{'p50':>6} {'p90':>6} {'p99':>6} {'max':>7}")
    print(hdr)
    print("-" * len(hdr))
    for g, s in summ.items():
        print(f"{g:>10} {s['v_star']:6.1f} {s['soft_pct']:6.1f} "
              f"{s['survivable_pct']:6.1f} {s['auth_p50']:7.2f} "
              f"{s['auth_p99']:7.2f} {s['frac_infeas_runs']:7.2f} "
              f"{s['frac_Tmax_steps']:6.2f} {s['frac_tilt_steps']:6.2f} "
              f"{s['n_ascent']:5d} {s['prop_p50']:6.0f} {s['prop_p90']:6.0f} "
              f"{s['prop_p99']:6.0f} {s['prop_max']:7.0f}")


def write_csv(path, rows):
    if not rows:
        return
    cols = sorted({k for r in rows for k in r if not isinstance(r[k], dict)})
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")


def plot_grid(grid_rows, out_dir, dh_vals, dv_vals):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
    except Exception as e:                                # pragma: no cover
        print(f"  [plot skipped: {e}]")
        return
    gates = sorted({(r["h_gate"], r["v_gate"]) for r in grid_rows})
    order = ["soft", "hard", "crash", "fuel_exhausted", "timeout", "diverged"]
    colors = ["tab:green", "tab:orange", "tab:red", "tab:purple",
              "tab:gray", "black"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(list(range(len(order) + 1)), len(colors))
    ncol = min(3, len(gates))
    nrow = int(np.ceil(len(gates) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.7 * ncol, 4.0 * nrow),
                             squeeze=False)
    for ax, (h, v) in zip(axes.ravel(), gates):
        Z = np.full((len(dv_vals), len(dh_vals)), np.nan)
        A = np.full_like(Z, np.nan)
        for r in grid_rows:
            if (r["h_gate"], r["v_gate"]) != (h, v):
                continue
            i, j = dv_vals.index(r["d_vz"]), dh_vals.index(r["d_rz"])
            Z[i, j] = order.index(r["outcome"])
            A[i, j] = r["auth_peak"]
        ax.pcolormesh(dh_vals, dv_vals, Z, cmap=cmap, norm=norm,
                      shading="nearest")
        if np.any(np.isfinite(A)):
            cs = ax.contour(dh_vals, dv_vals, A, levels=[0.8, 1.0],
                            colors="k", linewidths=[0.8, 1.8])
            ax.clabel(cs, fmt={0.8: "0.8", 1.0: "auth 1.0"}, fontsize=7)
        ax.set_title(f"gate {h:.0f} m / {v:.0f} m/s", fontsize=10)
        ax.set_xlabel(r"arrival altitude error  $\Delta r_z$ [m]")
        ax.set_ylabel(r"arrival speed error  $\Delta v_z$ [m/s]")
    for ax in axes.ravel()[len(gates):]:
        ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    fig.legend(handles, order, loc="lower center", ncol=len(order),
               frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    p = Path(out_dir) / "gate_feasibility_map.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


# ---------------------------------------------------------------------
# 7. LAW PROBE
# ---------------------------------------------------------------------

def probe_law(tc):
    """Report what the resolved law commands in three regimes."""
    m = m_dry + 2000.0
    hover = m * g0 / 1e6
    cases = [
        ("descending  21.0 m, -13 m/s", np.array([1.7, 0, 21.0, .17, 0, -13.0, m])),
        ("ASCENDING   10.0 m,  +1 m/s", np.array([1.7, 0, 10.0, .17, 0, +1.0, m])),
        ("near pad     0.2 m,  -1 m/s", np.array([0.2, 0, 0.2, .05, 0, -1.0, m])),
    ]
    print(f"\n  law probe  (m = {m/1e3:.1f} t, hover thrust = "
          f"{hover*1e3:.0f} kN)")
    for name, x in cases:
        u = tc(x)
        mag, tz = float(np.linalg.norm(u)), float(u[2])
        tag = ""
        if abs(tz - hover) < 1e-4 * hover:
            tag = "   <-- T_z == weight: HOVER EQUILIBRIUM (pre-fix branch)"
        elif abs(mag - T_min_MN) < 1e-9:
            tag = "   <-- T_min (post-fix branch)"
        tilt = np.degrees(np.arctan2(np.linalg.norm(u[:2]), u[2]))
        print(f"    {name}:  |T| = {mag*1e3:7.1f} kN   T_z = {tz*1e3:7.1f} kN"
              f"   tilt = {tilt:5.1f} deg{tag}")


# ---------------------------------------------------------------------
# 8. MAIN
# ---------------------------------------------------------------------

def parse_gates(items):
    out = []
    for s in items:
        h, v = s.split(",")
        out.append((float(h), float(v)))
    return out


def parse_range(spec):
    lo, hi, n = spec.split(":")
    return [float(x) for x in np.linspace(float(lo), float(hi), int(n))]


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # --- modules ---
    p.add_argument("--phys_module", default="standalone_eval_v2",
                   help="module DEFINING physics/enforcement/taxonomy")
    p.add_argument("--law_module", default="terminal_patch_eval",
                   help="module defining the terminal law, or 'local'")
    p.add_argument("--law_fn", default="terminal_command")

    # --- sweep ---
    p.add_argument("--gates", nargs="+",
                   default=["15,-12", "15,-14", "15,-15", "20,-16"],
                   help='candidates as "h_gate,v_gate" (v negative)')
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--mode", choices=["mc", "grid", "both"], default="both")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="gate_mc_out")
    p.add_argument("--no_probe", action="store_true")

    # --- integration / diagnostics ---
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--t_max", type=float, default=60.0)
    p.add_argument("--auth_def", choices=["cone", "commanded"],
                   default="cone")
    p.add_argument("--auth_min_alt", type=float, default=0.5)
    p.add_argument("--no_freefall", action="store_true")

    # --- terminal-law parameters ---
    p.add_argument("--h_c", type=float, default=0.0)
    p.add_argument("--v_td", type=float, default=0.0)
    p.add_argument("--tau_capture", type=float, default=0.3)
    p.add_argument("--a_cap", type=float, default=50.0)
    p.add_argument("--eps_rz", type=float, default=0.01)
    p.add_argument("--eps_v", type=float, default=0.01)
    p.add_argument("--t_go_min", type=float, default=0.2)
    p.add_argument("--t_go_max", type=float, default=60.0)
    p.add_argument("--no_horizontal", action="store_true")

    # --- arrival dispersion ---
    p.add_argument("--dh_mode", choices=["abs", "rel", "vel"], default="abs")
    p.add_argument("--dh_bias", type=float, default=6.0)
    p.add_argument("--dh_rel", type=float, default=0.40)
    p.add_argument("--dh_per_v", type=float, default=1.2)
    p.add_argument("--dh_sigma", type=float, default=3.0)
    p.add_argument("--dv_frac", type=float, default=0.94)
    p.add_argument("--dv_sigma", type=float, default=0.5)
    p.add_argument("--rh_mean", type=float, default=1.7)
    p.add_argument("--rh_sigma", type=float, default=1.0)
    p.add_argument("--vh_mean", type=float, default=0.17)
    p.add_argument("--vh_sigma", type=float, default=0.15)
    p.add_argument("--nobias", action="store_true",
                   help="arrive at the COMMANDED gate in the mean. "
                        "Conservative sizing: assumes the over-braking "
                        "bias vanishes as the fit improves.")

    # --- propellant at the gate ---
    p.add_argument("--prop_ample", type=float, default=5000.0)
    p.add_argument("--prop_lo", type=float, default=500.0)
    p.add_argument("--prop_hi", type=float, default=2500.0)
    p.add_argument("--prop_csv", default=None,
                   help="CSV of real per-IC arrival margins [kg]")
    p.add_argument("--prop_col", default="mass_margin_kg")
    p.add_argument("--nominal_margin", type=float, default=2000.0,
                   help="margin [kg] used for the deterministic grid")

    # --- grid ---
    p.add_argument("--grid_dh", default="-6:12:19",
                   help="lo:hi:n for d_r_z [m]. Use the '=' form, e.g. "
                        "--grid_dh=-6:12:19, because the value starts "
                        "with a minus sign.")
    p.add_argument("--grid_dv", default="-4:4:17",
                   help="lo:hi:n for d_v_z [m/s]. Use the '=' form, e.g. "
                        "--grid_dv=-4:4:17.")

    args = p.parse_args()

    load_backend(args.phys_module, args.law_module, args.law_fn)

    cfg = RunCfg(dt=args.dt, t_max=args.t_max,
                 freefall_on_exhaust=not args.no_freefall,
                 auth_def=args.auth_def, auth_min_alt=args.auth_min_alt,
                 h_c=args.h_c, v_td=args.v_td,
                 tau_capture=args.tau_capture, a_cap=args.a_cap,
                 eps_rz=args.eps_rz, eps_v=args.eps_v,
                 t_go_min=args.t_go_min, t_go_max=args.t_go_max,
                 no_horizontal=args.no_horizontal)
    tc = make_tc(cfg)
    thres = LandingThresholds()
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    arr = Arrival(dh_mode=args.dh_mode, dh_bias=args.dh_bias,
                  dh_rel=args.dh_rel, dh_per_v=args.dh_per_v,
                  dh_sigma=args.dh_sigma, dv_frac=args.dv_frac,
                  dv_sigma=args.dv_sigma, rh_mean=args.rh_mean,
                  rh_sigma=args.rh_sigma, vh_mean=args.vh_mean,
                  vh_sigma=args.vh_sigma)
    if args.nobias:
        arr.dh_bias = arr.dh_rel = arr.dh_per_v = 0.0
        arr.dv_frac = 1.0

    if args.prop_csv:
        import csv
        with open(args.prop_csv) as f:
            rd = csv.DictReader(f)
            if args.prop_col not in (rd.fieldnames or []):
                raise KeyError(f"column '{args.prop_col}' not in "
                               f"{args.prop_csv}; have {rd.fieldnames}")
            vals = np.array([float(r[args.prop_col]) for r in rd
                             if r[args.prop_col] not in (None, "")])
        if vals.size == 0:
            raise ValueError(f"no usable values in {args.prop_csv}")
        margin = lambda: float(rng.choice(vals))                 # noqa: E731
        prop_desc = (f"empirical {args.prop_csv}:{args.prop_col} "
                     f"(n={vals.size}, p05={np.percentile(vals,5):.0f}, "
                     f"p50={np.percentile(vals,50):.0f} kg)")
    else:
        margin = lambda: float(rng.uniform(args.prop_lo, args.prop_hi))  # noqa: E731
        prop_desc = (f"uniform [{args.prop_lo:.0f}, {args.prop_hi:.0f}] kg "
                     f"— SYNTHETIC; Pass B soft% mostly restates this")

    print("=" * 78)
    print("PATCH-LAW MONTE CARLO — no network, no solver")
    print("=" * 78)
    print(f"  physics / enforcement : {_PHYS_SRC}")
    print(f"  terminal law          : {_LAW_SRC}")
    print(f"  dt = {cfg.dt} s   tilt cone = {np.degrees(theta_max):.0f} deg   "
          f"T in [{T_min_MN*1e3:.0f}, {T_max_MN*1e3:.0f}] kN   "
          f"m_dry = {m_dry/1e3:.1f} t")
    print(f"  law cfg               : h_c={cfg.h_c}  v_td={cfg.v_td}  "
          f"t_go=[{cfg.t_go_min}, {cfg.t_go_max}]  a_cap={cfg.a_cap}  "
          f"tau={cfg.tau_capture}")
    print(f"  authority definition  : {cfg.auth_def}")
    print(f"  arrival bias          : mode={arr.dh_mode}  "
          f"dv_frac={arr.dv_frac}"
          + ("   [--nobias: commanded gate]" if args.nobias else ""))
    print(f"  margin (pass B)       : {prop_desc}")
    print(f"  samples per gate      : {args.n}")
    if not args.no_probe:
        probe_law(tc)

    all_rows, summ_A, summ_B, grid_rows = [], {}, {}, []
    dh_vals, dv_vals = parse_range(args.grid_dh), parse_range(args.grid_dv)

    for (h, v) in parse_gates(args.gates):
        key = f"{h:.0f}/{v:+.0f}"
        v_star, _ = switching_speed(h, m_dry + 2000.0)
        below = abs(v) > v_star
        if below:
            print(f"\n  !! gate {key}: |v_gate| = {abs(v):.1f} > "
                  f"v* = {v_star:.1f} m/s. The COMMANDED gate lies below "
                  f"the max-thrust braking arc — no throttle setting can "
                  f"land from it. The expert solver will still converge to "
                  f"it, because its terminal constraint is only to REACH "
                  f"the gate.")

        if args.mode in ("mc", "both"):
            ra = mc_gate(h, v, arr, cfg, thres, tc, args.n,
                         args.prop_ample, rng, "A_authority")
            rb = mc_gate(h, v, arr, cfg, thres, tc, args.n,
                         margin, rng, "B_fuel")
            all_rows += ra + rb
            sa, sb = summarise(ra), summarise(rb)
            for s in (sa, sb):
                s["v_star"] = v_star
                s["gate_below_arc"] = bool(below)
            summ_A[key], summ_B[key] = sa, sb

        if args.mode in ("grid", "both"):
            grid_rows += grid_gate(h, v, arr, cfg, thres, tc,
                                   dh_vals, dv_vals, args.nominal_margin)

    if summ_A:
        print_table(f"PASS A — control authority "
                    f"(propellant = {args.prop_ample:.0f} kg, ample)",
                    "propellant percentiles here are UNCENSORED — size "
                    "against these", summ_A)
        print_table("PASS B — with arrival margin dispersion",
                    "propellant percentiles here are CENSORED (expensive "
                    "runs became fuel_exhausted) — do NOT size on them",
                    summ_B)
        print("\n  soft%   Carradori GR-04/05/06 soft box")
        print("  surv%   soft + hard + fuel-exhausted whose free-fall "
              "impact is still inside the hard box")
        print("  auth    peak a_req / a_available along the run; > 1 means "
              "the law demanded more than the vehicle had")
        print("  Tmax/tilt  mean fraction of steps saturated")
        print("  asc     runs in which v_z reached >= 0 above 0.05 m")
        write_csv(out_dir / "runs.csv", all_rows)
        with open(out_dir / "summary.json", "w") as f:
            json.dump(dict(phys_module=_PHYS_SRC, law=_LAW_SRC,
                           pass_A=summ_A, pass_B=summ_B,
                           config=vars(args)), f, indent=2, default=str)
        print(f"\n  Saved: {out_dir/'runs.csv'}, {out_dir/'summary.json'}")

    if grid_rows:
        write_csv(out_dir / "grid.csv", grid_rows)
        plot_grid(grid_rows, out_dir, dh_vals, dv_vals)
        print(f"  Saved: {out_dir/'grid.csv'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
