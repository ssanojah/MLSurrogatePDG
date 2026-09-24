#!/usr/bin/env python3
"""
RETALT1 PDG: FORCESPRO fixed-dt, variable-horizon solver bank.

Expert used to label DAgger queries. One solver per horizon
n_int = 1..62 (pdg_bank/pdg_N01 ... pdg_N62), each with t_f pinned to
n_int * dt so every horizon uses the same time step as the policy.
Reuses the model of forces_pdg.py.

Usage:
    python forces_pdg_fixeddt.py generate 1 62     # build the bank
    python forces_pdg_fixeddt.py smoke             # compare with free-t_f expert

    from forces_pdg_fixeddt import solve_fixed_dt
    res = solve_fixed_dt(solver, x0, tf_pin, N_stages)
"""

import os
import numpy as np
import casadi
import forcespro
import forcespro.nlp

from forces_pdg import (
    # physical params
    g0, Isp, m_dry, m0, T_max, T_min,
    cos_tilt, tgm, gs_tol, tf_min, tf_max,
    # dimensions
    NVAR, NEQ, NH, M_SUBSTEPS,
    # scaling
    ndf, ndm, ndp, ndv, ndt, SCALE_X,
    # objective weights
    W_FUEL, W_REG, COST_SCALE,
    # callbacks + inequality bounds (reused verbatim)
    continuous_dynamics, ineq, H_LOWER, H_UPPER,
)


# =====================================================================
# Objective closure (forces_pdg.obj, parameterised by N_stages)
# =====================================================================

def make_obj(N_stages):
    """Running cost of forces_pdg.obj for a solver with N_stages stages."""
    denom = float(N_stages - 1)

    def obj(z):
        T_norm = casadi.sqrt(casadi.dot(z[0:3], z[0:3]) + 1e-9)
        dt = (z[10] * ndt) / denom
        return COST_SCALE * (W_FUEL * T_norm * dt + W_REG * T_norm ** 2 * dt)

    return obj


def objN(z):
    """Terminal stage carries a dummy control; no fuel term."""
    return 0.0 * z[9]


# =====================================================================
# Generation (one pinned-t_f solver for a given horizon)
# =====================================================================

def generate_fixed_dt_solver(N_stages, name, pin_tf=True,
                             tf_lb_phys=0.0, tf_ub_phys=None):
    """Generate ONE FORCESPRO solver with N_stages stages and t_f pinned.

    Parameters
    ----------
    N_stages : int   number of stages (= intervals + 1)
    name     : str   output directory / solver name
    pin_tf   : bool  pin t_f via xinitidx (True for the fixed-dt bank)
    tf_lb_phys, tf_ub_phys : float [s]  t_f box in physical seconds
    """
    if tf_ub_phys is None:
        tf_ub_phys = 2.0 * tf_max   # generous; t_f is pinned via xinit anyway

    model = forcespro.nlp.SymbolicModel(N_stages)
    model.nvar = NVAR
    model.neq = NEQ
    model.nh = NH
    model.npar = 0

    model.objective = make_obj(N_stages)     # per-horizon objective
    model.objectiveN = objN

    model.continuous_dynamics = continuous_dynamics   # reused verbatim
    model.E = np.concatenate([np.zeros((NEQ, NVAR - NEQ)), np.eye(NEQ)], axis=1)

    model.ineq = ineq                         # reused verbatim
    model.hl = H_LOWER
    model.hu = H_UPPER

    Fbox = 1.2 * T_max / ndf
    model.lb = np.array([-Fbox, -Fbox, -Fbox,
                         -np.inf, -np.inf, 0.0,
                         -np.inf, -np.inf, -np.inf,
                         m_dry / ndm,
                         tf_lb_phys / ndt])
    model.ub = np.array([Fbox, Fbox, Fbox,
                         np.inf, np.inf, np.inf,
                         np.inf, np.inf, np.inf,
                         m0 / ndm,
                         tf_ub_phys / ndt])

    model.xinitidx = range(3, 11) if pin_tf else range(3, 10)   # pin t_f
    model.xfinalidx = range(3, 9)                                # r,v = 0

    codeoptions = forcespro.CodeOptions(name)
    codeoptions.maxit = 2000
    codeoptions.printlevel = 0
    codeoptions.optlevel = 3
    codeoptions.cleanup = False
    codeoptions.timing = 1
    codeoptions.solvemethod = 'PDIP_NLP'
    codeoptions.nlp.hessian_approximation = 'bfgs'
    codeoptions.nlp.integrator.type = 'ERK4'
    codeoptions.nlp.integrator.Ts = 1.0 / (N_stages - 1)
    codeoptions.nlp.integrator.nodes = M_SUBSTEPS

    return model.generate_solver(codeoptions)


def generate_solver_bank(n_int_min=1, n_int_max=62, out_root='pdg_bank',
                         tf_lb_phys=0.0, tf_ub_phys=None):
    """Generate the whole bank: one solver per horizon n_int in [n_int_min,
    n_int_max].
    """
    out_abs = os.path.abspath(out_root)
    os.makedirs(out_abs, exist_ok=True)
    cwd = os.getcwd()
    try:
        os.chdir(out_abs)
        for n_int in range(n_int_min, n_int_max + 1):
            N_stages = n_int + 1
            name = f'pdg_N{n_int:02d}'          # valid identifier (no slash)
            print(f"[bank] codegen n_int={n_int:2d} (N_stages={N_stages}) -> "
                  f"{out_root}/{name}")
            generate_fixed_dt_solver(N_stages, name, pin_tf=True,
                                     tf_lb_phys=tf_lb_phys, tf_ub_phys=tf_ub_phys)
    finally:
        os.chdir(cwd)
    print(f"[bank] done: {n_int_max - n_int_min + 1} solvers in {out_root}/")


# =====================================================================
# Solve (forces_pdg.solve_ocp with t_f pinned and N_stages threaded)
# =====================================================================

def _initial_guess(x0_phys, tf_pin, N_stages):
    """Linear-interpolation IC -> target guess with t_f set to the pinned value."""
    r0, v0, m_ic = x0_phys[0:3], x0_phys[3:6], x0_phys[6]
    a1 = np.linspace(1.0, 0.0, N_stages)
    a2 = 1.0 - a1
    x_guess = np.zeros((N_stages, 7))
    for k in range(N_stages):
        x_guess[k, 0:3] = a1[k] * r0
        x_guess[k, 3:6] = a1[k] * v0
        x_guess[k, 6] = a1[k] * m_ic + a2[k] * 1.1 * m_dry
    u_guess = np.tile(np.array([0.0, 0.0, m0 * g0]), (N_stages, 1))

    Z = np.zeros((N_stages, NVAR))
    Z[:, 0:3] = u_guess / ndf
    Z[:, 3:10] = x_guess / np.array([ndp, ndp, ndp, ndv, ndv, ndv, ndm])
    Z[:, 10] = tf_pin / ndt              # pinned t_f
    return Z.reshape(-1)


def _stage_arrays(output, N_stages):
    """Return the stage vectors [z_1, ..., z_{N_stages}] from a FORCESPRO output dict."""
    import re
    indexed = []
    for key in output.keys():
        m = re.fullmatch(r'x(\d+)', str(key))
        if m:
            indexed.append((int(m.group(1)), key))
    if len(indexed) < N_stages:
        raise KeyError(
            f"expected >= {N_stages} stage outputs 'x<n>', found {len(indexed)}; "
            f"actual output keys: {sorted(map(str, output.keys()))}")
    indexed.sort()
    return [output[key] for _, key in indexed[:N_stages]]


def solve_fixed_dt(solver, x0_phys, tf_pin, N_stages, z0=None):
    """Solve one pinned-t_f OCP."""
    x0_phys = np.asarray(x0_phys, dtype=float).flatten()

    # xinit for xinitidx = range(3,11): [r/ndp, v/ndv, m/ndm, t_f/ndt]
    xinit_scaled = np.concatenate([
        x0_phys / np.array([ndp, ndp, ndp, ndv, ndv, ndv, ndm]),
        [tf_pin / ndt],
    ])
    problem = {
        "x0": (_initial_guess(x0_phys, tf_pin, N_stages) if z0 is None
               else np.asarray(z0, dtype=float).ravel()),
        "xinit": xinit_scaled,
        "xfinal": np.zeros(6),
    }

    output, exitflag, info = solver.solve(problem)

    stages = _stage_arrays(output, N_stages)  # width-agnostic key parsing

    x_traj = np.zeros((N_stages, 8))         # SI: [r, v, m, t_f]
    u_traj = np.zeros((N_stages - 1, 3))     # MN
    for k in range(N_stages):
        zk = stages[k]
        x_traj[k] = zk[3:11] * SCALE_X
        if k < N_stages - 1:
            u_traj[k] = (zk[0:3] * ndf) / 1e6

    return {
        'x_traj': x_traj, 'u_traj': u_traj,
        'm_f': float(x_traj[-1, 6]),
        'tf_eff': float((N_stages - 1) * tf_pin / (N_stages - 1)),  # = tf_pin
        'exitflag': int(exitflag),
        'it': int(info.it),
        'solvetime': float(info.solvetime),
        'res_eq': float(info.res_eq),
        'res_ineq': float(info.res_ineq),
    }


# =====================================================================
# solve_fn factory for FixedDtExpert (fixed_dt_expert.py)
# =====================================================================

def make_forcespro_solve_fn(bank_dir='pdg_bank', accept_exitflag_0=False,
                            res_tol=1e-6):
    """Load the bank and return solve_fn(x0_phys, dt, n_int) for FixedDtExpert."""
    bank = {}
    for entry in sorted(os.listdir(bank_dir)):
        if entry.startswith('pdg_N'):
            n_int = int(entry.split('pdg_N')[1][:2])
            bank[n_int] = forcespro.nlp.Solver.from_directory(
                os.path.join(bank_dir, entry))
    if not bank:
        raise FileNotFoundError(
            f"No pdg_N* solvers in {bank_dir}/ — run generate_solver_bank first.")
    print(f"[bank] loaded {len(bank)} solvers: n_int in "
          f"[{min(bank)}..{max(bank)}]")

    def solve_fn(x0_phys, dt, n_int):
        n_int = int(n_int)
        solver = bank.get(n_int)
        if solver is None:
            return {'feasible': False, 'x_traj': None, 'u_traj': None,
                    'm_f': None, 'status': f'no_solver_n{n_int}',
                    'solve_time': 0.0}

        tf_pin = n_int * float(dt)          # pinned final time
        r = solve_fixed_dt(solver, x0_phys, tf_pin, N_stages=n_int + 1)

        feasible = (r['exitflag'] == 1) or (
            accept_exitflag_0 and r['exitflag'] == 0
            and r['res_eq'] < res_tol and r['res_ineq'] < res_tol)

        if not feasible:
            return {'feasible': False, 'x_traj': None, 'u_traj': None,
                    'm_f': None, 'status': f'exitflag={r["exitflag"]}',
                    'solve_time': r['solvetime']}

        # FixedDtExpert contract: 7-state x_traj (drop the pinned t_f column)
        return {'feasible': True,
                'x_traj': r['x_traj'][:, :7],
                'u_traj': r['u_traj'],
                'm_f': r['m_f'],
                'status': 'exitflag=1',
                'solve_time': r['solvetime']}

    return solve_fn


# =====================================================================
# Smoke test (requires the n_int = 60 solver)
# =====================================================================

def _smoke_test(bank_dir='pdg_bank'):
    """At k = 0 the fixed-dt bank (n_int = 60, dt = t_f*/60) should closely
    reproduce the free-t_f solve's first control.
    """
    from fixed_dt_expert import FixedDtExpert
    x0 = np.array([-2386.0, 223.0, 6038.0, 199.0, -18.0, -251.0, m0])

    # free-t_f reference (one solver, N_STAGES = 61)
    ref_solver = forcespro.nlp.Solver.from_directory('pdg_expert')
    from forces_pdg import solve_ocp
    ref = solve_ocp(ref_solver, x0, tf_guess=50.0)
    tf_star = ref['tf']; dt = tf_star / 60.0
    print(f"free-t_f ref: exitflag {ref['exitflag']}, t_f*={tf_star:.3f}s, "
          f"dt={dt:.4f}s, m_f={ref['m_f']/1e3:.3f}t, u0={ref['u_traj'][0]}")

    expert = FixedDtExpert(make_forcespro_solve_fn(bank_dir), n_bracket=2)
    res = expert.query(x0, dt=dt, n_int_nominal=60, return_traj=True)
    print(f"fixed-dt bank: feasible {res['n_feasible']}/5, winner n_int="
          f"{res['n_int_opt']}, m_f={None if res['m_f'] is None else res['m_f']/1e3:.3f}t")
    if res['feasible']:
        print(f"  u0={res['u_opt']}  |u0-u0_ref|="
              f"{np.linalg.norm(res['u_opt'] - ref['u_traj'][0]):.4e} MN")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'generate':
        # e.g.  python forces_pdg_fixeddt.py generate 1 62
        lo = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        hi = int(sys.argv[3]) if len(sys.argv) > 3 else 62
        generate_solver_bank(n_int_min=lo, n_int_max=hi)
    elif len(sys.argv) > 1 and sys.argv[1] == 'smoke':
        _smoke_test()
    else:
        print("usage: forces_pdg_fixeddt.py [generate LO HI | smoke]")
