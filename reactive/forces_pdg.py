"""
FORCESPRO expert solver with a variable horizon (reactive branch).

Same model as the root forces_pdg.py, with the number of stages and the
t_f bounds as parameters of generate_solver() and solve_ocp(). Used to
build the three solvers of the reactive expert bank.

Usage:
    model, solver = generate_solver(n_stages=31, tf_lo=5.0, tf_hi=25.0,
                                    name='pdg_expert_med')
    result = solve_ocp(solver, x0, tf_guess=15.0, n_stages=31)
"""

import numpy as np
import casadi
import forcespro
import forcespro.nlp
import matplotlib.pyplot as plt


# =============================================================================
# 1. PHYSICAL PARAMETERS (RETALT1, SI units, thrust in NEWTONS)
# =============================================================================

g0    = 9.807
Isp   = 372.2
m_dry = 59.3e3
m_overhead = 0.5e3                 # kg, propellant reserve kept above m_dry
m0    = 70.0e3

T_max = 1.179e6                    # N  (100% throttle)
T_min = 472e3                      # N  (40% throttle)

gamma_gs = np.deg2rad(30.0)        # glide-slope: min elevation from horizontal
tgm      = float(np.tan(gamma_gs))
gs_tol   = 0.1                     # m, drops cone tip just below the pad

theta_max = np.deg2rad(15.0)       # max thrust tilt from vertical
cos_tilt  = float(np.cos(theta_max))

rho = 1.225                        # constant sea-level density (matches IPOPT)
S_D = np.pi / 4 * 6.0**2
C_D = 1.0

tf_min, tf_max = 10.0, 80.0

N_STAGES  = 61
NVAR      = 11                     # 3 controls + 8 states
NEQ       = 8
NH        = 3
M_SUBSTEPS = 2

# Stage layout (controls first):
#   z = [ F_x F_y F_z | r_x r_y r_z | v_x v_y v_z | m | t_f ]   (all SCALED)
#         0   1   2     3   4   5     6   7   8     9   10


# =============================================================================
# 2. SCALING
# =============================================================================

ndf = 0.1 * T_max                  # force scale  [N]
ndm = 0.1 * m0                     # mass scale   [kg]
ndg = ndf / ndm                    # acceleration [m/s^2]
ndv = ndg                          # velocity     [m/s]
ndp = ndv                          # position     [m]
ndt = 10.0                         # time scale so t_f (~10-80 s) is O(1-8)

# Per-channel scale for the 8-state x = [r(3), v(3), m, t_f]
SCALE_X = np.array([ndp, ndp, ndp, ndv, ndv, ndv, ndm, ndt])

W_FUEL = ndg / (Isp * g0)              # scaled mass loss per ||T~||*dt  (= d(scaled fuel))
W_REG  = 1e-1 * (ndf / 1e6)**2
COST_SCALE = 1.0 / (W_FUEL * (50.0 / (N_STAGES - 1)))   # ~260


# =============================================================================
# 3. DYNAMICS  (scaled, time-dilated to tau in [0,1])
# =============================================================================

def continuous_dynamics(x, u):
    """dx_scaled/dtau = t_f_phys * f_physical(x_phys) / scale"""
    r  = x[0:3] * ndp
    v  = x[3:6] * ndv
    m  = x[6]   * ndm
    tf = x[7]   * ndt                                   # physical t_f [s]
    F  = u * ndf                                        # thrust [N]

    v_norm = casadi.sqrt(casadi.dot(v, v) + 1e-6)
    D      = -0.5 * rho * S_D * C_D * v_norm * v        # drag [N]
    F_norm = casadi.sqrt(casadi.dot(F, F) + 1.0)        # [N], eps in N^2

    r_dot = v
    v_dot = (F + D) / m + casadi.vertcat(0, 0, -g0)
    m_dot = -F_norm / (Isp * g0)                        # Isp absorbs back-pressure

    # scale rates down, dilate by physical t_f; last channel dt_f/dtau = 0
    return tf * casadi.vertcat(r_dot / ndp, v_dot / ndv, m_dot / ndm, 0)


# =============================================================================
# 4. OBJECTIVE  (module-level defaults for the N=61 solver)
# =============================================================================

def obj(z):
    """Dense running cost (stages 1..N-1): fuel (impulse) + L2 thrust
    regulariser.
    """
    T_norm = casadi.sqrt(casadi.dot(z[0:3], z[0:3]) + 1e-9)   # scaled thrust norm
    dt     = (z[10] * ndt) / (N_STAGES - 1)
    return COST_SCALE * (W_FUEL * T_norm * dt + W_REG * T_norm**2 * dt)


def objN(z):
    """Terminal stage carries a dummy control (discarded); no fuel term."""
    return 0.0 * z[9]


# =============================================================================
# 5. PATH INEQUALITIES  (hl <= h(z) <= hu), in scaled coordinates
# =============================================================================

def ineq(z):
    """Path inequalities h(z), in scaled units.

    0: thrust magnitude ||F||        in [T_min, T_max]
    1: tilt  F_z - cos(theta)*||F||  >= 0
    2: glide tan(g)*||r_xy|| - r_z   <= 0
    """
    F = z[0:3]
    r = z[3:6]
    F_norm = casadi.sqrt(casadi.dot(F, F) + 1e-9)
    tilt   = z[2] - cos_tilt * F_norm
    glide  = tgm * casadi.sqrt(r[0]**2 + r[1]**2 + 1e-9) - (r[2] + gs_tol / ndp)
    return casadi.vertcat(F_norm, tilt, glide)


# Bounds on the inequality vector (thrust bounds scaled by ndf):
H_LOWER = np.array([T_min / ndf, 0.0,     -np.inf])
H_UPPER = np.array([T_max / ndf, np.inf,   0.0])


# =============================================================================
# 6. SOLVER GENERATION  (parameterised horizon and t_f bounds)
# =============================================================================

def generate_solver(n_stages=N_STAGES, tf_lo=tf_min, tf_hi=tf_max,
                    name='pdg_expert', t_ref=None):
    """Generate a FORCESPRO solver for the fuel-optimal PDG OCP.

    Parameters
    ----------
    n_stages : int
        Number of collocation stages (= shooting intervals + 1).
        Default 61 (= 60 intervals, matching IPOPT).
    tf_lo, tf_hi : float
        Bounds on the free final time [s].
    name : str
        Solver directory name (also used as CodeOptions tag).
    t_ref : float or None
        Reference time for COST_SCALE calculation [s].
        If None, uses the midpoint (tf_lo + tf_hi) / 2.

    Returns
    -------
    model, solver : FORCESPRO model and compiled solver objects.
    """
    n_int = n_stages - 1

    if t_ref is None:
        t_ref = (tf_lo + tf_hi) / 2.0

    cost_scale = 1.0 / (W_FUEL * (t_ref / n_int))

    # --- Objective closures (capture local n_int and cost_scale) ----------
    def _obj(z):
        T_norm = casadi.sqrt(casadi.dot(z[0:3], z[0:3]) + 1e-9)
        dt     = (z[10] * ndt) / n_int
        return cost_scale * (W_FUEL * T_norm * dt + W_REG * T_norm**2 * dt)

    def _objN(z):
        return 0.0 * z[9]

    # --- Model setup ------------------------------------------------------
    model = forcespro.nlp.SymbolicModel(n_stages)
    model.nvar = NVAR
    model.neq  = NEQ
    model.nh   = NH
    model.npar = 0

    model.objective  = _obj
    model.objectiveN = _objN

    model.continuous_dynamics = continuous_dynamics
    model.E = np.concatenate([np.zeros((NEQ, NVAR - NEQ)), np.eye(NEQ)], axis=1)

    model.ineq = ineq
    model.hl   = H_LOWER
    model.hu   = H_UPPER

    Fbox = 1.2 * T_max / ndf
    model.lb = np.array([-Fbox, -Fbox, -Fbox,
                         -np.inf, -np.inf, 0.0,
                         -np.inf, -np.inf, -np.inf,
                         (m_dry + m_overhead) / ndm,
                         tf_lo / ndt])
    model.ub = np.array([ Fbox,  Fbox,  Fbox,
                          np.inf,  np.inf,  np.inf,
                          np.inf,  np.inf,  np.inf,
                          m0 / ndm,
                          tf_hi / ndt])

    model.xinitidx  = range(3, 10)     # pin [r, v, m]; t_f free
    model.xfinalidx = range(3, 9)      # pin [r, v] = 0; m, t_f free

    codeoptions = forcespro.CodeOptions(name)
    codeoptions.maxit      = 2000
    codeoptions.printlevel = 0
    codeoptions.optlevel   = 3
    codeoptions.cleanup    = False
    codeoptions.timing     = 1
    codeoptions.solvemethod = 'PDIP_NLP'
    codeoptions.nlp.hessian_approximation = 'bfgs'
    codeoptions.nlp.integrator.type  = 'ERK4'
    codeoptions.nlp.integrator.Ts    = 1.0 / n_int
    codeoptions.nlp.integrator.nodes = M_SUBSTEPS

    print(f"Generating solver '{name}': N={n_stages}, "
          f"tf=[{tf_lo:.0f}, {tf_hi:.0f}] s, t_ref={t_ref:.1f} s, "
          f"COST_SCALE={cost_scale:.2f}")

    solver = model.generate_solver(codeoptions)
    return model, solver


# =============================================================================
# 7. INITIAL GUESS
# =============================================================================

def compute_initial_guess(x0_phys, tf_guess, n_stages=N_STAGES):
    """Linear interpolation from the IC to the landing target; returns a FLAT
    1-D scaled guess of length n_stages*NVAR (stage-major).
    """
    r0, v0, m_ic = x0_phys[0:3], x0_phys[3:6], x0_phys[6]
    a1 = np.linspace(1.0, 0.0, n_stages)      # 1 -> 0
    a2 = 1.0 - a1                              # 0 -> 1

    x_guess = np.zeros((n_stages, 7))
    for k in range(n_stages):
        x_guess[k, 0:3] = a1[k] * r0                       # r0 -> 0
        x_guess[k, 3:6] = a1[k] * v0                       # v0 -> 0
        x_guess[k, 6]   = a1[k] * m_ic + a2[k] * 1.1 * m_dry   # m0 -> 1.1 m_dry

    # Gravity-compensating thrust guess (points up), in Newtons
    u_guess = np.tile(np.array([0.0, 0.0, m0 * g0]), (n_stages, 1))

    Z = np.zeros((n_stages, NVAR))
    Z[:, 0:3]  = u_guess / ndf
    Z[:, 3:10] = x_guess / np.array([ndp, ndp, ndp, ndv, ndv, ndv, ndm])
    Z[:, 10]   = tf_guess / ndt
    return Z.reshape(-1)


# =============================================================================
# 8. SOLVE + EXTRACT
# =============================================================================

def solve_ocp(solver, x0_phys, tf_guess=50.0, n_stages=N_STAGES):
    """Solve the OCP and extract the trajectory in physical SI units.

    Parameters
    ----------
    solver : FORCESPRO solver object (loaded or freshly generated).
    x0_phys : array (7,) — physical IC [r, v, m] in SI.
    tf_guess : float — initial guess for free final time [s].
    n_stages : int — number of collocation stages (must match the solver).

    Returns
    -------
    dict with 'x_traj' (n_stages, 8), 'u_traj' (n_stages-1, 3) in MN,
    'tf', 'm_f', 'exitflag', solver diagnostics, and 'n_stages'.
    """
    x0_phys = np.asarray(x0_phys, dtype=float).flatten()

    xinit_scaled = x0_phys / np.array([ndp, ndp, ndp, ndv, ndv, ndv, ndm])
    problem = {
        "x0":     compute_initial_guess(x0_phys, tf_guess, n_stages),
        "xinit":  xinit_scaled,
        "xfinal": np.zeros(6),
    }

    output, exitflag, info = solver.solve(problem)

    x_traj = np.zeros((n_stages, 8))     # physical SI: [r, v, m, t_f]
    u_traj = np.zeros((n_stages - 1, 3)) # MN
    for k in range(n_stages):
        zk = output['x{0:02d}'.format(k + 1)]
        x_traj[k] = zk[3:11] * SCALE_X
        if k < n_stages - 1:
            u_traj[k] = (zk[0:3] * ndf) / 1e6

    tf_opt    = float(x_traj[0, 7])
    m_f       = float(x_traj[-1, 6])
    tf_spread = float(np.max(np.abs(x_traj[:, 7] - tf_opt)))

    return {
        'x_traj': x_traj, 'u_traj': u_traj,
        'tf': tf_opt, 't_f': tf_opt, 'm_f': m_f,
        'exitflag': int(exitflag), 'it': int(info.it),
        'solvetime': float(info.solvetime),
        'res_eq': float(info.res_eq), 'res_ineq': float(info.res_ineq),
        'tf_spread': tf_spread,
        'n_stages': n_stages,          # used downstream for subsampling
    }


# =============================================================================
# 9. SAVE  (dataset.py 6-key schema, physical units)
# =============================================================================

def save_trajectory(filepath, x_traj, u_traj, status, t_f, m_f):
    """Write one solved trajectory in the NPZ schema read by dataset.py."""
    x_traj = np.asarray(x_traj, dtype=np.float64)
    u_traj = np.asarray(u_traj, dtype=np.float64)
    assert x_traj.shape == (N_STAGES, 8)
    assert u_traj.shape == (N_STAGES - 1, 3)
    np.savez(filepath, x_traj=x_traj, u_traj=u_traj, ic=x_traj[0],
             status=int(status), t_f=float(t_f), m_f=float(m_f))


# =============================================================================
# 10. PLOT
# =============================================================================

def plot_trajectory(result, title_suffix=''):
    """Altitude, vertical velocity and thrust magnitude against time."""
    x_traj, u_traj, tf = result['x_traj'], result['u_traj'], result['tf']
    n_st = result.get('n_stages', N_STAGES)
    t_x = np.linspace(0, tf, n_st)
    t_u = np.linspace(0, tf, n_st - 1)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(t_x, x_traj[:, 2], 'b-', lw=2, label='altitude $r_z$')
    axes[0].axhline(0, color='k', lw=0.5); axes[0].set_ylabel('Altitude [m]')
    axes[0].set_title(f'FORCESPRO PDG (scaled) — exitflag {result["exitflag"]}, '
                      f't_f = {tf:.1f} s, m_f = {x_traj[-1,6]/1000:.2f} t{title_suffix}')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(t_x, x_traj[:, 5], 'r-', lw=2, label='$v_z$')
    axes[1].axhline(0, color='k', lw=0.5); axes[1].set_ylabel('Vertical vel [m/s]')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    T_mag = np.linalg.norm(u_traj, axis=1)
    axes[2].step(t_u, T_mag, 'g-', lw=2, where='post', label='$\\|T\\|$ [MN]')
    axes[2].axhline(T_min/1e6, color='gray', ls='--', lw=1, label='T_min')
    axes[2].axhline(T_max/1e6, color='gray', ls='-.', lw=1, label='T_max')
    axes[2].set_ylabel('Thrust [MN]'); axes[2].set_xlabel('Time [s]')
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout(); plt.savefig('forces_pdg.png', dpi=120)
    print("Plot saved to forces_pdg.png"); plt.show()


# =============================================================================
# 11. MAIN
# =============================================================================

def main():
    """Generate the default 61-stage solver and solve the nominal IC."""
    model, solver = generate_solver()

    x0 = np.array([-2386.0, 223.0, 6038.0, 199.0, -18.0, -251.0, m0])
    print("=" * 60)
    print("Solving Guadagnini IC with FORCESPRO (scaled formulation) ...")
    print("=" * 60)

    result = solve_ocp(solver, x0, tf_guess=50.0)
    print(f"  exitflag  : {result['exitflag']}  (1 = locally optimal)")
    print(f"  iters     : {result['it']}")
    print(f"  solvetime : {result['solvetime']*1e3:.1f} ms")
    print(f"  t_f       : {result['tf']:.3f} s  (spread {result['tf_spread']:.2e})")
    print(f"  m_f       : {result['m_f']/1000:.2f} t   (IPOPT ref: 61.49 t)")
    print(f"  res_eq    : {result['res_eq']:.2e}")
    print(f"  res_ineq  : {result['res_ineq']:.2e}")

    if result['exitflag'] == 1:
        save_trajectory('traj_test.npz', result['x_traj'], result['u_traj'],
                        status=0, t_f=result['tf'], m_f=result['m_f'])
        print("  saved -> traj_test.npz")
        plot_trajectory(result, ' (Guadagnini IC)')
    else:
        print("  Non-optimal exit — inspect res_eq / res_ineq.")
        plot_trajectory(result, ' (CHECK)')


if __name__ == '__main__':
    main()
