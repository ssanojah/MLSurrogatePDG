"""
RETALT1 fuel-optimal powered descent guidance: CasADi + IPOPT formulation.

Direct multiple shooting (60 intervals, RK4) of the fuel-optimal powered
descent problem, solved with IPOPT. Also provides the shared constants and
the CasADi dynamics used by other scripts.

Usage:
    python ipopt_pdg.py                  # solve the nominal IC and plot

    from ipopt_pdg import solve_pdg, ParametricPDGSolver
    result = solve_pdg(x0, tf_guess=50.0)
    solver = ParametricPDGSolver()
    result = solver.solve(x0, tf_guess=50.0)
"""

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import time

# =============================================================================
# 1. PHYSICAL PARAMETERS (RETALT1 first stage)
# =============================================================================

g0     = 9.807                          # standard gravity [m/s²]
Isp    = 372.2                          # sea-level specific impulse [s]
m_dry  = 59.3e3                         # dry mass [kg]
m0     = 70.0e3                         # wet mass at burn start [kg]

g_vec  = np.array([0.0, 0.0, -g0])      # gravity vector (ENU)

# Thrust bounds [MN]
T_min_MN = 472e-3                       # 40% throttle, single engine
T_max_MN = 1179e-3                      # 100% throttle, single engine

# Glide-slope constraint
gamma_gs = np.deg2rad(30.0)
sin_gs   = float(np.sin(gamma_gs))   # 0.5
eps_r    = 1e-3                       # regularization for ||r|| at origin

# Tilt constraint (Szmuk 2016 Eq. 33)
theta_max = np.deg2rad(15.0)
cos_tilt  = float(np.cos(theta_max))

# Aerodynamic drag (Szmuk Eq. 1)
rho   = 1.225                           # sea-level air density [kg/m³]
S_D   = np.pi / 4 * 6.0**2             # reference area [m²], body diam = 6 m
C_D   = 1.0                             # drag coefficient (ASSUMED placeholder)

tf_min, tf_max = 10.0, 80.0

TF_MIN_DEFAULT, TF_MAX_DEFAULT = tf_min, tf_max

# Discretization
N = 60                                   # shooting intervals
M_rk4 = 2

nx = 7                                   # state dim: [r(3), v(3), m(1)]
nu = 3                                   # control dim: [T_x, T_y, T_z] in MN


# =============================================================================
# 2. DYNAMICS
# =============================================================================

def build_dynamics_function():
    """Build f(x, u) -> x_dot as a CasADi Function."""
    r   = ca.SX.sym('r', 3)
    v   = ca.SX.sym('v', 3)
    m   = ca.SX.sym('m', 1)
    x   = ca.vertcat(r, v, m)

    T_c    = ca.SX.sym('T_c', 3)          # thrust in MN
    T_phys = 1e6 * T_c                    # convert to N for dynamics

    # Aerodynamic drag
    eps_v  = 1e-6
    v_norm = ca.sqrt(ca.dot(v, v) + eps_v)
    D      = -0.5 * rho * S_D * C_D * v_norm * v

    # Thrust magnitude (for mass flow rate)
    eps_T  = 1e-12
    T_norm = ca.sqrt(ca.dot(T_c, T_c) + eps_T)   # in MN

    # Equations of motion
    r_dot = v
    v_dot = (T_phys + D) / m + g_vec
    m_dot = -(1e6 * T_norm) / (Isp * g0)

    x_dot = ca.vertcat(r_dot, v_dot, m_dot)

    f = ca.Function('f_dynamics', [x, T_c], [x_dot], ['x', 'u'], ['x_dot'])
    return f


def build_rk4_integrator(f):
    """Build one shooting-interval integrator: (x_k, u_k, dt) -> x_{k+1}."""
    x  = ca.SX.sym('x', nx)
    u  = ca.SX.sym('u', nu)
    dt = ca.SX.sym('dt')                  # interval duration = t_f / N

    h  = dt / M_rk4                        # sub-step size
    x_next = x
    for _ in range(M_rk4):
        k1 = f(x_next, u)
        k2 = f(x_next + h/2 * k1, u)
        k3 = f(x_next + h/2 * k2, u)
        k4 = f(x_next + h * k3, u)
        x_next = x_next + h/6 * (k1 + 2*k2 + 2*k3 + k4)

    F = ca.Function('F_rk4', [x, u, dt], [x_next], ['x', 'u', 'dt'], ['x_next'])
    return F


# =============================================================================
# 3. NLP CONSTRUCTION AND SOLVE
# =============================================================================

def solve_pdg(x0_phys, tf_guess=50.0, verbose=True):
    """Solve the fuel-optimal powered descent guidance problem.

    Parameters
    ----------
    x0_phys : array (7,)
        Initial state [r_x, r_y, r_z, v_x, v_y, v_z, m0].
    tf_guess : float
        Initial guess for free final time [s].
    verbose : bool
        If True, print IPOPT output.

    Returns
    -------
    result : dict with keys 'x_traj', 'u_traj', 'tf', 'status', 'sol'
    """
    # Build dynamics and integrator
    f     = build_dynamics_function()
    F_rk4 = build_rk4_integrator(f)

    # ------------------------------------------------------------------
    # Decision variables: w = [X_0, U_0, X_1, U_1, ..., U_{N-1}, X_N, t_f]
    # ------------------------------------------------------------------
    w   = []        # symbolic decision variables
    w0  = []        # initial guess
    lbw = []        # lower bounds
    ubw = []        # upper bounds

    g   = []        # constraints
    lbg = []        # constraint lower bounds
    ubg = []        # constraint upper bounds

    J = 0           # objective

    # Create symbolic variables for states and controls at each node
    X = []          # X[k] is the symbolic state at node k
    U = []          # U[k] is the symbolic control at interval k

    # Free final time — single scalar decision variable
    t_f = ca.SX.sym('t_f')
    dt  = t_f / N   # physical time per shooting interval

    T_z_guess = 1.3 * m0 * g0 / 1e6                     # MN
    v_horiz   = x0_phys[3:5]                              # [v_x, v_y]
    v_h_norm  = np.linalg.norm(v_horiz) + 1e-10
    horiz_dir = -v_horiz / v_h_norm
    T_h_mag   = np.tan(np.deg2rad(10.0)) * T_z_guess     # ~10° tilt
    u_guess   = np.array([horiz_dir[0] * T_h_mag,
                          horiz_dir[1] * T_h_mag,
                          T_z_guess])

    # Clip control guess to thrust bounds
    u_guess_norm = np.linalg.norm(u_guess)
    if u_guess_norm > T_max_MN:
        u_guess = u_guess * T_max_MN / u_guess_norm
    if u_guess_norm < T_min_MN:
        u_guess = u_guess * T_min_MN / u_guess_norm

    # State initial guess: ballistic Euler forward propagation
    dt_guess = tf_guess / N
    x_guess_traj = np.zeros((N + 1, nx))
    x_guess_traj[0] = x0_phys.copy()
    for k in range(N):
        rk = x_guess_traj[k, 0:3]
        vk = x_guess_traj[k, 3:6]
        mk = x_guess_traj[k, 6]
        a_thrust = u_guess * 1e6 / mk
        a_grav   = g_vec
        v_next   = vk + (a_thrust + a_grav) * dt_guess
        r_next   = rk + vk * dt_guess
        m_next   = mk - np.linalg.norm(u_guess) * 1e6 / (Isp * g0) * dt_guess
        x_guess_traj[k+1, 0:3] = r_next
        x_guess_traj[k+1, 3:6] = v_next
        x_guess_traj[k+1, 6]   = max(m_next, m_dry)

    # ------------------------------------------------------------------
    # Populate decision variables, bounds, and initial guess
    # ------------------------------------------------------------------
    for k in range(N + 1):
        # State at node k
        X_k = ca.SX.sym(f'X_{k}', nx)
        X.append(X_k)
        w.append(X_k)

        # State bounds: only mass is bounded, positions/velocities are free
        lbw += [-1e10, -1e10, 0.0,        # r: effectively unbounded
                -1e10, -1e10, -1e10,         # v: effectively unbounded
                 m_dry]                       # m >= m_dry
        ubw += [ 1e10,  1e10,  1e10,
                 1e10,  1e10,  1e10,
                 m0]                          # m <= m0

        # Initial guess for state
        w0 += x_guess_traj[k].tolist()

        # Control at interval k (only for k < N)
        if k < N:
            U_k = ca.SX.sym(f'U_{k}', nu)
            U.append(U_k)
            w.append(U_k)

            # Control bounds: generous box (path constraints enforce thrust mag)
            lbw += [-2.0, -2.0, -2.0]
            ubw += [ 2.0,  2.0,  2.0]

            # Initial guess for control
            w0 += u_guess.tolist()

    # t_f as last decision variable
    w.append(t_f)
    lbw.append(tf_min)
    ubw.append(tf_max)
    w0.append(tf_guess)

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    # (A) Initial condition: X_0 = x0_phys
    g.append(X[0] - x0_phys)
    lbg += [0.0] * nx
    ubg += [0.0] * nx

    # (B) Dynamics gap-closing: X_{k+1} = F_rk4(X_k, U_k, dt)
    for k in range(N):
        x_next_k = F_rk4(X[k], U[k], dt)
        g.append(X[k+1] - x_next_k)
        lbg += [0.0] * nx
        ubg += [0.0] * nx

    # (C) Terminal constraints: r_f = 0, v_f = 0  (hard equality)
    g.append(X[N][:6])                   # [r_x, r_y, r_z, v_x, v_y, v_z] = 0
    lbg += [0.0] * 6
    ubg += [0.0] * 6

    # (D) Path constraints at each shooting interval
    eps_T = 1e-12
    for k in range(N):
        T_k      = U[k]
        T_norm_k = ca.sqrt(ca.dot(T_k, T_k) + eps_T)

        # Thrust magnitude: T_min <= ||T|| <= T_max
        g.append(T_norm_k)
        lbg.append(float(T_min_MN))
        ubg.append(float(T_max_MN))

        # Tilt constraint: T_z - cos(θ_max) · ||T|| >= 0
        h_tilt_k = T_k[2] - cos_tilt * T_norm_k
        g.append(h_tilt_k)
        lbg.append(0.0)
        ubg.append(1e10)

        cos2_gs = float(np.cos(gamma_gs)**2)
        sin2_gs = float(np.sin(gamma_gs)**2)
        r_k = X[k][:3]
        h_gs_k = cos2_gs * r_k[2]**2 - sin2_gs * (r_k[0]**2 + r_k[1]**2)
        g.append(h_gs_k)
        lbg.append(0.0)
        ubg.append(1e10)

    # ------------------------------------------------------------------
    # Cost function
    # ------------------------------------------------------------------
    J += -X[N][6] / m0

    # Small stage regularizer on thrust (helps with bang-bang switching)
    eps_cost = 1e-3
    for k in range(N):
        J += eps_cost * ca.dot(U[k], U[k]) * (dt)

    # ------------------------------------------------------------------
    # Assemble and solve
    # ------------------------------------------------------------------
    w_cat = ca.vertcat(*w)
    g_cat = ca.vertcat(*g)

    nlp = {'x': w_cat, 'f': J, 'g': g_cat}

    opts = {
        'ipopt.max_iter': 3000,
        'ipopt.tol': 1e-6,
        'ipopt.acceptable_tol': 1e-4,
        'ipopt.print_level': 5 if verbose else 0,
        'print_time': verbose,
    }

    solver = ca.nlpsol('pdg_solver', 'ipopt', nlp, opts)

    t_start = time.time()
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    t_solve = time.time() - t_start

    # ------------------------------------------------------------------
    # Extract solution
    # ------------------------------------------------------------------
    w_opt = sol['x'].full().flatten()
    stats = solver.stats()
    ipopt_status = stats['return_status']

    # Parse: [X_0(7), U_0(3), X_1(7), U_1(3), ..., U_{N-1}(3), X_N(7), t_f(1)]
    x_traj = np.zeros((N + 1, nx))
    u_traj = np.zeros((N, nu))

    idx = 0
    for k in range(N + 1):
        x_traj[k] = w_opt[idx:idx + nx]
        idx += nx
        if k < N:
            u_traj[k] = w_opt[idx:idx + nu]
            idx += nu
    tf_opt = w_opt[idx]

    if verbose:
        print(f"\n{'='*60}")
        print(f"IPOPT status: {ipopt_status}")
        print(f"Solve time:   {t_solve:.3f} s")
        print(f"Final time:   t_f = {tf_opt:.3f} s")
        print(f"Terminal mass: {x_traj[-1, 6]/1000:.2f} t")
        print(f"  (fuel used: {(m0 - x_traj[-1, 6])/1000:.2f} t)")
        print(f"Terminal pos:  {x_traj[-1, :3]}")
        print(f"Terminal vel:  {x_traj[-1, 3:6]}")
        print(f"{'='*60}")

        # Per-node diagnostics
        T_mags = np.linalg.norm(u_traj, axis=1) * 1e3   # kN
        print(f"\nThrust: min = {T_mags.min():.1f} kN, max = {T_mags.max():.1f} kN")
        print(f"  (bounds: [{T_min_MN*1e3:.1f}, {T_max_MN*1e3:.1f}] kN)")

        tilt_angles = np.zeros(N)
        for k in range(N):
            T_vec = u_traj[k]
            T_mag = np.linalg.norm(T_vec)
            if T_mag > 1e-10:
                cos_a = T_vec[2] / T_mag
                tilt_angles[k] = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
        print(f"Tilt:   min = {tilt_angles.min():.2f}°, max = {tilt_angles.max():.2f}°")
        print(f"  (limit: {np.degrees(theta_max):.1f}°)")

    result = {
        'x_traj': x_traj,
        'u_traj': u_traj,
        'tf': tf_opt,
        't_f': tf_opt,               # alias for consistency with other scripts
        'm_f': float(x_traj[-1, 6]), # terminal mass [kg]
        'status': ipopt_status,
        'solve_time': t_solve,
        'sol': sol,
    }
    return result


# =============================================================================
# 3b. PARAMETRIC NLP SOLVER (for DAgger expert queries)
# =============================================================================

class ParametricPDGSolver:
    """Parametric NLP solver for fuel-optimal powered descent guidance.

    Parameters
    ----------
    print_level : int
        IPOPT verbosity (0 = silent, 5 = full output).
        Set at construction time — cannot change between solves.
        For per-solve verbosity, use solve_pdg().
    tf_min : float, optional
        Lower bound on the free final time [s]. Defaults to the
        module-level value (10 s) used for dataset generation.
        DAgger expert instances should pass a lower floor (~3 s):
        at late-trajectory query states the natural remaining flight
        time drops below 10 s, and an ACTIVE tf lower bound forces the
        expert to return a time-dilated (lower-thrust) label that is
        not the optimal continuation (violates Bellman's principle of
        optimality for the subproblem).
    tf_max : float, optional
        Upper bound on the free final time [s]. Defaults to the
        module-level value (80 s).
    """

    # IPOPT status strings that count as convergence
    SUCCESS_STATUSES = {'Solve_Succeeded', 'Solved_To_Acceptable_Level'}

    def __init__(self, print_level=0, tf_min=None, tf_max=None):
        t_build_start = time.time()

        # ---- Per-instance free-final-time bounds ----
        self._tf_min = float(tf_min) if tf_min is not None else TF_MIN_DEFAULT
        self._tf_max = float(tf_max) if tf_max is not None else TF_MAX_DEFAULT
        if not (0.0 < self._tf_min < self._tf_max):
            raise ValueError(
                f"Invalid t_f bounds: [{self._tf_min}, {self._tf_max}] "
                f"(need 0 < tf_min < tf_max)")

        # Build dynamics and integrator (once)
        self._f = build_dynamics_function()
        self._F_rk4 = build_rk4_integrator(self._f)

        # ---- Parametric initial condition ----
        self._p_x0 = ca.SX.sym('p_x0', nx)

        # ---- Build NLP structure (once) ----
        self._build_nlp(print_level)

        self._build_time = time.time() - t_build_start

    def _build_nlp(self, print_level):
        """Construct the NLP with parametric x0."""
        F_rk4 = self._F_rk4

        # Decision variables and bounds (structural — same for every IC)
        w   = []
        lbw = []
        ubw = []

        g   = []
        lbg = []
        ubg = []

        J = 0

        X = []
        U = []

        # Free final time — symbolic decision variable
        t_f = ca.SX.sym('t_f')
        dt  = t_f / N

        # Variables and bounds at each shooting node
        for k in range(N + 1):
            X_k = ca.SX.sym(f'X_{k}', nx)
            X.append(X_k)
            w.append(X_k)

            # State bounds
            lbw += [-1e10, -1e10, 0.0,
                    -1e10, -1e10, -1e10,
                     m_dry]
            ubw += [ 1e10,  1e10,  1e10,
                     1e10,  1e10,  1e10,
                     m0]

            if k < N:
                U_k = ca.SX.sym(f'U_{k}', nu)
                U.append(U_k)
                w.append(U_k)

                lbw += [-2.0, -2.0, -2.0]
                ubw += [ 2.0,  2.0,  2.0]

        # t_f as last decision variable (per-instance bounds)
        w.append(t_f)
        lbw.append(self._tf_min)
        ubw.append(self._tf_max)

        # ---- Constraints ----

        # (A) Initial condition: X[0] = p_x0  (parametric)
        g.append(X[0] - self._p_x0)
        lbg += [0.0] * nx
        ubg += [0.0] * nx

        # (B) Dynamics gap-closing
        for k in range(N):
            x_next_k = F_rk4(X[k], U[k], dt)
            g.append(X[k + 1] - x_next_k)
            lbg += [0.0] * nx
            ubg += [0.0] * nx

        # (C) Terminal constraints
        g.append(X[N][:6])
        lbg += [0.0] * 6
        ubg += [0.0] * 6

        # (D) Path constraints at each interval
        eps_T = 1e-12
        cos2_gs = float(np.cos(gamma_gs)**2)
        sin2_gs = float(np.sin(gamma_gs)**2)

        for k in range(N):
            T_k      = U[k]
            T_norm_k = ca.sqrt(ca.dot(T_k, T_k) + eps_T)

            # Thrust magnitude
            g.append(T_norm_k)
            lbg.append(float(T_min_MN))
            ubg.append(float(T_max_MN))

            # Tilt constraint
            h_tilt_k = T_k[2] - cos_tilt * T_norm_k
            g.append(h_tilt_k)
            lbg.append(0.0)
            ubg.append(1e10)

            # Glide-slope
            r_k = X[k][:3]
            h_gs_k = (cos2_gs * r_k[2]**2
                      - sin2_gs * (r_k[0]**2 + r_k[1]**2))
            g.append(h_gs_k)
            lbg.append(0.0)
            ubg.append(1e10)

        # ---- Cost function ----
        J += -X[N][6] / m0

        eps_cost = 1e-3
        for k in range(N):
            J += eps_cost * ca.dot(U[k], U[k]) * dt

        # ---- Assemble NLP with parameter ----
        w_cat = ca.vertcat(*w)
        g_cat = ca.vertcat(*g)

        nlp = {'x': w_cat, 'f': J, 'g': g_cat, 'p': self._p_x0}

        opts = {
            'ipopt.max_iter': 3000,
            'ipopt.tol': 1e-6,
            'ipopt.acceptable_tol': 1e-4,
            'ipopt.print_level': print_level,
            'print_time': False,
        }

        self._solver = ca.nlpsol('pdg_param', 'ipopt', nlp, opts)

        # Store bounds (they never change)
        self._lbw = lbw
        self._ubw = ubw
        self._lbg = lbg
        self._ubg = ubg

    def _compute_initial_guess(self, x0_phys, tf_guess):
        """Physics-informed initial guess for IPOPT."""
        # Control guess
        T_z_guess = 1.3 * m0 * g0 / 1e6
        v_horiz   = x0_phys[3:5]
        v_h_norm  = np.linalg.norm(v_horiz) + 1e-10
        horiz_dir = -v_horiz / v_h_norm
        T_h_mag   = np.tan(np.deg2rad(10.0)) * T_z_guess
        u_guess   = np.array([horiz_dir[0] * T_h_mag,
                              horiz_dir[1] * T_h_mag,
                              T_z_guess])

        u_guess_norm = np.linalg.norm(u_guess)
        if u_guess_norm > T_max_MN:
            u_guess = u_guess * T_max_MN / u_guess_norm
        if u_guess_norm < T_min_MN:
            u_guess = u_guess * T_min_MN / u_guess_norm

        # Ballistic Euler forward propagation
        dt_guess = tf_guess / N
        x_guess_traj = np.zeros((N + 1, nx))
        x_guess_traj[0] = x0_phys.copy()
        for k in range(N):
            rk = x_guess_traj[k, 0:3]
            vk = x_guess_traj[k, 3:6]
            mk = x_guess_traj[k, 6]
            a_thrust = u_guess * 1e6 / mk
            a_grav   = g_vec
            v_next   = vk + (a_thrust + a_grav) * dt_guess
            r_next   = rk + vk * dt_guess
            m_next   = mk - (np.linalg.norm(u_guess) * 1e6
                             / (Isp * g0) * dt_guess)
            x_guess_traj[k + 1, 0:3] = r_next
            x_guess_traj[k + 1, 3:6] = v_next
            x_guess_traj[k + 1, 6]   = max(m_next, m_dry)

        # Build w0 vector: [X_0, U_0, X_1, U_1, ..., X_N, t_f]
        w0 = []
        for k in range(N + 1):
            w0 += x_guess_traj[k].tolist()
            if k < N:
                w0 += u_guess.tolist()
        w0.append(tf_guess)

        return w0

    def solve(self, x0_phys, tf_guess=50.0, verbose=False):
        """Solve the OCP from a new initial condition.

        Parameters
        ----------
        x0_phys : array (7,)
            Initial state [r_x, r_y, r_z, v_x, v_y, v_z, m].
        tf_guess : float
            Initial guess for free final time [s].
        verbose : bool
            If True, print solution summary (IPOPT output is
            controlled by print_level at construction time).

        Returns
        -------
        dict — same format as solve_pdg(), with keys:
            'x_traj', 'u_traj', 'tf', 't_f', 'm_f',
            'status', 'solve_time', 'sol'
        """
        x0_phys = np.asarray(x0_phys, dtype=float).flatten()

        # Compute initial guess (cheap, ~0.1 ms)
        w0 = self._compute_initial_guess(x0_phys, tf_guess)

        # Solve with parametric x0 (the expensive part)
        t_start = time.time()
        sol = self._solver(
            x0=w0,
            lbx=self._lbw,
            ubx=self._ubw,
            lbg=self._lbg,
            ubg=self._ubg,
            p=x0_phys,       # <-- the parameter value
        )
        t_solve = time.time() - t_start

        # Extract solution (same parsing as solve_pdg)
        w_opt = sol['x'].full().flatten()
        stats = self._solver.stats()
        ipopt_status = stats['return_status']

        x_traj = np.zeros((N + 1, nx))
        u_traj = np.zeros((N, nu))

        idx = 0
        for k in range(N + 1):
            x_traj[k] = w_opt[idx:idx + nx]
            idx += nx
            if k < N:
                u_traj[k] = w_opt[idx:idx + nu]
                idx += nu
        tf_opt = float(w_opt[idx])

        if verbose:
            print(f"  IPOPT: {ipopt_status}, t_f={tf_opt:.2f}s, "
                  f"m_f={x_traj[-1, 6]/1000:.2f}t, "
                  f"solve={t_solve:.2f}s")

        return {
            'x_traj': x_traj,
            'u_traj': u_traj,
            'tf': tf_opt,
            't_f': tf_opt,
            'm_f': float(x_traj[-1, 6]),
            'status': ipopt_status,
            'solve_time': t_solve,
            'sol': sol,
        }

    @property
    def build_time(self):
        """Time spent building the NLP at construction [s]."""
        return self._build_time

    def __repr__(self):
        return (f"ParametricPDGSolver(build_time={self._build_time:.2f}s, "
                f"N={N}, nx={nx}, nu={nu}, "
                f"tf_bounds=[{self._tf_min:.1f}, {self._tf_max:.1f}] s)")


# =============================================================================
# 4. PLOTTING
# =============================================================================

def plot_trajectory(result, title_suffix=''):
    """Altitude, vertical velocity and thrust magnitude, plus a 3D view."""
    x_traj = result['x_traj']
    u_traj = result['u_traj']
    tf     = result['tf']

    t_x = np.linspace(0, tf, N + 1)
    t_u = np.linspace(0, tf, N)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Altitude
    axes[0].plot(t_x, x_traj[:, 2], 'b-', lw=2, label='altitude $r_z$')
    axes[0].axhline(0, color='k', lw=0.5)
    axes[0].set_ylabel('Altitude [m]')
    axes[0].set_title(f'CasADi + IPOPT PDG — {result["status"]}, '
                      f't_f = {tf:.1f} s, '
                      f'm_f = {x_traj[-1,6]/1000:.2f} t'
                      f'{title_suffix}')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Vertical velocity
    axes[1].plot(t_x, x_traj[:, 5], 'r-', lw=2, label='$v_z$')
    axes[1].axhline(0, color='k', lw=0.5)
    axes[1].set_ylabel('Vertical velocity [m/s]')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Thrust magnitude
    T_mag = np.linalg.norm(u_traj, axis=1)
    axes[2].step(t_u, T_mag, 'g-', lw=2, where='post', label='$\\|T_c\\|$ [MN]')
    axes[2].axhline(T_min_MN, color='gray', ls='--', lw=1, label=f'T_min = {T_min_MN*1e3:.0f} kN')
    axes[2].axhline(T_max_MN, color='gray', ls='-.', lw=1, label=f'T_max = {T_max_MN*1e3:.0f} kN')
    axes[2].set_ylabel('Thrust [MN]')
    axes[2].set_xlabel('Time [s]')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('ipopt_pdg.png', dpi=120)
    print("Plot saved to ipopt_pdg.png")
    plt.show()

    # 3D trajectory
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.plot3D(x_traj[:, 0], x_traj[:, 1], x_traj[:, 2], 'b-', lw=2)
    ax3d.set_xlabel('East [m]')
    ax3d.set_ylabel('North [m]')
    ax3d.set_zlabel('Up [m]')
    ax3d.set_title('3D Trajectory')
    plt.show()


# =============================================================================
# 5. MAIN: solve the nominal (Guadagnini) IC
# =============================================================================

def main():

    # ---- Realistic Guadagnini IC ----
    print("\n" + "=" * 60)
    print("TEST 2: Realistic IC  [-2386, 223, 6038] m, [199, -18, -251] m/s")
    print("=" * 60)
    x0_real = np.array([-2386.0, 223.0, 6038.0,  199.0, -18.0, -251.0,  m0])
    result_real = solve_pdg(x0_real, tf_guess=50.0)

    if result_real['status'] == 'Solve_Succeeded':
        print("\n>>> Realistic IC PASSED!")
        print(x0_real)
        plot_trajectory(result_real, title_suffix=' (Guadagnini IC)')
    else:
        print(f"\n>>> Realistic IC result: {result_real['status']}")
        plot_trajectory(result_real, title_suffix=' (Guadagnini IC - check)')


if __name__ == '__main__':
    main()
