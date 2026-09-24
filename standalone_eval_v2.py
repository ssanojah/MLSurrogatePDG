#!/usr/bin/env python3
"""
Standalone policy evaluation (open loop and closed loop).

Evaluates a trained PDGTransformer or MLP on the test split of a dataset.
Closed-loop rollouts use the nonlinear dynamics with thrust magnitude and
tilt enforcement; each IC is classified as soft, hard, crash, airborne,
fuel exhausted or diverged. Writes a JSON summary, per-IC CSV and figures.

Usage:
    python standalone_eval_v2.py --run_dir runs/<run> \\
        --data_dir data/<dataset> --mode closed_loop --ttg

    python standalone_eval_v2.py --run_dir runs/<run> \\
        --traj_file data/<dataset>/traj_00042.npz --mode both --ttg
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import casadi as ca
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d')
import numpy as np
import torch
import torch.nn as nn


# =====================================================================
# 1. PHYSICAL CONSTANTS (RETALT1 first stage, same values as the expert OCP)
# =====================================================================

g0       = 9.807                         # standard gravity [m/s^2]
Isp      = 372.2                         # sea-level specific impulse [s]
m_dry    = 59.3e3                        # dry mass [kg]
m0       = 70.0e3                        # wet mass at burn start [kg]
g_vec    = np.array([0.0, 0.0, -g0])     # gravity in ENU frame

# Thrust bounds [MN] — single Vulcain-derived engine
T_min_MN = 472e-3                        # 40 % throttle
T_max_MN = 1179e-3                       # 100 % throttle

# Glide-slope constraint
gamma_gs   = np.deg2rad(30.0)

# Tilt constraint
theta_max  = np.deg2rad(15.0)

# Aerodynamic drag
rho = 1.225                              # sea-level air density [kg/m^3]
S_D = np.pi / 4 * 6.0**2                 # reference area, body diam 6 m
C_D = 1.0                                # drag coefficient (placeholder)

# Discretisation
N      = 60                              # shooting intervals
M_rk4  = 2                               # RK4 sub-steps per interval
nx     = 7                               # state:   [r(3), v(3), m]
nu     = 3                               # control: [T_x, T_y, T_z] in MN


# =====================================================================
# 2. DYNAMICS — CasADi symbolic RK4
# =====================================================================

def build_dynamics_function():
    """Build f(x, u) -> xdot as a CasADi Function."""
    r   = ca.SX.sym("r", 3)
    v   = ca.SX.sym("v", 3)
    m   = ca.SX.sym("m", 1)
    x   = ca.vertcat(r, v, m)
    T_c = ca.SX.sym("T_c", 3)               # thrust in MN
    T_phys = 1e6 * T_c                      # -> Newtons

    # Aerodynamic drag
    eps_v  = 1e-6
    v_norm = ca.sqrt(ca.dot(v, v) + eps_v)
    D      = -0.5 * rho * S_D * C_D * v_norm * v

    # Thrust magnitude for mass-flow rate
    eps_T  = 1e-12
    T_norm = ca.sqrt(ca.dot(T_c, T_c) + eps_T)   # MN

    r_dot = v
    v_dot = (T_phys + D) / m + g_vec
    m_dot = -(1e6 * T_norm) / (Isp * g0)

    x_dot = ca.vertcat(r_dot, v_dot, m_dot)
    return ca.Function("f_dyn", [x, T_c], [x_dot], ["x", "u"], ["x_dot"])


def build_rk4_integrator(f_dyn):
    """Build (x_k, u_k, dt) -> x_{k+1} with M_rk4 RK4 sub-steps."""
    x  = ca.SX.sym("x", nx)
    u  = ca.SX.sym("u", nu)
    dt = ca.SX.sym("dt")
    h  = dt / M_rk4
    x_next = x
    for _ in range(M_rk4):
        k1 = f_dyn(x_next, u)
        k2 = f_dyn(x_next + h / 2 * k1, u)
        k3 = f_dyn(x_next + h / 2 * k2, u)
        k4 = f_dyn(x_next + h * k3, u)
        x_next = x_next + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return ca.Function("F_rk4", [x, u, dt], [x_next],
                       ["x", "u", "dt"], ["x_next"])


# Build once at module level — stateless CasADi graphs.
_f_dyn = build_dynamics_function()
_F_rk4 = build_rk4_integrator(_f_dyn)


# =====================================================================
# 3. NORMALISATION (standalone, mirrors dataset.NormalizationStats)
# =====================================================================

class NormStats:
    """Per-channel mean/std for states (8-dim) and actions (3-dim)."""

    def __init__(self, state_mean, state_std, action_mean, action_std):
        self.s_mu  = np.asarray(state_mean,  dtype=np.float32)
        self.s_sig = np.asarray(state_std,   dtype=np.float32)
        self.a_mu  = np.asarray(action_mean, dtype=np.float32)
        self.a_sig = np.asarray(action_std,  dtype=np.float32)

    def norm_state(self, x):
        """(x - mu) / sigma — works on any shape (..., 8)."""
        return (x - self.s_mu) / self.s_sig

    def norm_action(self, u):
        return (u - self.a_mu) / self.a_sig

    def unnorm_action(self, u_n):
        return u_n * self.a_sig + self.a_mu

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["state_mean"], d["state_std"],
                   d["action_mean"], d["action_std"])

    def summary(self):
        names_s = ["r_x", "r_y", "r_z", "v_x", "v_y", "v_z", "m", "t_f"]
        names_a = ["T_cx", "T_cy", "T_cz"]
        lines = ["Normalisation statistics", "=" * 50, "", "States:"]
        for i, n in enumerate(names_s):
            lines.append(f"  {n:>4s}: mean={self.s_mu[i]:12.4f}  "
                         f"std={self.s_sig[i]:12.4f}")
        lines.append("\nActions:")
        for i, n in enumerate(names_a):
            lines.append(f"  {n:>4s}: mean={self.a_mu[i]:12.4f}  "
                         f"std={self.a_sig[i]:12.4f}")
        return "\n".join(lines)


# =====================================================================
# 4. MODELS (standalone mirrors of model.py / mlp_model.py)
# =====================================================================

@dataclass
class TConfig:
    state_dim:   int = 8
    action_dim:  int = 3
    d_model:     int = 64
    nhead:       int = 4
    num_layers:  int = 2
    d_ff:        int = 128
    dropout:     float = 0.1
    max_seq_len: int = 60

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))


class PDGTransformer(nn.Module):
    """Causal encoder-only transformer: (B, N, 8) -> (B, N, 3)."""

    def __init__(self, cfg: Optional[TConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = TConfig()
        self.cfg = cfg
        c = cfg

        self.input_proj = nn.Linear(c.state_dim, c.d_model)
        self.pos_embedding = nn.Embedding(c.max_seq_len, c.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=c.d_model, nhead=c.nhead,
            dim_feedforward=c.d_ff, dropout=c.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer,
                                                         c.num_layers)
        self.final_norm = nn.LayerNorm(c.d_model)
        self.output_head = nn.Linear(c.d_model, c.action_dim)

    def forward(self, states, mask=None):
        B, S, _ = states.shape
        x = self.input_proj(states) + self.pos_embedding(
            torch.arange(S, device=states.device))
        if mask is None:
            mask = torch.triu(
                torch.full((S, S), float("-inf"), device=states.device),
                diagonal=1)
        x = self.transformer_encoder(x, mask=mask)
        return self.output_head(self.final_norm(x))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@dataclass
class MLPConfig:
    state_dim:   int = 8
    action_dim:  int = 3
    hidden_dims: list = field(default_factory=lambda: [256, 256, 128])
    activation:  str = "gelu"
    dropout:     float = 0.0

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(**d)


class PDGMLP(nn.Module):
    """State-feedback MLP: (B, 8) -> (B, 3) or (B, N, 8) -> (B, N, 3)."""

    def __init__(self, cfg: Optional[MLPConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = MLPConfig()
        self.cfg = cfg

        act_fn = nn.GELU if cfg.activation == "gelu" else nn.ReLU
        layers = []
        in_dim = cfg.state_dim
        for h_dim in cfg.hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(act_fn())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, cfg.action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, mask=None):
        """`mask` accepted for interface compatibility; ignored."""
        if x.dim() == 3:
            B, S, D = x.shape
            return self.net(x.reshape(B * S, D)).reshape(B, S, -1)
        return self.net(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PDGMLPLegacy(nn.Module):
    """MLP as defined in mlp_model.py (Li & Wang 2025): Input(8) ->
    [Linear(h)+ReLU] x L -> Linear(3).
    """

    def __init__(self, state_dim=8, action_dim=3, hidden_dim=256,
                 num_hidden_layers=6, activation="relu", dropout=0.0):
        super().__init__()
        act_fn = nn.GELU if activation == "gelu" else nn.ReLU
        layers, in_dim = [], state_dim
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(in_dim, hidden_dim), act_fn()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.hidden = nn.Sequential(*layers)
        self.output_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, mask=None):
        if x.dim() == 3:
            B, S, D = x.shape
            return self.forward(x.reshape(B * S, D)).reshape(B, S, -1)
        return self.output_head(self.hidden(x))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =====================================================================
# 5. TRAJECTORY LOADING & DATA SPLIT
# =====================================================================

def load_traj(path):
    """Load one expert NPZ -> dict."""
    d = np.load(path, allow_pickle=False)
    return dict(
        x_traj = d["x_traj"],         # (N+1, 8)
        u_traj = d["u_traj"],         # (N,   3)
        ic     = d["ic"],             # (8,)
        status = int(d["status"]),
        t_f    = float(d["t_f"]),
        m_f    = float(d["m_f"]),
    )


def split_by_ic(files, train_frac=0.70, val_frac=0.15, seed=42):
    """Deterministic IC-level split -> (train, val, test) file lists."""
    files = [Path(f) for f in files]
    n = len(files)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_tr = max(1, int(n * train_frac))
    n_va = max(1, int(n * val_frac))
    return ([files[i] for i in idx[:n_tr]],
            [files[i] for i in idx[n_tr:n_tr + n_va]],
            [files[i] for i in idx[n_tr + n_va:]])


# =====================================================================
# 6. LANDING CLASSIFICATION
# =====================================================================

CATEGORIES = ["soft", "hard", "crash", "airborne",
              "fuel_exhausted", "diverged"]

CAT_COLORS = {
    "soft":           "tab:green",
    "hard":           "tab:orange",
    "crash":          "tab:red",
    "airborne":       "tab:blue",
    "fuel_exhausted": "tab:purple",
    "diverged":       "tab:gray",
}

CAT_LABELS = {
    "soft":           "Soft",
    "hard":           "Hard",
    "crash":          "Crash",
    "airborne":       "Airborne at horizon",
    "fuel_exhausted": "Fuel exhausted",
    "diverged":       "Diverged",
}


@dataclass
class LandingThresholds:
    """All numeric thresholds in one place — traceable to sources."""
    # Soft box (Carradori GR-04/05/06)
    soft_rh_m:        float = 10.0    # ||r_xy|| at touchdown        (GR-04)
    soft_vz_ms:       float = 2.0     # |v_z|    at touchdown        (GR-05)
    soft_vh_axis_ms:  float = 1.5     # |v_x|, |v_y| PER AXIS        (GR-06)

    # Hard / survivable box (Rosa et al. 2023 via Carradori Ch. 8)
    hard_vz_ms:       float = 10.0    # |v_z|    at touchdown
    hard_vh_norm_ms:  float = 3.0     # ||v_xy|| at touchdown (NORM, not axis)

    # Actuation limits (mirror the expert OCP; used for plot guide lines)
    theta_max_rad:    float = np.deg2rad(15.0)
    T_min_MN:         float = 472e-3
    T_max_MN:         float = 1179e-3

    # Touchdown detection
    grazing_alt_m:    float = 1.0     # horizon-end r_z below this -> touchdown
    bisect_tol_m:     float = 1e-6
    bisect_max_iter:  int   = 60

    # Rollout safety
    max_pos_m:        float = 50_000  # terminate if ||r|| > this


def find_touchdown(x_traj, u_traj, tf, thres: LandingThresholds):
    """Scan a closed-loop trajectory for the first ground crossing and bisect
    to find the exact touchdown state.

    Returns
    -------
    td_state : (7,) state at touchdown, or None
    td_time  : float time of touchdown [s], or None
    td_type  : "bisection" | "grazing" | "none"
    """
    dt = tf / N
    K = u_traj.shape[0]

    for k in range(K):
        rz_k   = x_traj[k, 2]
        rz_kp1 = x_traj[k + 1, 2]

        if rz_k >= 0 and rz_kp1 < 0:
            t_lo, t_hi = 0.0, dt
            x_lo = x_traj[k].copy()

            for _ in range(thres.bisect_max_iter):
                t_mid = 0.5 * (t_lo + t_hi)
                x_mid = np.array(_F_rk4(x_lo, u_traj[k], t_mid)).flatten()
                if x_mid[2] > 0:
                    t_lo = t_mid
                else:
                    t_hi = t_mid
                if (t_hi - t_lo) < thres.bisect_tol_m / max(
                        1.0, abs(x_traj[k, 5])):
                    break

            t_td = 0.5 * (t_lo + t_hi)
            x_td = np.array(_F_rk4(x_lo, u_traj[k], t_td)).flatten()
            return x_td, k * dt + t_td, "bisection"

    # No zero-crossing: check grazing at horizon end
    rz_final = x_traj[-1, 2]
    if 0 <= rz_final <= thres.grazing_alt_m:
        return x_traj[-1].copy(), K * dt, "grazing"

    return None, None, "none"


def classify_landing(td_state, thres: LandingThresholds):
    """Classify a touchdown state into soft / hard / crash."""
    r_h = np.linalg.norm(td_state[:2])
    vz  = abs(td_state[5])
    vx  = abs(td_state[3])
    vy  = abs(td_state[4])
    vh  = np.linalg.norm(td_state[3:5])

    details = dict(r_h=r_h, r_z=td_state[2], vz_signed=float(td_state[5]), 
                   vx=vx, vy=vy, vh=vh,
                   m_f=td_state[6])

    if (r_h <= thres.soft_rh_m and
            vz <= thres.soft_vz_ms and
            vx <= thres.soft_vh_axis_ms and
            vy <= thres.soft_vh_axis_ms):
        return "soft", details

    if vz <= thres.hard_vz_ms and vh <= thres.hard_vh_norm_ms:
        return "hard", details

    return "crash", details


# =====================================================================
# 7. THRUST ENFORCEMENT — magnitude bounds AND tilt cone
# =====================================================================

def enforce_thrust(u_raw, clip_tilt=True):
    """Project a raw network thrust command onto the admissible set.

    Parameters
    ----------
    u_raw     : (3,) raw thrust command [MN]
    clip_tilt : if False, only the magnitude bound is applied (used to
                quantify the contribution of the tilt projection)
    """
    u = np.asarray(u_raw, dtype=float).copy()
    T_mag = np.linalg.norm(u)

    if T_mag < 1e-12:
        return np.array([0.0, 0.0, T_min_MN])

    # --- Magnitude bounds (direction-preserving) ---
    if T_mag > T_max_MN:
        u *= (T_max_MN / T_mag)
        T_mag = T_max_MN
    elif T_mag < T_min_MN:
        u *= (T_min_MN / T_mag)
        T_mag = T_min_MN

    if not clip_tilt:
        return u

    # --- Tilt cone projection (magnitude- and azimuth-preserving) ---
    T_h = np.linalg.norm(u[:2])
    tilt = np.arctan2(T_h, u[2])          # handles u_z < 0 correctly

    if tilt <= theta_max:
        return u

    if T_h < 1e-12:
        # Straight down with no horizontal component: no azimuth to
        # preserve, so flip to straight up.
        return np.array([0.0, 0.0, T_mag])

    azimuth = u[:2] / T_h
    u_new = np.zeros(3)
    u_new[:2] = azimuth * (T_mag * np.sin(theta_max))
    u_new[2]  = T_mag * np.cos(theta_max)
    return u_new


def tilt_deg(u):
    """Angle of a thrust vector from vertical, in degrees."""
    mag = np.linalg.norm(u)
    if mag < 1e-12:
        return 0.0
    return float(np.degrees(np.arctan2(np.linalg.norm(u[:2]), u[2])))


# =====================================================================
# 8. CLOSED-LOOP ROLLOUT
# =====================================================================

def _state_8(x7, tf, k, use_ttg):
    """Build the 8-dim state: physics(7) + time channel."""
    s = np.zeros(8, dtype=np.float32)
    s[:7] = x7
    s[7]  = tf * (N - k) / N if use_ttg else tf
    return s


def rollout_closed_loop(
    model: nn.Module,
    norms: NormStats,
    ic_7: np.ndarray,
    tf: float,
    thres: LandingThresholds,
    is_mlp: bool = False,
    use_ttg: bool = False,
    clip_tilt: bool = True,
    device: torch.device = torch.device("cpu"),
):
    """Autoregressive rollout with constraint enforcement.

    Returns
    -------
    result : dict, including
        reached_ground : bool (physical; use this, not category strings)
        tilt_raw_deg   : (K,) tilt of the RAW network output
        n_clip_mag     : steps where the magnitude bound was active
        n_clip_tilt    : steps where the tilt cone was active
    """
    dt = tf / N

    x_traj       = np.zeros((N + 1, nx))
    u_traj       = np.zeros((N, nu))     # enforced, applied to dynamics
    u_traj_raw   = np.zeros((N, nu))     # raw network output
    T_mag_kN     = np.zeros(N)
    T_mag_raw_kN = np.zeros(N)
    tilt_arr     = np.zeros(N)
    tilt_raw_arr = np.zeros(N)

    x_traj[0] = ic_7.copy()

    if not is_mlp:
        state_buf = np.zeros((1, N, 8), dtype=np.float32)

    completed      = True
    reached_ground = False
    term_reason    = ""
    steps_done     = 0
    n_clip_mag     = 0
    n_clip_tilt    = 0

    model.eval()
    with torch.no_grad():
        for k in range(N):
            # --- 1. Build input ---
            s8 = _state_8(x_traj[k], tf, k, use_ttg)

            if is_mlp:
                s8_norm = norms.norm_state(s8.reshape(1, -1))
                inp = torch.from_numpy(s8_norm).float().to(device)
                a_norm = model(inp)[0].cpu().numpy()
            else:
                state_buf[0, k, :] = s8
                buf_norm = norms.norm_state(
                    state_buf.reshape(-1, 8)).reshape(1, N, 8)
                inp = torch.from_numpy(buf_norm).float().to(device)
                a_norm = model(inp)[0, k, :].cpu().numpy()

            # --- 2. Un-normalise -> physical thrust [MN] ---
            u_raw = norms.unnorm_action(a_norm.reshape(1, -1)).flatten()
            u_raw = u_raw.astype(np.float64)
            u_traj_raw[k] = u_raw

            # --- 3. Enforce the admissible set ---
            u_k = enforce_thrust(u_raw, clip_tilt=clip_tilt)
            u_traj[k] = u_k

            # --- 4. Diagnostics: raw vs enforced ---
            mag_raw = np.linalg.norm(u_raw)
            mag_enf = np.linalg.norm(u_k)
            T_mag_raw_kN[k] = mag_raw * 1e3
            T_mag_kN[k]     = mag_enf * 1e3
            tilt_raw_arr[k] = tilt_deg(u_raw)
            tilt_arr[k]     = tilt_deg(u_k)

            if abs(mag_raw - mag_enf) > 1e-10:
                n_clip_mag += 1
            if clip_tilt and tilt_raw_arr[k] > np.degrees(theta_max) + 1e-9:
                n_clip_tilt += 1

            # --- 5. Propagate ---
            x_next = np.array(_F_rk4(x_traj[k], u_k, dt)).flatten()

            # --- 6. Termination checks ---
            if np.any(np.isnan(x_next)):
                x_traj[k + 1] = x_next
                completed, term_reason = False, f"NaN at step {k+1}"
                steps_done = k + 1
                break

            # GROUND CONTACT — normal physical end of trajectory.
            if x_next[2] < 0.0:
                x_traj[k + 1] = x_next
                reached_ground = True
                term_reason = f"ground contact at step {k+1}"
                steps_done = k + 1
                break

            if np.linalg.norm(x_next[:3]) > thres.max_pos_m:
                x_traj[k + 1] = x_next
                completed, term_reason = (
                    False,
                    f"diverged ||r||={np.linalg.norm(x_next[:3]):.0f} m")
                steps_done = k + 1
                break

            # Mass floor: m_dry
            if x_next[6] < m_dry:
                x_traj[k + 1] = x_next
                completed, term_reason = (
                    False, f"mass={x_next[6]:.0f} kg < m_dry at step {k+1}")
                steps_done = k + 1
                break

            x_traj[k + 1] = x_next
            steps_done = k + 1

    # Trim to actual length
    K = steps_done
    x_traj       = x_traj[:K + 1]
    u_traj       = u_traj[:K]
    u_traj_raw   = u_traj_raw[:K]
    T_mag_kN     = T_mag_kN[:K]
    T_mag_raw_kN = T_mag_raw_kN[:K]
    tilt_arr     = tilt_arr[:K]
    tilt_raw_arr = tilt_raw_arr[:K]

    # --- Touchdown detection and classification ---
    td_state, td_time, td_type = find_touchdown(x_traj, u_traj, tf, thres)

    if td_state is not None:
        reached_ground = True
        category, details = classify_landing(td_state, thres)
    elif "mass" in term_reason:
        category, details = "fuel_exhausted", {}
    elif not completed:
        category, details = "diverged", {}
    else:
        category, details = "airborne", {}

    return dict(
        x_traj=x_traj, u_traj=u_traj, u_traj_raw=u_traj_raw,
        T_mag_kN=T_mag_kN, T_mag_raw_kN=T_mag_raw_kN,
        tilt_deg=tilt_arr, tilt_raw_deg=tilt_raw_arr,
        completed=completed, reached_ground=reached_ground,
        term_reason=term_reason, steps_done=K,
        n_clip_mag=n_clip_mag, n_clip_tilt=n_clip_tilt,
        td_state=td_state, td_time=td_time, td_type=td_type,
        category=category, details=details, tf=tf,
    )


# =====================================================================
# 9. OPEN-LOOP (TEACHER-FORCED) EVALUATION
# =====================================================================

def eval_open_loop(model, norms, expert, use_ttg=False,
                   device=torch.device("cpu")):
    """Teacher-forced: feed expert states in and compare predicted actions to
    expert actions.
    """
    x_states = expert["x_traj"][:-1].copy()      # (N, 8)
    u_expert = expert["u_traj"].copy()           # (N, 3)
    tf = expert["t_f"]

    if use_ttg:
        for k in range(x_states.shape[0]):
            x_states[k, 7] = tf * (N - k) / N

    x_norm = norms.norm_state(x_states)

    model.eval()
    with torch.no_grad():
        inp = torch.from_numpy(x_norm[np.newaxis]).float().to(device)
        pred_norm = model(inp)[0].cpu().numpy()

    u_pred = norms.unnorm_action(pred_norm)      # (N, 3) MN — raw

    diff = u_pred - u_expert
    mse  = float(np.mean(diff ** 2))
    per_step_err = np.sqrt(np.mean(diff ** 2, axis=1))

    return dict(u_pred=u_pred, u_expert=u_expert, mse=mse,
                per_step_err=per_step_err, tf=tf,
                x_expert=expert["x_traj"])


# =====================================================================
# 9b. CONTROL DEVIATION vs EXPERT (index-matched, per category)
# =====================================================================

def summarise_control_deviation(results, experts):
    """Mean / min / max thrust and tilt deviation, overall and by category."""

    def dev(r, e):
        K = r["u_traj"].shape[0]
        if K == 0:
            return None
        u_pol = np.asarray(r["u_traj"], dtype=float)          # applied [MN]
        u_exp = np.asarray(e["u_traj"][:K], dtype=float)      # expert  [MN]
        dT = (np.linalg.norm(u_pol, axis=1)
              - np.linalg.norm(u_exp, axis=1)) * 1e3          # kN
        tilt = lambda u: np.degrees(
            np.arctan2(np.linalg.norm(u[:, :2], axis=1), u[:, 2]))
        return dT, tilt(u_pol) - tilt(u_exp)

    pairs = [(r["category"], d) for r, e in zip(results, experts)
             if (d := dev(r, e)) is not None]

    def block(items):
            if not items:
                return None
            dT   = np.concatenate([d[0] for _, d in items])
            dtil = np.concatenate([d[1] for _, d in items])
            steps = np.array([len(d[0]) for _, d in items], dtype=float)

            def stats(a):
                return dict(mean=float(a.mean()),           # signed: bias
                            rmse=float(np.sqrt((a ** 2).mean())),
                            mean_abs=float(np.abs(a).mean()),
                            min=float(a.min()), max=float(a.max()))

            return dict(
                n_traj=len(items), n_steps=int(dT.size),
                steps_per_traj=float(steps.mean()),
                dT_kN=stats(dT), dtilt_deg=stats(dtil),
            )

    out = {"all": block(pairs)}
    for c in CATEGORIES:
        out[c] = block([it for it in pairs if it[0] == c])
    return dict(
        definition="policy applied control minus expert control at the "
                   "same step index (same time, different state)",
        sign_convention="positive = policy above expert",
        by_category=out,
    )


# =====================================================================
# 10. FIGURE OUTPUT — helpers and single-IC comparison
# =====================================================================

FIG_EXT      = "svg"     # set from --fig_format
FIG_DPI      = 300       # only affects rasterised artists
RASTER_DENSE = False     # set from --raster_dense


def _save(fig, out_dir, name):
    """Write one figure to out_dir/<name>.<FIG_EXT> and close it."""
    if out_dir is None:
        plt.show()
        plt.close(fig)
        return
    path = Path(out_dir) / f"{name}.{FIG_EXT}"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"  Saved: {path.name}")
    plt.close(fig)


def _cat_legend(ax, counts, fontsize=8, loc="best"):
    """Landing-category legend built from a {category: n} mapping."""
    cats = [c for c in CATEGORIES if counts.get(c, 0) > 0]
    if not cats:
        return
    handles = [plt.Rectangle((0, 0), 1, 1, color=CAT_COLORS[c])
               for c in cats]
    labels = [f"{CAT_LABELS[c]} (n={counts[c]})" for c in cats]
    ax.legend(handles, labels, fontsize=fontsize, loc=loc, frameon=False)


def plot_single_comparison(expert, cl_result, ol_result=None,
                           out_dir=None, tag=""):
    """Expert-vs-policy comparison for a single IC, written as six independent
    figures:
    """
    tf = expert["t_f"]
    sfx = f"_{tag}" if tag else ""
    t_s_exp = np.linspace(0, tf, expert["x_traj"].shape[0])
    t_c_exp = np.linspace(0, tf * (N - 1) / N, N)

    x_exp = expert["x_traj"][:, :7]
    u_exp = expert["u_traj"]
    T_exp_kN = np.linalg.norm(u_exp, axis=1) * 1e3

    style_exp = dict(color="tab:blue", lw=2, label="Expert")
    style_pol = dict(color="tab:red",  lw=2, ls="--", label="Policy (CL)")
    style_ol  = dict(color="tab:green", lw=1.5, ls=":",
                     label="Policy (OL pred)")

    t_s_cl = (np.linspace(0, tf, cl_result["x_traj"].shape[0])
              if cl_result else None)

    if cl_result:
        cat = cl_result["category"]
        print(f"  [figure set{sfx}] classification: "
              f"{CAT_LABELS.get(cat, cat).upper()}, t_f = {tf:.1f} s")

    # --- Altitude ---
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(t_s_exp, x_exp[:, 2], **style_exp)
    if cl_result:
        ax.plot(t_s_cl, cl_result["x_traj"][:, 2], **style_pol)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Altitude $r_z$ [m]")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    _save(fig, out_dir, f"altitude{sfx}")

    # --- Vertical velocity ---
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(t_s_exp, x_exp[:, 5], **style_exp)
    if cl_result:
        ax.plot(t_s_cl, cl_result["x_traj"][:, 5], **style_pol)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("$v_z$ [m/s]")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    _save(fig, out_dir, f"vz{sfx}")

    # --- Horizontal position norm ---
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(t_s_exp, np.linalg.norm(x_exp[:, :2], axis=1), **style_exp)
    if cl_result:
        ax.plot(t_s_cl,
                np.linalg.norm(cl_result["x_traj"][:, :2], axis=1),
                **style_pol)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\|r_h\|$ [m]")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    _save(fig, out_dir, f"rh{sfx}")

    # --- Thrust magnitude ---
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.step(t_c_exp, T_exp_kN, where="post", **style_exp)
    if cl_result:
        K_cl = cl_result["u_traj"].shape[0]
        if K_cl > 0:
            t_c_cl = np.linspace(0, tf * (K_cl - 1) / N, K_cl)
            ax.step(t_c_cl, cl_result["T_mag_kN"], where="post", **style_pol)
            ax.step(t_c_cl, cl_result["T_mag_raw_kN"], where="post",
                    color="tab:red", alpha=0.35, lw=1,
                    label="Policy raw (pre-clip)")
    if ol_result:
        ax.step(t_c_exp, np.linalg.norm(ol_result["u_pred"], axis=1) * 1e3,
                where="post", **style_ol)
    ax.axhline(T_min_MN * 1e3, color="gray", ls="--", lw=1,
               label=f"$T_{{min}}$={T_min_MN*1e3:.0f} kN")
    ax.axhline(T_max_MN * 1e3, color="gray", ls="-.", lw=1,
               label=f"$T_{{max}}$={T_max_MN*1e3:.0f} kN")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Thrust [kN]")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.grid(alpha=0.3)
    _save(fig, out_dir, f"thrust{sfx}")

    # --- Tilt angle: enforced AND raw ---
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    tilt_exp = np.array([tilt_deg(u_exp[k]) for k in range(N)])
    ax.step(t_c_exp, tilt_exp, where="post", **style_exp)
    if cl_result and cl_result["tilt_deg"].shape[0] > 0:
        K_cl = cl_result["tilt_deg"].shape[0]
        t_c_cl = np.linspace(0, tf * (K_cl - 1) / N, K_cl)
        ax.step(t_c_cl, cl_result["tilt_deg"], where="post", **style_pol)
        ax.step(t_c_cl, cl_result["tilt_raw_deg"], where="post",
                color="tab:red", alpha=0.35, lw=1,
                label="Policy raw (pre-clip)")
    ax.axhline(np.degrees(theta_max), color="red", ls="--", lw=1,
               label=rf"$\theta_{{max}}$={np.degrees(theta_max):.0f}$\degree$")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Tilt [deg]")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.3)
    _save(fig, out_dir, f"tilt{sfx}")

    # --- 2-D trajectory, East-Up ---
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.plot(x_exp[:, 0], x_exp[:, 2], **style_exp)
    if cl_result:
        ax.plot(cl_result["x_traj"][:, 0], cl_result["x_traj"][:, 2],
                **style_pol)
    ax.plot(x_exp[0, 0], x_exp[0, 2], "go", ms=10, label="Start")
    ax.plot(0, 0, "k*", ms=15, label="Target")
    if cl_result:
        xf = cl_result["x_traj"][-1]
        ax.plot(xf[0], xf[2], "rx", ms=12, mew=2,
                label=f"Policy end ({np.linalg.norm(xf[:3]):.0f} m)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("East $r_x$ [m]")
    ax.set_ylabel("Up $r_z$ [m]")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    _save(fig, out_dir, f"traj_eastup{sfx}")


# =====================================================================
# 11. FIGURE OUTPUT — terminal-condition histograms
# =====================================================================

def _terminal_state(r):
    """State to report for an IC: the bisected touchdown state if the vehicle
    landed, otherwise the last integrated state.
    """
    if r["td_state"] is not None:
        return r["td_state"]
    return r["x_traj"][-1]


def _stacked_hist(ax, values_by_cat, bins=40, clip_pct=99.0,
                  clip_lo_pct=None, xlabel="", vlines=None):
    """Stacked histogram over landing categories with percentile clipping.

    Returns
    -------
    counts : {category: n} for the categories actually drawn, for use in
             a per-figure legend. Empty dict if there was no data.
    """
    nonempty = [v for v in values_by_cat.values() if len(v) > 0]
    all_vals = np.concatenate(nonempty) if nonempty else np.array([])
    all_vals = all_vals[np.isfinite(all_vals)]

    if all_vals.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        return {}

    hi = float(np.percentile(all_vals, clip_pct))
    lo = (float(np.percentile(all_vals, clip_lo_pct))
          if clip_lo_pct is not None else float(all_vals.min()))
    if hi <= lo:
        hi = lo + 1e-6
    n_over = int((all_vals > hi).sum()) + int((all_vals < lo).sum())
    edges = np.linspace(lo, hi, bins + 1)

    data, colors, counts = [], [], {}
    for cat in CATEGORIES:
        v = values_by_cat.get(cat, np.array([]))
        v = v[np.isfinite(v)] if len(v) else v
        if len(v) == 0:
            continue
        data.append(np.clip(v, lo, hi))
        colors.append(CAT_COLORS[cat])
        counts[cat] = int(len(v))

    ax.hist(data, bins=edges, stacked=True, color=colors,
            edgecolor="none")

    if vlines:
        x0, x1 = ax.get_xlim()
        for xv, style, lab in vlines:
            if not (x0 <= xv <= x1):
                continue          # outside the clipped range; skip
            ax.axvline(xv, color="k", ls=style, lw=1.0)
            ax.text(xv, ax.get_ylim()[1] * 0.97, f" {lab}", rotation=90,
                    va="top", ha="left", fontsize=6.5, color="k")

    if n_over > 0:
        rng = (f"p{clip_lo_pct:g}-p{clip_pct:g}" if clip_lo_pct is not None
               else f"p{clip_pct:g}")
        ax.text(0.98, 0.97,
                f"clipped at {rng}\n({n_over} in edge bins)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=6.5, color="dimgray")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)
    return counts


def plot_batch_summary(results, out_dir, thres: LandingThresholds):
    """Terminal-condition histograms, stacked by landing category, written as
    six independent figures:
    """
    out_dir = Path(out_dir)
    n = len(results)

    cats = [r["category"] for r in results]
    counts = {c: cats.count(c) for c in CATEGORIES}

    # --- Console summary ---
    print(f"\n{'='*64}")
    print("BATCH SUMMARY")
    print(f"{'='*64}")
    for c in CATEGORIES:
        pct = 100 * counts[c] / n if n else 0.0
        print(f"  {CAT_LABELS[c]:>20s}: {counts[c]:5d} / {n}  ({pct:5.1f}%)")
    landed = sum(1 for r in results if r["reached_ground"])
    print(f"  {'-- reached ground':>20s}: {landed:5d} / {n}  "
          f"({100*landed/n if n else 0:5.1f}%)")
    print(f"{'='*64}")

    # --- Enforcement statistics ---
    tot_steps = sum(r["steps_done"] for r in results)
    tot_mag   = sum(r["n_clip_mag"] for r in results)
    tot_tilt  = sum(r["n_clip_tilt"] for r in results)
    if tot_steps:
        print(f"  Enforcement: magnitude bound {tot_mag}/{tot_steps} steps "
              f"({100*tot_mag/tot_steps:.1f}%), "
              f"tilt cone {tot_tilt}/{tot_steps} steps "
              f"({100*tot_tilt/tot_steps:.1f}%)")
        print(f"{'='*64}")

    # --- Gather per-category arrays ---
    def by_cat(fn, subset=None):
        out = {c: [] for c in CATEGORIES}
        for r in results:
            if subset is not None and not subset(r):
                continue
            try:
                out[r["category"]].append(fn(r))
            except (IndexError, KeyError, ValueError):
                pass
        return {c: np.asarray(v, dtype=float) for c, v in out.items()}

    speed  = by_cat(lambda r: np.linalg.norm(_terminal_state(r)[3:6]))
    vz     = by_cat(lambda r: float(_terminal_state(r)[5]))
    mass_t = by_cat(lambda r: _terminal_state(r)[6] / 1e3)
    rh     = by_cat(lambda r: np.linalg.norm(_terminal_state(r)[:2]))
    tilt_r = by_cat(lambda r: r["tilt_raw_deg"][-1]
                    if len(r["tilt_raw_deg"]) else np.nan)
    alt    = by_cat(lambda r: r["x_traj"][-1, 2],
                    subset=lambda r: not r["reached_ground"])
    n_omitted = landed

    figsize = (6.0, 4.2)

    # --- Terminal speed ---
    fig, ax = plt.subplots(figsize=figsize)
    c = _stacked_hist(ax, speed, xlabel=r"$\|v\|$  [m/s]")
    _cat_legend(ax, c)
    _save(fig, out_dir, "terminal_speed")

    # --- Terminal vertical velocity (signed) ---
    fig, ax = plt.subplots(figsize=figsize)
    c = _stacked_hist(ax, vz, clip_lo_pct=1.0,
                      xlabel=r"$v_z$  [m/s]   (negative = descending)",
                      vlines=[(0.0, "-",  "stationary"),
                              (-thres.soft_vz_ms, "--", "soft limit"),
                              (-thres.hard_vz_ms, ":",  "hard limit")])
    _cat_legend(ax, c, loc="upper left")
    _save(fig, out_dir, "terminal_vz")

    # --- Terminal altitude (non-landed ICs only) ---
    fig, ax = plt.subplots(figsize=figsize)
    c = _stacked_hist(ax, alt, xlabel=r"$r_z$  [m]")
    ax.text(0.98, 0.78,
            f"{n_omitted} landed ICs omitted\n"
            r"($r_z = 0$ by bisection clamp)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, color="dimgray")
    _cat_legend(ax, c)
    _save(fig, out_dir, "terminal_altitude")

    # --- Terminal mass ---
    fig, ax = plt.subplots(figsize=figsize)
    c = _stacked_hist(ax, mass_t, xlabel=r"$m_f$  [t]")
    _cat_legend(ax, c)
    _save(fig, out_dir, "terminal_mass")

    # --- Terminal thrust tilt (raw NN output) ---
    fig, ax = plt.subplots(figsize=figsize)
    c = _stacked_hist(ax, tilt_r,
                      xlabel="tilt from vertical  [deg]",
                      vlines=[(np.degrees(thres.theta_max_rad), "--",
                               "tilt limit")])
    _cat_legend(ax, c)
    _save(fig, out_dir, "terminal_tilt_raw")

    # --- Terminal horizontal position ---
    fig, ax = plt.subplots(figsize=figsize)
    c = _stacked_hist(ax, rh, xlabel=r"$\|r_h\|$  [m]",
                      vlines=[(thres.soft_rh_m, "--", "soft limit")])
    _cat_legend(ax, c)
    _save(fig, out_dir, "terminal_rh")


# =====================================================================
# 12. FIGURE OUTPUT — 3D trajectories
# =====================================================================

def plot_3d_trajectories(results, out_dir, max_plot=0,
                         elev=22, azim=-125):
    """All closed-loop trajectories, coloured by landing category."""
    out_dir = Path(out_dir)
    subset = results if max_plot <= 0 else results[:max_plot]
    n = len(subset)
    if n == 0:
        print("  (no trajectories to plot in 3D)")
        return

    # Draw good outcomes last so they are not buried under failures
    draw_order = ["diverged", "fuel_exhausted", "airborne",
                  "crash", "hard", "soft"]
    alpha = 0.55 if n < 100 else (0.30 if n < 500 else 0.15)
    lw    = 0.9  if n < 100 else (0.60 if n < 500 else 0.40)

    groups = [(cat, [r for r in subset if r["category"] == cat])
              for cat in draw_order]
    groups = [(cat, g) for cat, g in groups if g]
    counts = {cat: len(g) for cat, g in groups}

    # --- Oblique 3D view ---
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    for cat, group in groups:
        for r in group:
            X = r["x_traj"]
            ax.plot(X[:, 0], X[:, 1], X[:, 2], color=CAT_COLORS[cat],
                    alpha=alpha, lw=lw, rasterized=RASTER_DENSE)
    ax.scatter([0], [0], [0], color="k", marker="*", s=140,
               depthshade=False)

    # Ground plane at z = 0
    xl, yl = ax.get_xlim(), ax.get_ylim()
    gx, gy = np.meshgrid(np.linspace(xl[0], xl[1], 2),
                         np.linspace(yl[0], yl[1], 2))
    ax.plot_surface(gx, gy, np.zeros_like(gx), alpha=0.10,
                    color="gray", shade=False)

    ax.set_xlabel("East $r_x$ [m]")
    ax.set_ylabel("North $r_y$ [m]")
    ax.set_zlabel("Up $r_z$ [m]")
    ax.view_init(elev=elev, azim=azim)
    handles = [Line2D([0], [0], color=CAT_COLORS[c], lw=2,
                      label=f"{CAT_LABELS[c]} (n={counts[c]})")
               for c, _ in groups]
    handles.append(Line2D([0], [0], color="k", marker="*", ls="none",
                          ms=12, label="Landing pad"))
    ax.legend(handles=handles, fontsize=8, loc="upper left", frameon=False)
    _save(fig, out_dir, "trajectories_3d")

    # --- Top-down projection, full extent ---
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    for cat, group in groups:
        for r in group:
            X = r["x_traj"]
            ax.plot(X[:, 0], X[:, 1], color=CAT_COLORS[cat],
                    alpha=alpha, lw=lw, rasterized=RASTER_DENSE)
    ax.plot(0, 0, "k*", ms=16, zorder=10)
    ax.add_patch(plt.Circle((0, 0), 10.0, fill=False, color="k",
                            ls="--", lw=1.2, zorder=9))
    ax.set_xlabel("East $r_x$ [m]")
    ax.set_ylabel("North $r_y$ [m]")
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", adjustable="datalim")
    handles = [Line2D([0], [0], color=CAT_COLORS[c], lw=2,
                      label=f"{CAT_LABELS[c]} (n={counts[c]})")
               for c, _ in groups]
    handles.append(Line2D([0], [0], color="k", marker="*", ls="none",
                          ms=12, label="Landing pad"))
    handles.append(Line2D([0], [0], color="k", ls="--", lw=1.2,
                          label="10 m soft-landing radius"))
    ax.legend(handles=handles, fontsize=8, loc="best", frameon=False)
    _save(fig, out_dir, "trajectories_topdown")

    # --- Zoomed top-down dispersion ---
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for cat, group in groups:
        for r in group:
            X = r["x_traj"]
            ax.plot(X[:, 0], X[:, 1], color=CAT_COLORS[cat],
                    alpha=alpha, lw=lw, rasterized=RASTER_DENSE)
        ends = np.array([_terminal_state(r)[:2] for r in group])
        ax.scatter(ends[:, 0], ends[:, 1], color=CAT_COLORS[cat],
                   s=14, edgecolors="none", alpha=0.85,
                   rasterized=RASTER_DENSE)

    ax.add_patch(plt.Circle((0, 0), 10.0, fill=False, color="k",
                            ls="--", lw=1.5, zorder=9))
    ax.plot(0, 0, "k*", ms=18, zorder=10)
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    ax.set_xlabel("East $r_x$ [m]")
    ax.set_ylabel("North $r_y$ [m]")
    ax.grid(alpha=0.25)
    ax.set_aspect("equal")
    handles = [Line2D([0], [0], color=CAT_COLORS[c], marker="o", ls="none",
                      ms=5, label=f"{CAT_LABELS[c]} (n={counts[c]})")
               for c, _ in groups]
    handles.append(Line2D([0], [0], color="k", ls="--", lw=1.5,
                          label="10 m soft limit"))
    ax.legend(handles=handles, fontsize=8, loc="upper right", frameon=False)
    _save(fig, out_dir, "terminal_dispersion")


# =====================================================================
# 13. FIGURE OUTPUT — trajectory bundles (optional)
# =====================================================================

def plot_batch_trajectories(results, experts, out_dir, max_plot=50):
    """Overlay bundles, one figure each: altitude, v_z, thrust."""
    out_dir = Path(out_dir)
    n_plot = min(len(results), max_plot)

    fig_alt, ax_alt = plt.subplots(figsize=(6.5, 4.4))
    fig_vz,  ax_vz  = plt.subplots(figsize=(6.5, 4.4))
    fig_T,   ax_T   = plt.subplots(figsize=(6.5, 4.4))

    for i in range(n_plot):
        r, exp = results[i], experts[i]
        tf  = r["tf"]
        col = CAT_COLORS.get(r["category"], "tab:gray")

        t_s_cl = np.linspace(0, tf, r["x_traj"].shape[0])
        t_s_ex = np.linspace(0, tf, exp["x_traj"].shape[0])
        K_cl = r["u_traj"].shape[0]

        ax_alt.plot(t_s_cl, r["x_traj"][:, 2], color=col, alpha=0.25,
                    lw=0.7, rasterized=RASTER_DENSE)
        ax_alt.plot(t_s_ex, exp["x_traj"][:, 2], color="tab:blue",
                    alpha=0.08, lw=0.5, rasterized=RASTER_DENSE)
        ax_vz.plot(t_s_cl, r["x_traj"][:, 5], color=col, alpha=0.25,
                   lw=0.7, rasterized=RASTER_DENSE)
        ax_vz.plot(t_s_ex, exp["x_traj"][:, 5], color="tab:blue",
                   alpha=0.08, lw=0.5, rasterized=RASTER_DENSE)
        if K_cl > 0:
            t_c_cl = np.linspace(0, tf * (K_cl - 1) / N, K_cl)
            ax_T.plot(t_c_cl, r["T_mag_kN"], color=col, alpha=0.25,
                      lw=0.7, rasterized=RASTER_DENSE)
        t_c_ex = np.linspace(0, tf * (N - 1) / N, N)
        ax_T.plot(t_c_ex, np.linalg.norm(exp["u_traj"], axis=1) * 1e3,
                  color="tab:blue", alpha=0.08, lw=0.5,
                  rasterized=RASTER_DENSE)

    proxies = [Line2D([0], [0], color="tab:blue", lw=2, label="Expert")]
    proxies += [Line2D([0], [0], color=CAT_COLORS[c], lw=2,
                       label=CAT_LABELS[c]) for c in CATEGORIES]

    ax_alt.axhline(0, color="k", lw=0.5)
    ax_alt.set_xlabel("Time [s]")
    ax_alt.set_ylabel("Altitude $r_z$ [m]")
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


def write_ic_csv(results, experts, out_dir):
    """One row per IC: initial condition -> landing category."""
    import csv
    with open(Path(out_dir) / "ic_outcomes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["r_x0", "r_y0", "r_z0", "v_x0", "v_y0", "v_z0",
                    "category"])
        for r, e in zip(results, experts):
            w.writerow([*e["x_traj"][0, :6], r["category"]])
    print(f"  Saved: ic_outcomes.csv  ({len(results)} rows)")

# =====================================================================
# 14. MODEL LOADING
# =====================================================================

def load_model_from_dir(run_dir, device, model_type="auto",
                        checkpoint_file=None, norm_file=None):
    """Load model + norms from a training run directory.

    Returns
    -------
    model, norms, is_mlp
    """
    run_dir = Path(run_dir)

    # --- Discover checkpoint ---
    if checkpoint_file:
        ckpt_path = run_dir / checkpoint_file
    else:
        for name in ("best_model.pt", "model_ttg.pt", "model_const.pt"):
            if (run_dir / name).exists():
                ckpt_path = run_dir / name
                break
        else:
            pts = sorted(run_dir.glob("*.pt"))
            if not pts:
                raise FileNotFoundError(f"No .pt checkpoint in {run_dir}")
            ckpt_path = pts[0]

    # --- Discover norms ---
    if norm_file:
        ns_path = run_dir / norm_file
    else:
        for name in ("norm_stats.npz", "norm_ttg.npz", "norm_const.npz"):
            if (run_dir / name).exists():
                ns_path = run_dir / name
                break
        else:
            npzs = sorted(run_dir.glob("norm*.npz"))
            if not npzs:
                raise FileNotFoundError(f"No norm*.npz in {run_dir}")
            ns_path = npzs[0]

    # --- Config ---
    cfg_path = run_dir / "model_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            raw_cfg = json.load(f)
    else:
        raw_cfg = {}
        print("  (no model_config.json — using default architecture)")

    if model_type == "auto":
        model_type = "mlp" if (raw_cfg.get("model_type") == "mlp"
                               or "hidden_dims" in raw_cfg) else "transformer"
    is_mlp = (model_type == "mlp")

    if is_mlp and "num_hidden_layers" in raw_cfg:
        keys = {"state_dim", "action_dim", "hidden_dim",
                "num_hidden_layers", "activation", "dropout"}
        model = PDGMLPLegacy(**{k: v for k, v in raw_cfg.items()
                                if k in keys}).to(device)
        arch_name = "MLP (legacy, handoff 13)"
    elif is_mlp:
        keys = {"state_dim", "action_dim", "hidden_dims",
                "activation", "dropout"}
        cfg = MLPConfig(**{k: v for k, v in raw_cfg.items() if k in keys})
        model = PDGMLP(cfg).to(device)
        arch_name = "MLP"
    else:
        keys = {"state_dim", "action_dim", "d_model", "nhead",
                "num_layers", "d_ff", "dropout", "max_seq_len"}
        cfg = TConfig(**{k: v for k, v in raw_cfg.items() if k in keys})
        model = PDGTransformer(cfg).to(device)
        arch_name = "Transformer"

    raw_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt:
        model.load_state_dict(raw_ckpt["model_state_dict"])
        epoch = raw_ckpt.get("epoch", "?")
        bvl   = raw_ckpt.get("best_val_loss", float("nan"))
    else:
        model.load_state_dict(raw_ckpt)
        epoch, bvl = "?", float("nan")

    model.eval()
    norms = NormStats.load(ns_path)

    print(f"Loaded {arch_name} from {run_dir}")
    print(f"  Checkpoint: {ckpt_path.name}")
    print(f"  Norms:      {ns_path.name}")
    print(f"  Parameters: {model.count_parameters():,}")
    if epoch != "?":
        print(f"  Epoch: {epoch}, best_val_loss: {bvl:.6f}")
    print(norms.summary())

    return model, norms, is_mlp


def _print_config_banner(arch_tag, args):
    """Echo the evaluation configuration so logs are self-describing."""
    print(f"\nArchitecture: {arch_tag}   TTG: {args.ttg}   "
          f"tilt enforcement: {not args.no_clip_tilt}")
    print(f"Magnitude bounds: [{T_min_MN*1e3:.0f}, {T_max_MN*1e3:.0f}] kN"
          f"   tilt cone: {np.degrees(theta_max):.0f} deg")
    if args.no_clip_tilt:
        print("  NOTE: tilt cone DISABLED — the policy may command thrust "
              "directions\n        the vehicle cannot physically produce. "
              "Use for comparison only.")


# =====================================================================
# 15. RUNNERS
# =====================================================================

def run_single(args):
    """Evaluate on a single trajectory file."""
    device = torch.device("cpu")
    thres = LandingThresholds()
    clip_tilt = not args.no_clip_tilt

    model, norms, is_mlp = load_model_from_dir(
        args.run_dir, device, model_type=args.model_type,
        checkpoint_file=args.checkpoint, norm_file=args.norms)
    arch_tag = "MLP" if is_mlp else "Transformer"

    expert = load_traj(args.traj_file)
    if expert["status"] != 0:
        print(f"WARNING: trajectory status = {expert['status']} "
              f"(expert did not converge)")

    ic_7 = expert["ic"][:7]
    tf   = expert["t_f"]

    print(f"\nIC: r=[{ic_7[0]:.1f}, {ic_7[1]:.1f}, {ic_7[2]:.1f}] m  "
          f"v=[{ic_7[3]:.1f}, {ic_7[4]:.1f}, {ic_7[5]:.1f}] m/s  "
          f"m={ic_7[6]:.0f} kg  t_f={tf:.2f} s")
    _print_config_banner(arch_tag, args)

    cl_result = ol_result = None

    if args.mode in ("closed_loop", "both"):
        print("\n--- Closed-loop rollout ---")
        t0 = time.time()
        cl_result = rollout_closed_loop(
            model, norms, ic_7, tf, thres, is_mlp=is_mlp,
            use_ttg=args.ttg, clip_tilt=clip_tilt, device=device)
        dt_ms = (time.time() - t0) * 1e3
        print(f"  Time: {dt_ms:.1f} ms  ({dt_ms/N:.2f} ms/step)")
        print(f"  Steps completed: {cl_result['steps_done']}/{N}")
        print(f"  Reached ground: {cl_result['reached_ground']}")
        print(f"  Category: {CAT_LABELS[cl_result['category']].upper()}")

        if cl_result["td_state"] is not None:
            d = cl_result["details"]
        
            print(f"  Touchdown ({cl_result['td_type']}) at "
                  f"t={cl_result['td_time']:.3f} s:")
            print(f"    ||r_h|| = {d['r_h']:.2f} m   r_z = {d['r_z']:.3f} m")
            print(f"    |v_z|   = {d['vz_signed']:.2f} m/s")
            print(f"    |v_x|   = {d['vx']:.2f} m/s   |v_y| = {d['vy']:.2f} m/s")
            print(f"    m_f     = {d['m_f']/1e3:.3f} t")
        else:
            xf = cl_result["x_traj"][-1]
            print(f"  Final state: r_z={xf[2]:.1f} m  v_z={xf[5]:.2f} m/s  "
                  f"||r||={np.linalg.norm(xf[:3]):.1f} m")
        if cl_result["term_reason"]:
            print(f"  Termination: {cl_result['term_reason']}")

        K = max(cl_result["steps_done"], 1)
        print(f"  Enforcement: magnitude {cl_result['n_clip_mag']}/{K} steps, "
              f"tilt {cl_result['n_clip_tilt']}/{K} steps")
        if len(cl_result["tilt_raw_deg"]):
            print(f"  Raw tilt: max {cl_result['tilt_raw_deg'].max():.1f} deg, "
                  f"terminal {cl_result['tilt_raw_deg'][-1]:.1f} deg")

    if args.mode in ("open_loop", "both"):
        print("\n--- Open-loop (teacher-forced, no enforcement) ---")
        ol_result = eval_open_loop(model, norms, expert,
                                   use_ttg=args.ttg, device=device)
        print(f"  MSE (physical MN^2): {ol_result['mse']:.2e}")
        rms_kN = np.sqrt(ol_result["mse"]) * 1e3
        print(f"  RMS thrust error: {rms_kN:.2f} kN "
              f"({rms_kN / (T_max_MN*1e3) * 100:.2f}% of T_max)")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_single_comparison(expert, cl_result, ol_result, out_dir=out,
                           tag=Path(args.traj_file).stem)


def run_batch(args):
    """Evaluate on the test split of a dataset directory."""
    device = torch.device("cpu")
    thres = LandingThresholds()
    clip_tilt = not args.no_clip_tilt

    model, norms, is_mlp = load_model_from_dir(
        args.run_dir, device, model_type=args.model_type,
        checkpoint_file=args.checkpoint, norm_file=args.norms)
    arch_tag = "MLP" if is_mlp else "Transformer"

    data_dir = Path(args.data_dir)
    all_files = sorted(data_dir.glob("traj_*.npz"))
    print(f"\nFound {len(all_files)} trajectory files in {data_dir}")
    _, _, test_files = split_by_ic(all_files, seed=args.seed)
    print(f"Test split: {len(test_files)} files")
    _print_config_banner(arch_tag, args)

    if args.max_trajs > 0:
        test_files = test_files[:args.max_trajs]
        print(f"  (limited to {len(test_files)} trajectories)")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results, experts = [], []
    n_skip = 0

    if not args.quiet:
        print(f"\n{'#':>5s}  {'Cat':>15s}  {'rz_f':>8s}  {'|vz|':>8s}  "
              f"{'||rh||':>8s}  {'steps':>5s}  Notes")
        print("-" * 78)

    for i, fpath in enumerate(test_files):
        exp = load_traj(fpath)
        if exp["status"] != 0:
            n_skip += 1
            continue

        ic_7 = exp["ic"][:7]
        tf   = exp["t_f"]

        if args.mode in ("closed_loop", "both"):
            r = rollout_closed_loop(
                model, norms, ic_7, tf, thres, is_mlp=is_mlp,
                use_ttg=args.ttg, clip_tilt=clip_tilt, device=device)
        else:
            r = dict(category="airborne", td_state=None,
                     x_traj=exp["x_traj"][:, :7], u_traj=exp["u_traj"],
                     u_traj_raw=exp["u_traj"], T_mag_kN=np.zeros(0),
                     T_mag_raw_kN=np.zeros(0), tilt_deg=np.zeros(0),
                     tilt_raw_deg=np.zeros(0), steps_done=N, tf=tf,
                     completed=True, reached_ground=False, term_reason="",
                     td_time=None, td_type="none", details={},
                     n_clip_mag=0, n_clip_tilt=0)

        results.append(r)
        experts.append(exp)

        if not args.quiet:
            xt = _terminal_state(r)
            note = r["term_reason"][:32] if r["term_reason"] else ""
            print(f"{i:5d}  {r['category']:>15s}  {xt[2]:8.2f}  "
                  f"{abs(xt[5]):8.2f}  {np.linalg.norm(xt[:2]):8.2f}  "
                  f"{r['steps_done']:5d}  {note}")

    if n_skip:
        print(f"\n  Skipped {n_skip} non-converged expert trajectories")

    # --- Figures ---
    if results and args.mode != "open_loop":
        plot_batch_summary(results, out, thres)
        plot_3d_trajectories(results, out, max_plot=args.max_3d)

        if args.plot_bundles:
            plot_batch_trajectories(results, experts, out,
                                    max_plot=min(len(results), 100))

        # One example figure per category encountered
        seen = set()
        for i, r in enumerate(results):
            if r["category"] not in seen:
                seen.add(r["category"])
                plot_single_comparison(
                    experts[i], r, out_dir=out,
                    tag=f"example_{r['category']}_{i:04d}")

    # --- Open-loop batch statistics ---
    if args.mode in ("open_loop", "both"):
        print("\n--- Open-loop statistics (no enforcement) ---")
        mse_arr = np.array([eval_open_loop(model, norms, e,
                                           use_ttg=args.ttg,
                                           device=device)["mse"]
                            for e in experts])
        rms_kN = np.sqrt(mse_arr) * 1e3
        print(f"  RMS thrust error: {rms_kN.mean():.2f} "
              f"+/- {rms_kN.std():.2f} kN")
        print(f"  As % of T_max:    "
              f"{(rms_kN / (T_max_MN*1e3) * 100).mean():.2f}%")

    # --- JSON summary ---
    tot_steps = sum(r["steps_done"] for r in results)
    dev_summary = (summarise_control_deviation(results, experts)
                   if results and args.mode != "open_loop" else None)
    summary = dict(
        n_total=len(results),
        n_skipped=n_skip,
        architecture=arch_tag,
        use_ttg=args.ttg,
        tilt_enforcement=clip_tilt,
        enforcement="magnitude bounds"
                    + (" + 15 deg tilt cone" if clip_tilt else " only"),
        categories={c: sum(1 for r in results if r["category"] == c)
                    for c in CATEGORIES},
        n_reached_ground=sum(1 for r in results if r["reached_ground"]),
        control_deviation=dev_summary,
        clip_steps=dict(
            total=tot_steps,
            magnitude=sum(r["n_clip_mag"] for r in results),
            tilt=sum(r["n_clip_tilt"] for r in results),
        ),
        thresholds=dict(
            soft_rh_m=thres.soft_rh_m,
            soft_vz_ms=thres.soft_vz_ms,
            soft_vh_axis_ms=thres.soft_vh_axis_ms,
            hard_vz_ms=thres.hard_vz_ms,
            hard_vh_norm_ms=thres.hard_vh_norm_ms,
            grazing_alt_m=thres.grazing_alt_m,
            theta_max_deg=float(np.degrees(thres.theta_max_rad)),
        ),
    )
    with open(out / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n  Saved: eval_summary.json")

    if results:
        write_ic_csv(results, experts, out)


# =====================================================================
# 16. CLI
# =====================================================================

def main():
    p = argparse.ArgumentParser(
        description="RETALT1 PDG — Standalone Evaluation (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--run_dir", required=True,
                   help="Training run directory containing the checkpoint "
                        "and normalisation stats")
    p.add_argument("--mode", choices=["open_loop", "closed_loop", "both"],
                   default="both", help="Evaluation mode (default: both)")
    p.add_argument("--out_dir", default="eval_results",
                   help="Output directory for plots and reports")
    p.add_argument("--model_type", choices=["auto", "transformer", "mlp"],
                   default="auto",
                   help="Architecture. 'auto' inspects model_config.json")
    p.add_argument("--ttg", action="store_true", default=False,
                   help="Use time-to-go: channel 7 = t_f*(N-k)/N. MUST "
                        "match the convention used during training — a "
                        "mismatch silently destroys performance.")
    p.add_argument("--no_clip_tilt", action="store_true", default=False,
                   help="Disable the 15 deg tilt cone (magnitude bounds "
                        "only). Use to quantify how much of the "
                        "performance comes from the enforcement layer "
                        "rather than the policy.")
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint filename inside --run_dir")
    p.add_argument("--norms", default=None,
                   help="Normalisation filename inside --run_dir")

    # Single IC
    p.add_argument("--traj_file", default=None,
                   help="Path to a single trajectory NPZ file")

    # Batch
    p.add_argument("--data_dir", default=None,
                   help="Directory with traj_*.npz files (batch mode)")
    p.add_argument("--max_trajs", type=int, default=0,
                   help="Max test trajectories (0 = all)")
    p.add_argument("--max_3d", type=int, default=0,
                   help="Max trajectories in the 3D figure (0 = all)")
    p.add_argument("--plot_bundles", action="store_true", default=False,
                   help="Also produce the v1 bundle overlay figure")
    p.add_argument("--quiet", action="store_true", default=False,
                   help="Suppress the per-trajectory table")
    p.add_argument("--seed", type=int, default=42,
                   help="Split seed — MUST match training")

    # Figures
    p.add_argument("--fig_format", choices=["svg", "pdf", "png", "eps"],
                   default="svg",
                   help="Figure file format (default: svg). Every figure "
                        "is written as a separate file with no in-figure "
                        "title, for captioning in the report.")
    p.add_argument("--fig_dpi", type=int, default=300,
                   help="DPI. Only affects raster output and rasterised "
                        "artists; vector formats ignore it otherwise.")
    p.add_argument("--raster_dense", action="store_true", default=False,
                   help="Rasterise the trajectory lines in the bundle and "
                        "3D figures while keeping axes, labels and legend "
                        "as vectors. Use if the SVGs from a few-hundred-IC "
                        "sweep get too heavy for the typesetter.")

    args = p.parse_args()

    global FIG_EXT, FIG_DPI, RASTER_DENSE
    FIG_EXT      = args.fig_format
    FIG_DPI      = args.fig_dpi
    RASTER_DENSE = args.raster_dense

    if args.traj_file is None and args.data_dir is None:
        p.error("Provide either --traj_file (single IC) or "
                "--data_dir (batch evaluation)")

    if args.traj_file:
        run_single(args)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
