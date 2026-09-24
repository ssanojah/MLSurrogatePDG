#!/usr/bin/env python3
"""
Timing benchmark for the hybrid guidance policy.

Measures the network guidance call, the forward pass versus sequence
length, the analytic terminal-law call and the total guidance time per
descent. Writes timing.json.

Usage:
    python timing_bench.py --run_dir runs/<gate_run> --ttg \\
        --data_dir data/<gate_dataset> --n_reps 2000 --threads 1
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from standalone_eval_v2 import (
    N, load_model_from_dir, _state_8, enforce_thrust, load_traj, split_by_ic,
)
from gate_patch_eval import terminal_command


def stats_ms(samples_ns):
    """Latency summary. p99/max are the real-time-relevant entries."""
    a = np.asarray(samples_ns, dtype=np.float64) / 1e6
    return dict(
        n=int(a.size),
        mean_ms=float(a.mean()),
        std_ms=float(a.std(ddof=1)) if a.size > 1 else 0.0,
        p50_ms=float(np.percentile(a, 50)),
        p95_ms=float(np.percentile(a, 95)),
        p99_ms=float(np.percentile(a, 99)),
        max_ms=float(a.max()),
        min_ms=float(a.min()),
    )


def _print_block(name, s, budget_ms=None):
    print(f"\n  {name}")
    print(f"    mean {s['mean_ms']:8.3f} ms   p50 {s['p50_ms']:8.3f}   "
          f"p95 {s['p95_ms']:8.3f}   p99 {s['p99_ms']:8.3f}   "
          f"max {s['max_ms']:8.3f}   (n={s['n']})")
    if budget_ms is not None:
        print(f"    budget {budget_ms:.1f} ms  ->  "
              f"{100 * s['p99_ms'] / budget_ms:.2f}% of budget at p99")


# =====================================================================
# (A) network guidance call, as implemented in rollout_hybrid
# =====================================================================

def bench_network_call(model, norms, x7, tf, k, use_ttg, is_mlp, device,
                       n_reps, n_warmup):
    """One guidance call, timed exactly as rollout_hybrid performs it: state ->
    _state_8 -> normalise -> forward -> unnormalise -> enforce_thrust.
    """
    if not is_mlp:
        buf = np.zeros((1, N, 8), dtype=np.float32)
        for j in range(k + 1):
            buf[0, j, :] = _state_8(x7, tf, j, use_ttg)

    def one_call():
        s8 = _state_8(x7, tf, k, use_ttg)
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
            norms.unnorm_action(a_n.reshape(1, -1)).flatten(), dtype=np.float64)
        return enforce_thrust(u_raw, clip_tilt=True)

    with torch.no_grad():
        for _ in range(n_warmup):
            one_call()
        out = np.empty(n_reps, dtype=np.int64)
        for i in range(n_reps):
            t0 = time.perf_counter_ns()
            one_call()
            out[i] = time.perf_counter_ns() - t0
    return out


# =====================================================================
# (B) forward-pass cost vs sequence length (transformer only)
# =====================================================================

def bench_seq_scaling(model, norms, x7, tf, use_ttg, device,
                      lengths, n_reps, n_warmup):
    """Raw forward-pass cost at several sequence lengths, model only (no
    normalisation, no enforcement).
    """
    res = {}
    for L in lengths:
        buf = np.zeros((1, L, 8), dtype=np.float32)
        for j in range(L):
            buf[0, j, :] = _state_8(x7, tf, j, use_ttg)
        inp = torch.from_numpy(
            norms.norm_state(buf.reshape(-1, 8)).reshape(1, L, 8)).float()
        inp = inp.to(device)
        with torch.no_grad():
            for _ in range(n_warmup):
                model(inp)
            s = np.empty(n_reps, dtype=np.int64)
            for i in range(n_reps):
                t0 = time.perf_counter_ns()
                model(inp)
                s[i] = time.perf_counter_ns() - t0
        res[str(L)] = stats_ms(s)
    return res


# =====================================================================
# (C) analytic terminal law
# =====================================================================

def bench_terminal_call(x7, cfg, n_reps, n_warmup):
    for _ in range(n_warmup):
        terminal_command(x7, cfg)
    out = np.empty(n_reps, dtype=np.int64)
    for i in range(n_reps):
        t0 = time.perf_counter_ns()
        terminal_command(x7, cfg)
        out[i] = time.perf_counter_ns() - t0
    return out


# =====================================================================
# main
# =====================================================================

def main():
    p = argparse.ArgumentParser(
        description="Deployment latency of the hybrid guidance policy")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--data_dir", default=None,
                   help="Expert dataset. If given, real ICs and t_f are used "
                        "and the per-descent total is averaged over --n_traj "
                        "of them. Otherwise a single representative state is "
                        "synthesised and the total is arithmetic.")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--seed", type=int, default=42,
                   help="split seed — match training so the states timed come "
                        "from the test split")
    p.add_argument("--ttg", action="store_true", default=False)
    p.add_argument("--model_type", default="auto")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--norms", default=None)

    p.add_argument("--threads", type=int, default=1,
                   help="torch intra-op threads. 1 is the conservative, "
                        "reportable case; the HPC default is 4.")
    p.add_argument("--n_reps", type=int, default=2000)
    p.add_argument("--n_warmup", type=int, default=200,
                   help="Discarded. The first calls include lazy allocation "
                        "and kernel selection and are not representative.")
    p.add_argument("--n_traj", type=int, default=20,
                   help="ICs over which to average the per-descent total")
    p.add_argument("--node", type=int, default=N - 1,
                   help="Node index for the single-call benchmark. Default is "
                        "the last node: the full-buffer cost, i.e. the worst "
                        "case of the as-implemented rollout.")
    p.add_argument("--dt_term", type=float, default=0.05)
    p.add_argument("--n_term_steps", type=int, default=61,
                   help="Terminal-phase calls per descent. 61 = the measured "
                        "median 3.05 s at dt_term = 0.05 s.")
    p.add_argument("--expert_ms", type=float, default=35.0,
                   help="FORCESPRO solve time to compare against. Measure it "
                        "on THIS machine before quoting the ratio.")
    a = p.parse_args()

    torch.set_num_threads(a.threads)
    device = torch.device("cpu")
    out_dir = Path(a.out_dir) if a.out_dir else Path(a.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  DEPLOYMENT LATENCY — hybrid guidance policy")
    print("=" * 72)
    print(f"  platform     : {platform.platform()}")
    print(f"  processor    : {platform.processor() or 'unknown'}")
    print(f"  python       : {platform.python_version()}   "
          f"torch {torch.__version__}")
    print(f"  torch threads: {a.threads}  (interop "
          f"{torch.get_num_interop_threads()})")
    print(f"  reps         : {a.n_reps}  (+{a.n_warmup} warmup, discarded)")

    model, norms, is_mlp = load_model_from_dir(
        a.run_dir, device, model_type=a.model_type,
        checkpoint_file=a.checkpoint, norm_file=a.norms)
    model.eval()
    n_par = sum(q.numel() for q in model.parameters())
    arch = "MLP" if is_mlp else "Transformer"
    print(f"  model        : {arch}, {n_par:,} parameters")

    cfg = dict(dt_term=a.dt_term, h_c=0.0, v_td=0.0, tau_capture=0.3,
               t_term_max=60.0, no_horizontal=False,
               eps_rz=0.01, eps_v=0.01,
               t_go_min=0.2, t_go_max=60.0, a_cap=50.0)

    # ---- states to time on -------------------------------------------
    if a.data_dir:
        files = sorted(Path(a.data_dir).glob("traj_*.npz"))
        if not files:
            raise SystemExit(f"No traj_*.npz in {a.data_dir}")
        _, _, test_files = split_by_ic(files, seed=a.seed)
        picks = []
        for f in test_files:
            e = load_traj(f)
            if e["status"] != 0:
                continue
            picks.append((e["ic"][:7].astype(float), float(e["t_f"]),
                          e["x_traj"][-1, :7].astype(float)))
            if len(picks) >= a.n_traj:
                break
        if not picks:
            raise SystemExit("No converged trajectories in the test split.")
        print(f"  states       : {len(picks)} real test-split ICs "
              f"from {a.data_dir}")
    else:
        # Representative approach state and gate state (medians, 20/-16 run).
        picks = [(np.array([-2300.0, 200.0, 6000.0, 199.0, -18.0, -251.0,
                            70000.0]), 50.0,
                  np.array([0.9, 0.0, 23.8, 0.15, 0.0, -15.69, 61840.0]))]
        print("  states       : synthetic (no --data_dir); "
              "timing is shape-driven so this is representative")

    x_ic, tf, x_gate = picks[0]
    budget_nn_ms = 1000.0 * tf / N
    budget_term_ms = 1000.0 * a.dt_term

    # ---- (A) ----------------------------------------------------------
    print("\n" + "-" * 72)
    print(f"  (A) network guidance call, as implemented "
          f"(node {a.node}, full buffer)")
    print("-" * 72)
    sA = stats_ms(bench_network_call(model, norms, x_ic, tf, a.node,
                                     a.ttg, is_mlp, device,
                                     a.n_reps, a.n_warmup))
    _print_block("network call", sA, budget_nn_ms)

    # ---- (B) ----------------------------------------------------------
    sB = None
    if not is_mlp:
        lengths = [L for L in (1, 15, 30, 45, N) if L <= N]
        print("\n" + "-" * 72)
        print("  (B) forward pass vs sequence length (model only)")
        print("-" * 72)
        sB = bench_seq_scaling(model, norms, x_ic, tf, a.ttg, device,
                               lengths, max(200, a.n_reps // 4), a.n_warmup)
        print(f"    {'seq_len':>8s} {'p50 [ms]':>10s} {'p99 [ms]':>10s}")
        for L in lengths:
            s = sB[str(L)]
            print(f"    {L:>8d} {s['p50_ms']:>10.3f} {s['p99_ms']:>10.3f}")
        r = sB[str(N)]["p50_ms"] / max(sB["1"]["p50_ms"], 1e-12)
        print(f"    full-sequence / single-step = {r:.2f}x  "
              f"-> a KV-cached deployment saves up to {100*(1-1/r):.0f}%")

    # ---- (C) ----------------------------------------------------------
    print("\n" + "-" * 72)
    print("  (C) analytic terminal law call")
    print("-" * 72)
    sC = stats_ms(bench_terminal_call(x_gate, cfg, a.n_reps, a.n_warmup))
    _print_block("terminal_command", sC, budget_term_ms)

    # ---- (D) ----------------------------------------------------------
    nn_total_ms = N * sA["mean_ms"]
    term_total_ms = a.n_term_steps * sC["mean_ms"]
    total_ms = nn_total_ms + term_total_ms
    expert_total_ms = N * a.expert_ms

    print("\n" + "=" * 72)
    print("  (D) PER-DESCENT GUIDANCE COMPUTE  (excludes plant integration)")
    print("=" * 72)
    print(f"    network   {N:3d} calls x {sA['mean_ms']:7.3f} ms = "
          f"{nn_total_ms:9.2f} ms")
    print(f"    terminal  {a.n_term_steps:3d} calls x {sC['mean_ms']:7.3f} ms = "
          f"{term_total_ms:9.2f} ms")
    print(f"    TOTAL                              = {total_ms:9.2f} ms")
    print(f"\n    FORCESPRO onboard, same cadence: {N} x {a.expert_ms:.1f} ms "
          f"= {expert_total_ms:.0f} ms")
    print(f"    speed-up (per call, p99 vs expert mean): "
          f"{a.expert_ms / max(sA['p99_ms'], 1e-9):.1f}x")
    print(f"    speed-up (per descent, means):           "
          f"{expert_total_ms / max(total_ms, 1e-9):.1f}x")
    print("\n    NOTE: --expert_ms is a stored constant unless you measured "
          "it\n    on this machine. A cross-machine ratio is not a result.")

    summary = dict(
        platform=platform.platform(),
        processor=platform.processor(),
        python=platform.python_version(),
        torch=torch.__version__,
        torch_threads=a.threads,
        architecture=arch,
        n_parameters=int(n_par),
        use_ttg=bool(a.ttg),
        n_reps=a.n_reps, n_warmup=a.n_warmup,
        node_timed=a.node,
        network_call=sA,
        network_vs_seq_len=sB,
        terminal_call=sC,
        budget_network_call_ms=budget_nn_ms,
        budget_terminal_call_ms=budget_term_ms,
        per_descent=dict(
            n_network_calls=N, n_terminal_calls=a.n_term_steps,
            network_ms=nn_total_ms, terminal_ms=term_total_ms,
            total_ms=total_ms,
            expert_ms_per_solve=a.expert_ms,
            expert_total_ms=expert_total_ms,
        ),
        caveats=[
            "Network call is timed as implemented: the full 60-step buffer is "
            "re-run every node. Pessimistic vs a KV-cached deployment.",
            "Desktop/HPC CPython + PyTorch, not a flight processor. The "
            "defensible claim is the ratio against FORCESPRO on the same "
            "machine, and that the network cost is data-independent.",
            "p99 and max are the real-time-relevant statistics, not the mean.",
            "--expert_ms is a constant unless re-measured on this machine.",
        ],
    )
    with open(out_dir / "timing.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out_dir / 'timing.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
