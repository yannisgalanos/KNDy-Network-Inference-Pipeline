"""
calibration_check.py
--------------------
Pre-inference calibration: verify that the 8-D summary statistics vary
smoothly and discriminatively across the parameter space.

Includes the calcium forward model so that simulated summary statistic
match the data representation used during SNPE training.

Usage
~~~~~
  python calibration_check.py --n-samples 500 --n-workers 16
  python calibration_check.py --n-samples 100 --n-workers 4 --output cal/
  python calibration_check.py --calcium-params calcium_params.json --n-samples 500

Output
~~~~~~
  <o>/calibration_scatter.png
  <o>/calibration_correlation.png
  <o>/calibration_histograms.png
  <o>/calibration_data.npz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, qmc

from config import SimulationConfig, InferenceConfig
from simulation import simulate_kndy_network, generate_neuron_positions
from statistics import compute_summary_statistics, STAT_NAMES
from calcium_model import calcium_forward_model


# ─────────────────────────────────────────────────────────────────────────────
# Default calcium params (GCaMP6s at 10Hz)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CALCIUM_PARAMS = {
    "dt_imaging": 0.1,
    "amplitude": 0.001,
    "baseline": 0.004,
    "noise_std": 0.002,
    "tau_rise": 0.18,
    "tau_decay": 2.6,
}


# ─────────────────────────────────────────────────────────────────────────────
# Worker function (module-level for multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_worker(args):
    """Run one simulation with calcium forward model, return summary stats."""
    theta, sim_config, seed, calcium_params = args
    sigma, mean_deg, beta_val = theta

    cfg = SimulationConfig(
        n_neurons        = sim_config.n_neurons,
        n_clusters       = sim_config.n_clusters,
        arena_size       = sim_config.arena_size,
        cluster_radius   = sim_config.cluster_radius,
        min_cluster_center_distance = sim_config.min_cluster_center_distance,
        sigma_spatial    = float(np.clip(sigma,       1.0, 1e4)),
        mean_degree      = float(np.clip(mean_deg,    0.5, sim_config.n_neurons - 1)),
        beta_val         = float(np.clip(beta_val,    0.01, 10.0)),
        h0               = sim_config.h0,
        w_NKB            = sim_config.w_NKB,
        w_Dyn            = sim_config.w_Dyn,
        dt               = sim_config.dt,
        T                = sim_config.T,
        burnin           = sim_config.burnin,
        seed             = seed,
    )
    positions = generate_neuron_positions(cfg)
    result    = simulate_kndy_network(cfg, positions=positions)

    calcium = calcium_forward_model(
        result["spikes"], dt_fine=cfg.dt, seed=seed, **calcium_params
    )
    return compute_summary_statistics(calcium, dt=calcium_params["dt_imaging"])


def sample_prior_lhs(n_samples, prior_low, prior_high, seed=42):
    d = len(prior_low)
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    unit_samples = sampler.random(n=n_samples)
    return qmc.scale(unit_samples, prior_low, prior_high)


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

PARAM_NAMES = ["sigma_spatial", "mean_degree", "beta"]
PARAM_UNITS = ["um", "", ""]


def plot_scatter_grid(thetas, stats, out_dir):
    n_params = thetas.shape[1]
    n_stats  = stats.shape[1]
    fig, axes = plt.subplots(n_stats, n_params,
                             figsize=(4 * n_params, 2.8 * n_stats),
                             constrained_layout=True)
    for i in range(n_stats):
        for j in range(n_params):
            ax = axes[i, j]
            col = thetas[:, j]
            if col.size == 0:
                continue
            ax.scatter(col, stats[:, i],
                       s=6, alpha=0.30, color="steelblue", edgecolors="none")
            n_bins = 15
            edges = np.linspace(col.min(), col.max(), n_bins + 1)
            bx, by = [], []
            for b in range(n_bins):
                mask = (col >= edges[b]) & (col < edges[b + 1])
                if mask.sum() > 2:
                    bx.append((edges[b] + edges[b + 1]) / 2)
                    by.append(stats[mask, i].mean())
            if len(bx) > 2:
                ax.plot(bx, by, "r-", lw=2, alpha=0.8)
            rho, pval = spearmanr(col, stats[:, i])
            ax.set_title(f"rho={rho:.2f} (p={pval:.1e})", fontsize=7)
            if i == n_stats - 1:
                label = PARAM_NAMES[j]
                if PARAM_UNITS[j]: label += f" ({PARAM_UNITS[j]})"
                ax.set_xlabel(label, fontsize=8)
            if j == 0:
                ax.set_ylabel(STAT_NAMES[i], fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle("Calibration: Stats vs Parameters", fontsize=13, fontweight="bold")
    path = out_dir / "calibration_scatter.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved {path}")


def plot_correlation_heatmap(thetas, stats, out_dir):
    n_params = thetas.shape[1]
    n_stats  = stats.shape[1]
    rho_matrix = np.zeros((n_stats, n_params))
    for i in range(n_stats):
        for j in range(n_params):
            rho_matrix[i, j], _ = spearmanr(thetas[:, j], stats[:, i])

    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    im = ax.imshow(rho_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Spearman rho")
    ax.set_xticks(range(n_params))
    ax.set_xticklabels(PARAM_NAMES, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(n_stats))
    ax.set_yticklabels(STAT_NAMES, fontsize=9)
    for i in range(n_stats):
        for j in range(n_params):
            colour = "white" if abs(rho_matrix[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{rho_matrix[i, j]:.2f}", ha="center",
                    va="center", fontsize=9, fontweight="bold", color=colour)
    ax.set_title("Rank Correlation: Stats <-> Parameters",
                 fontsize=12, fontweight="bold")
    ax.axhline(2.5, color="black", lw=1.5, ls="--")
    ax.text(n_params - 0.5, 1.0, "sync", fontsize=8, ha="right",
            fontstyle="italic", color="grey")
    ax.text(n_params - 0.5, 5.0, "IPL coupling\n(from calcium)", fontsize=8,
            ha="right", fontstyle="italic", color="grey")
    path = out_dir / "calibration_correlation.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved {path}")
    return rho_matrix


def plot_stat_histograms(stats, out_dir):
    n_stats = stats.shape[1]
    n_cols = 4; n_rows = (n_stats + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 3.5 * n_rows),
                             constrained_layout=True)
    axes_flat = axes.flatten()
    for j in range(len(axes_flat)):
        ax = axes_flat[j]
        if j < n_stats:
            col = stats[:, j]
            ax.hist(col, bins=30, color="slategrey", alpha=0.8,
                    edgecolor="none", density=True)
            ax.axvline(col.mean(), color="red", ls="--", lw=1.5,
                       label=f"mean={col.mean():.3f}")
            cv = col.std() / (abs(col.mean()) + 1e-12)
            ax.set_title(f"{STAT_NAMES[j]}\nCV={cv:.2f}", fontsize=10)
            ax.set_ylabel("Density"); ax.legend(fontsize=7)
        else:
            ax.set_visible(False)
    fig.suptitle("Stat Distributions Across Prior", fontsize=12, fontweight="bold")
    path = out_dir / "calibration_histograms.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def print_report(rho_matrix):
    print("\n" + "=" * 66)
    print("  CALIBRATION REPORT")
    print("=" * 66)

    print("\n  PARAMETER IDENTIFIABILITY:")
    for j, param in enumerate(PARAM_NAMES):
        col = np.abs(rho_matrix[:, j])
        max_rho = float(np.nanmax(col))
        best_idx = int(np.nanargmax(col))
        best_stat = STAT_NAMES[best_idx]
        if np.isnan(max_rho):
            status = "!! BROKEN — all rho values are NaN"
        elif max_rho < 0.10:
            status = "!! UNIDENTIFIABLE"
        elif max_rho < 0.25:
            status = "!! WEAKLY IDENTIFIABLE"
        elif max_rho < 0.40:
            status = "~  MODERATE"
        else:
            status = "ok GOOD"
        print(f"\n    {param}:")
        print(f"      Best |rho| = {max_rho:.3f}  (via {best_stat})")
        print(f"      Status:  {status}")

    print("\n  STAT INFORMATIVENESS:")
    for i, name in enumerate(STAT_NAMES):
        row = np.abs(rho_matrix[i, :])
        max_rho = float(np.nanmax(row))
        best_idx = int(np.nanargmax(row))
        best_param = PARAM_NAMES[best_idx]
        if np.isnan(max_rho):
            print(f"    {name:<26} !! BROKEN (all rho = NaN)")
        elif max_rho < 0.10:
            print(f"    {name:<26} !! uninformative (max |rho|={max_rho:.3f})")
        else:
            print(f"    {name:<26} ok (max |rho|={max_rho:.3f}, best: {best_param})")

    print("\n  REDUNDANCY CHECK (stat-stat profile |rho| > 0.85):")
    n_stats = rho_matrix.shape[0]
    found = False
    for i in range(n_stats):
        for k in range(i + 1, n_stats):
            r, _ = spearmanr(rho_matrix[i, :], rho_matrix[k, :])
            if abs(r) > 0.85:
                print(f"    {STAT_NAMES[i]} <-> {STAT_NAMES[k]}  (rho={r:.2f})")
                found = True
    if not found:
        print("    None found — all stats carry distinct information.")

    beta_idx = PARAM_NAMES.index("beta")
    n_beta_dominated = sum(
        1 for i in range(n_stats)
        if np.nanargmax(np.abs(rho_matrix[i, :])) == beta_idx
    )
    print(f"\n  BETA DOMINANCE: {n_beta_dominated}/{n_stats} stats have beta "
          f"as strongest correlate")
    if n_beta_dominated > n_stats * 0.6:
        print("    !! Still beta-dominated — consider more scale-invariant stats")
    else:
        print("    ok Good balance across parameters")

    print("\n" + "=" * 66)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Calibration check with calcium forward model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Parallel workers (set to CPU cores)")
    parser.add_argument("--output",    type=str, default="calibration")
    parser.add_argument("--calcium-params", type=str, default=None,
                        help="Path to calcium_params.json")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # ── Calcium params ────────────────────────────────────────────────────
    if args.calcium_params:
        with open(args.calcium_params) as f:
            calcium_params = json.load(f)
        print(f"Loaded calcium params from {args.calcium_params}")
    else:
        calcium_params = dict(DEFAULT_CALCIUM_PARAMS)
    print(f"Calcium params: {calcium_params}")

    # ── Configuration for N=52 / 4 clusters / scaled arena ────────────────
    sim_config = SimulationConfig(
        n_neurons    = 52,
        n_clusters   = 4,
        cluster_size = 13,
        arena_size   = 400.0,
        cluster_radius = 80.0,
        min_cluster_center_distance = 120.0,
        seed         = args.seed,
    )
    inf_config = InferenceConfig(
        prior_low  = [20.0, 5.0, 0.4],
        prior_high = [600.0, 30.0, 3.0],
    )
    prior_low  = np.array(inf_config.prior_low)
    prior_high = np.array(inf_config.prior_high)

    print(f"\nConfig: N={sim_config.n_neurons}, clusters={sim_config.n_clusters}, "
          f"arena={sim_config.arena_size}um")
    print(f"Prior: sigma [{prior_low[0]}, {prior_high[0]}], "
          f"k [{prior_low[1]}, {prior_high[1]}], "
          f"beta [{prior_low[2]}, {prior_high[2]}]")

    # ── Sample & simulate ─────────────────────────────────────────────────
    n = args.n_samples
    print(f"\nSampling {n} parameter vectors (LHS) ...")
    thetas = sample_prior_lhs(n, prior_low, prior_high, seed=args.seed)

    worker_args = [
        (thetas[i], sim_config, i + args.seed, calcium_params)
        for i in range(n)
    ]

    stats = np.zeros((n, len(STAT_NAMES)), dtype=np.float32)
    t0 = time.time()

    if args.n_workers > 1:
        print(f"Running on {args.n_workers} workers ...\n")
        with Pool(args.n_workers) as pool:
            for i, result in enumerate(
                pool.imap(_simulate_worker, worker_args, chunksize=4)
            ):
                stats[i] = result
                if (i + 1) % max(1, n // 20) == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    eta = (n - i - 1) / rate
                    print(f"  {i+1}/{n} done  "
                          f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
    else:
        print("Running sequentially ...\n")
        for i in range(n):
            try:
                stats[i] = _simulate_worker(worker_args[i])
            except Exception as e:
                print(f"  [WARNING] Sim {i} failed: {e}")
                stats[i] = np.nan
            if (i + 1) % max(1, n // 10) == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n - i - 1) / rate
                print(f"  {i+1}/{n} done  "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    total_time = time.time() - t0
    print(f"\nSimulations complete in {total_time/60:.1f} min")

    # Drop failed simulations
    valid = ~np.any(np.isnan(stats), axis=1)
    n_valid = valid.sum()
    print(f"{n_valid}/{n} simulations succeeded.")
    thetas = thetas[valid]; stats = stats[valid]

    # ── Save ──────────────────────────────────────────────────────────────
    np.savez(out / "calibration_data.npz", thetas=thetas, stats=stats,
             param_names=PARAM_NAMES, stat_names=STAT_NAMES)

    # ── Figures ───────────────────────────────────────────────────────────
    print("\nGenerating figures ...")
    plot_scatter_grid(thetas, stats, out)
    rho = plot_correlation_heatmap(thetas, stats, out)
    plot_stat_histograms(stats, out)
    print_report(rho)


if __name__ == "__main__":
    main()
