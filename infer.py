"""
infer.py
--------
Load a trained SNPE model and infer parameters from calcium imaging data.

Accepts continuous dF/F traces (CSV or .npy) — the same representation
the model was trained on via the calcium forward model.

Usage
~~~~~
  # Infer from real calcium data (CSV):
  python infer.py --model-dir models/ --traces data/recording.csv

  # Infer from real calcium data (npy):
  python infer.py --model-dir models/ --traces data/calcium.npy

  # Quick test with synthetic data:
  python infer.py --model-dir models/ --demo

  # Skip PPC:
  python infer.py --model-dir models/ --traces data/recording.csv --skip-ppc

Output
~~~~~~
  <o>/x_observed.npy         — 8-D summary statistics
  <o>/posterior_samples.npy   — (n, 4) posterior samples
  <o>/inference_summary.png   — raster + posterior marginals
  <o>/ppc_checks.png          — posterior predictive checks
"""

from __future__ import annotations

import argparse
import copy
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import SimulationConfig, InferenceConfig
from simulation import simulate_kndy_network, generate_neuron_positions
from statistics import compute_summary_statistics, detect_sync_events, STAT_NAMES
from calcium_model import calcium_forward_model, match_noise_to_data
from snpe_model import SNPEInference


def _print(msg: str):
    print(msg, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_traces(path: str) -> np.ndarray:
    """
    Load calcium traces from CSV or .npy.

    Returns (T, N) continuous array.
    """
    path = Path(path)

    if path.suffix == ".npy":
        traces = np.load(path).astype(np.float64)
    elif path.suffix in (".csv", ".tsv", ".txt"):
        # Detect header
        with open(path, 'r') as f:
            first_line = f.readline().strip()
        skip = 0
        try:
            np.array(first_line.replace(',', ' ').split(), dtype=float)
        except ValueError:
            skip = 1

        delimiter = '\t' if path.suffix == '.tsv' else ','
        traces = np.loadtxt(path, delimiter=delimiter, skiprows=skip)

        # Drop time index column if present
        if traces.shape[1] > 2:
            first_col = traces[:, 0]
            if np.all(np.diff(first_col) > 0):
                _print(f"  First column is time index — dropping it")
                traces = traces[:, 1:]
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    T, N = traces.shape
    if T < N:
        raise ValueError(
            f"Traces shape {traces.shape}: {T} time steps, {N} neurons. "
            f"Probably transposed.")

    _print(f"[Data] {T} time steps x {N} neurons  "
           f"|  range: [{traces.min():.3f}, {traces.max():.3f}]")
    return traces


def generate_synthetic_calcium(
    sim_config: SimulationConfig,
    calcium_params: dict,
) -> np.ndarray:
    """Generate synthetic calcium traces for testing."""
    cfg = copy.copy(sim_config)
    _print("[Demo] Generating synthetic calcium data ...")
    positions = generate_neuron_positions(cfg)
    result    = simulate_kndy_network(cfg, positions=positions)
    T, N      = result["spikes"].shape
    _print(f"[Demo] Simulated {T} steps x {N} neurons  "
           f"|  mean spike rate = {result['spikes'].mean():.4f}")

    calcium = calcium_forward_model(
        result["spikes"], dt_fine=cfg.dt, seed=cfg.seed, **calcium_params
    )
    _print(f"[Demo] Calcium traces: {calcium.shape}  "
           f"|  range: [{calcium.min():.3f}, {calcium.max():.3f}]")
    return calcium


# ─────────────────────────────────────────────────────────────────────────────
# PPC
# ─────────────────────────────────────────────────────────────────────────────

def run_ppc(
    samples: np.ndarray,
    x_obs: np.ndarray,
    sim_config: SimulationConfig,
    calcium_params: dict,
    n_sims: int = 20,
) -> dict:
    """
    Posterior predictive checks with calcium forward model.
    """
    _print(f"\n[PPC] Running {n_sims} posterior predictive simulations ...")

    rng = np.random.default_rng(99)
    idx = rng.choice(len(samples), size=min(n_sims, len(samples)), replace=False)
    ppc_stats = []

    for k, i in enumerate(idx):
        sigma, mean_deg, cluster_str, beta_val = samples[i]
        cfg = SimulationConfig(
            n_neurons=sim_config.n_neurons,
            n_clusters=sim_config.n_clusters,
            cluster_size=sim_config.cluster_size,
            arena_size=sim_config.arena_size,
            cluster_radius=sim_config.cluster_radius,
            min_cluster_center_distance=sim_config.min_cluster_center_distance,
            sigma_spatial=float(sigma), mean_degree=float(mean_deg),
            cluster_strength=float(cluster_str), beta_val=float(beta_val),
            h0=sim_config.h0, w_NKB=sim_config.w_NKB,
            w_Dyn=sim_config.w_Dyn, dt=sim_config.dt,
            T=sim_config.T, burnin=sim_config.burnin,
            seed=int(k),
        )
        positions = generate_neuron_positions(cfg)
        result = simulate_kndy_network(cfg, positions=positions)

        # Apply same calcium forward model used during training
        calcium = calcium_forward_model(
            result["spikes"], dt_fine=cfg.dt, seed=int(k), **calcium_params
        )
        ppc_stats.append(
            compute_summary_statistics(calcium, dt=calcium_params["dt_imaging"])
        )
        if (k + 1) % max(1, n_sims // 4) == 0:
            _print(f"  {k+1}/{n_sims} done")

    ppc = np.array(ppc_stats)

    _print("\n  PPC vs observed:")
    _print(f"  {'Statistic':<26} {'Observed':>10} {'PPC mean':>10} "
           f"{'PPC std':>10} {'z-score':>9}")
    _print("  " + "-" * 68)
    z_scores = {}
    for j, name in enumerate(STAT_NAMES):
        obs = float(x_obs[j])
        mu  = float(ppc[:, j].mean())
        sd  = float(ppc[:, j].std()) + 1e-8
        z   = (obs - mu) / sd
        z_scores[name] = z
        flag = "ok" if abs(z) < 2.0 else "!!"
        _print(f"  {name:<26} {obs:>10.4f} {mu:>10.4f} {sd:>10.4f} "
               f"{z:>8.2f}  {flag}")

    return {"ppc_stats": ppc, "z_scores": z_scores}


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def make_summary_figure(
    traces: np.ndarray,
    x_obs: np.ndarray,
    samples: np.ndarray,
    dt_imaging: float,
    inf_config: InferenceConfig,
    out_dir: Path,
):
    """Summary figure: traces + population + stats + posterior marginals."""
    fig = plt.figure(figsize=(20, 11))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)
    T, N = traces.shape
    t_ax = np.arange(T) * dt_imaging / 60.0  # convert to minutes

    # Row 0: traces heatmap + population mean + bar chart
    ax_r = fig.add_subplot(gs[0, :2])
    # Normalise each neuron for display
    traces_norm = traces.copy()
    for j in range(N):
        col = traces[:, j]
        mn, mx = col.min(), col.max()
        if mx - mn > 1e-12:
            traces_norm[:, j] = (col - mn) / (mx - mn)
    ax_r.imshow(traces_norm.T, aspect="auto", cmap="hot",
                interpolation="none", extent=[0, t_ax[-1], -0.5, N - 0.5])
    ax_r.set_xlabel("Time (min)"); ax_r.set_ylabel("Neuron")
    ax_r.set_title(f"Calcium traces ({T} frames, {N} neurons)")

    ax_p = fig.add_subplot(gs[0, 2])
    pop = traces.mean(axis=1)
    ax_p.plot(t_ax, pop, color="steelblue", lw=0.9)
    ax_p.set_xlabel("Time (min)"); ax_p.set_ylabel("Mean dF/F")
    ax_p.set_title("Population mean trace")

    ax_b = fig.add_subplot(gs[0, 3])
    n_sync = 3
    colours = ["steelblue"] * n_sync + ["darkorange"] * (len(STAT_NAMES) - n_sync)
    ax_b.barh(STAT_NAMES, x_obs, color=colours, alpha=0.8)
    ax_b.set_xlabel("Value"); ax_b.set_title("Summary statistics")
    from matplotlib.patches import Patch
    ax_b.legend(handles=[Patch(color="steelblue", label="sync"),
                         Patch(color="darkorange", label="IPL coupling")],
                fontsize=7, loc="lower right")

    # Row 1: posterior marginals
    names = inf_config.param_names
    units = ["um", "", "", ""]
    for k, (name, unit) in enumerate(zip(names, units)):
        ax = fig.add_subplot(gs[1, k])
        col = samples[:, k]
        ax.hist(col, bins=45, color="mediumslateblue",
                alpha=0.8, density=True, edgecolor="none")
        ax.axvline(col.mean(), color="red", lw=1.5, ls="--",
                   label=f"mean = {col.mean():.2f}")
        lo, hi = np.percentile(col, [2.5, 97.5])
        ax.axvspan(lo, hi, alpha=0.12, color="red", label="95% CI")
        ax.set_xlabel(f"{name}{' (' + unit + ')' if unit else ''}")
        ax.set_ylabel("Density"); ax.set_title(f"Posterior: {name}")
        ax.legend(fontsize=7)

    fig.suptitle("KNDy Parameter Inference  (amortised SNPE)",
                 fontsize=14, fontweight="bold", y=1.01)
    path = out_dir / "inference_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _print(f"  Figure saved to {path}")


def make_ppc_figure(ppc_stats, x_obs, out_dir):
    """PPC histograms."""
    n_stats = len(STAT_NAMES)
    n_cols = 4
    n_rows = (n_stats + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_flat = axes.flatten()
    for j in range(len(axes_flat)):
        ax = axes_flat[j]
        if j < n_stats:
            obs = float(x_obs[j])
            ax.hist(ppc_stats[:, j], bins=25, color="slategrey",
                    alpha=0.75, density=True, edgecolor="none", label="PPC")
            ax.axvline(obs, color="crimson", lw=2, label="observed")
            ax.set_title(STAT_NAMES[j], fontsize=10)
            ax.set_ylabel("Density", fontsize=8); ax.legend(fontsize=7)
        else:
            ax.set_visible(False)
    fig.suptitle("Posterior Predictive Checks", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "ppc_checks.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _print(f"  PPC figure saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Infer KNDy parameters from calcium imaging data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Directory containing trained SNPE model")
    parser.add_argument("--traces", type=str, default=None,
                        help="Path to calcium traces (CSV or .npy)")
    parser.add_argument("--dt-imaging", type=float, default=None,
                        help="Imaging frame period in seconds "
                             "(default: read from saved model params)")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory")
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="Number of posterior samples")
    parser.add_argument("--n-ppc", type=int, default=20,
                        help="Number of PPC simulations")
    parser.add_argument("--skip-ppc", action="store_true")
    parser.add_argument("--demo", action="store_true",
                        help="Use synthetic calcium data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # ── Configs ───────────────────────────────────────────────────────────
    sim_config = SimulationConfig(
        n_neurons    = 52,
        n_clusters   = 4,
        cluster_size = 13,
        arena_size   = 360.0,
        cluster_radius = 135.0,
        min_cluster_center_distance = 180.0,
        seed         = args.seed,
    )
    inf_config = InferenceConfig(
        prior_low  = [30.0, 8.0, 0.0, 0.4],
        prior_high = [400.0, 25.0, 0.9, 3.0],
    )

    # ── Load model ────────────────────────────────────────────────────────
    _print("=" * 60)
    _print("  KNDy Parameter Inference  (amortised SNPE)")
    _print("=" * 60)
    t_total = time.time()

    snpe = SNPEInference.load(args.model_dir, sim_config, inf_config)
    calcium_params = snpe.calcium_params

    # Override dt_imaging if specified
    if args.dt_imaging is not None:
        calcium_params["dt_imaging"] = args.dt_imaging

    dt_imaging = calcium_params["dt_imaging"]
    _print(f"  Calcium model: dt_imaging={dt_imaging}s, "
           f"noise_std={calcium_params['noise_std']}, "
           f"amplitude={calcium_params['amplitude']}")

    # ── Load or generate data ─────────────────────────────────────────────
    if args.traces is not None:
        traces = load_traces(args.traces)
    elif args.demo:
        traces = generate_synthetic_calcium(sim_config, calcium_params)
    else:
        raise ValueError("Provide --traces or --demo.")

    # ── Compute summary statistics ────────────────────────────────────────
    _print("\n[Stage 1] Computing summary statistics ...")
    t0 = time.time()
    x_obs = compute_summary_statistics(traces, dt=dt_imaging)
    _print(f"  Done in {time.time()-t0:.1f}s")
    _print(f"  {'Statistic':<28} {'Value':>10}")
    _print("  " + "-" * 40)
    for name, val in zip(STAT_NAMES, x_obs):
        _print(f"  {name:<28} {val:>10.4f}")

    # ── Condition & sample ────────────────────────────────────────────────
    _print("\n[Stage 2] Conditioning posterior on observation ...")
    snpe.condition_on(x_obs)
    samples = snpe.sample_posterior(args.n_samples)

    summary = snpe.posterior_summary(samples)
    _print(f"\n  Posterior summary ({args.n_samples} samples):")
    _print(f"  {'Parameter':<22} {'Mean':>8} {'Std':>8} {'2.5%':>8} {'97.5%':>8}")
    _print("  " + "-" * 56)
    for name, s in summary.items():
        lo, hi = s["ci_95"]
        _print(f"  {name:<22} {s['mean']:>8.3f} {s['std']:>8.3f} "
               f"{lo:>8.3f} {hi:>8.3f}")

    # ── PPC ───────────────────────────────────────────────────────────────
    ppc_result = None
    if not args.skip_ppc:
        ppc_result = run_ppc(
            samples, x_obs, sim_config, calcium_params, n_sims=args.n_ppc
        )

    # ── Save ──────────────────────────────────────────────────────────────
    np.save(out / "x_observed.npy", x_obs)
    np.save(out / "posterior_samples.npy", samples)
    _print(f"\n  Samples saved to {out}/")

    # ── Figures ───────────────────────────────────────────────────────────
    make_summary_figure(traces, x_obs, samples, dt_imaging, inf_config, out)
    if ppc_result is not None:
        make_ppc_figure(ppc_result["ppc_stats"], x_obs, out)

    _print(f"\n{'=' * 60}")
    _print(f"  Inference complete in {(time.time()-t_total)/60:.2f} min")
    _print(f"  Output: {out.resolve()}")


if __name__ == "__main__":
    main()
