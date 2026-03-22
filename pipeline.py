"""
pipeline.py
-----------
KNDy Network Connectivity Inference Pipeline.

Stages
------
  0  Load / generate spike data and neuron positions
  1  Compute 8-D summary statistics from observed data
     (3 sync-event features + 5 scale-invariant IPL coupling features)
  2  SBI-SNPE posterior inference -> posterior over [sigma, k, c, beta]
  3  Posterior-predictive checks (PPC)

CLI usage
---------
  python pipeline.py --demo
  python pipeline.py --spikes data/spikes.npy
  python pipeline.py --demo --skip-inference

Options
-------
  --spikes          PATH   (T, N) binary numpy array
  --positions       PATH   (N, 2) position array in um  [optional]
  --output          DIR    directory for results and figures  [default: results]
  --num-simulations INT    total SNPE simulations  [default: 500]
  --num-rounds      INT    number of sequential SNPE rounds  [default: 3]
  --skip-inference         skip SNPE (Stage 1 only)
  --demo                   generate synthetic data and run full pipeline
  --seed            INT    random seed
"""

from __future__ import annotations

import argparse
import copy
import time
import warnings
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import SimulationConfig, InferenceConfig
from simulation import simulate_kndy_network, generate_neuron_positions
from statistics import (
    STAT_NAMES,
    compute_summary_statistics,
    detect_sync_events,
)


# ─────────────────────────────────────────────────────────────────────────────
# SBI-SNPE inference wrapper
# ─────────────────────────────────────────────────────────────────────────────

try:
    import torch
    from sbi.inference import SNPE
    from sbi.utils import BoxUniform
    _SBI_AVAILABLE = True
except ImportError:
    _SBI_AVAILABLE = False
    warnings.warn(
        "torch / sbi not installed.  "
        "Install with:  pip install torch sbi\n"
        "Inference stages will be unavailable."
    )


class SNPEInference:
    """
    Sequential Neural Posterior Estimation (SNPE-C / APT) via ``sbi``.

    Inferred parameters theta = [sigma, k, c, beta]:
        sigma  sigma_spatial    : spatial connectivity length-scale (um)
        k      mean_degree      : mean synaptic partners per neuron
        c      cluster_strength : within-cluster coupling boost [0, 1]
        beta   beta_val         : Ising inverse temperature
    """

    def __init__(self, sim_config: SimulationConfig, inf_config: InferenceConfig):
        if not _SBI_AVAILABLE:
            raise ImportError("sbi is not installed.  Run: pip install torch sbi")
        self.sim_config = sim_config
        self.inf_config = inf_config

        low  = torch.tensor(inf_config.prior_low,  dtype=torch.float32)
        high = torch.tensor(inf_config.prior_high, dtype=torch.float32)
        self.prior = BoxUniform(low=low, high=high)

        self._inference  = SNPE(prior=self.prior)
        self.posterior_  = None
        self._x_obs_t    = None

        # Manual standardization (populated in train())
        self._x_mean: Optional["torch.Tensor"] = None
        self._x_std:  Optional["torch.Tensor"] = None

    def _simulate_one(self, theta_tensor: "torch.Tensor") -> "torch.Tensor":
        """Simulate one KNDy network from theta and return 8-D summary stats."""
        sigma, mean_deg, cluster_str, beta_val = theta_tensor.numpy()
        cfg = SimulationConfig(
            n_neurons        = self.sim_config.n_neurons,
            n_clusters       = self.sim_config.n_clusters,
            arena_size       = self.sim_config.arena_size,
            sigma_spatial    = float(np.clip(sigma,       1.0, 1e4)),
            mean_degree      = float(np.clip(mean_deg,    0.5,
                                             self.sim_config.n_neurons - 1)),
            cluster_strength = float(np.clip(cluster_str, 0.0, 1.0)),
            beta_val         = float(np.clip(beta_val,    0.01, 10.0)),
            h0               = self.sim_config.h0,
            w_NKB            = self.sim_config.w_NKB,
            w_Dyn            = self.sim_config.w_Dyn,
            dt               = self.sim_config.dt,
            T                = self.sim_config.T,
            burnin           = self.sim_config.burnin,
            seed             = None,
        )
        positions = generate_neuron_positions(cfg)
        result    = simulate_kndy_network(cfg, positions=positions)
        stats     = compute_summary_statistics(result["spikes"], dt=cfg.dt)
        return torch.tensor(stats, dtype=torch.float32)

    def _standardize_x(self, xs: "torch.Tensor") -> "torch.Tensor":
        """Standardize summary stats with a floor on std to prevent div-by-zero."""
        if self._x_mean is None:
            self._x_mean = xs.mean(dim=0)
            self._x_std  = xs.std(dim=0)
            # Floor: if a stat has zero (or near-zero) variance, set std=1
            # so it passes through unscaled rather than exploding
            self._x_std  = torch.where(self._x_std > 1e-8, self._x_std,
                                       torch.ones_like(self._x_std))
        return (xs - self._x_mean) / self._x_std

    def train(
        self,
        x_observed:      np.ndarray,
        num_simulations: Optional[int] = None,
        num_rounds:      Optional[int] = None,
        verbose:         bool = True,
    ) -> "SNPEInference":
        n_sims   = num_simulations or self.inf_config.abc_population_size * 10
        n_rounds = num_rounds      or self.inf_config.abc_n_rounds

        x_obs_raw = torch.tensor(x_observed, dtype=torch.float32)
        proposal  = self.prior

        for rnd in range(n_rounds):
            sims_this_round = n_sims // n_rounds
            if verbose:
                _print(f"\n=== SNPE Round {rnd+1}/{n_rounds}  "
                       f"({sims_this_round} simulations) ===")

            t0 = time.time()
            thetas = proposal.sample((sims_this_round,))
            xs_raw = torch.stack([self._simulate_one(t) for t in thetas])

            # Manual standardization on round 0; reuse same mean/std after
            xs = self._standardize_x(xs_raw)

            # Log any zero-variance stats (informational)
            if verbose and rnd == 0:
                raw_std = xs_raw.std(dim=0)
                zero_var = (raw_std < 1e-8).nonzero(as_tuple=True)[0]
                if len(zero_var) > 0:
                    from statistics import STAT_NAMES
                    names = [STAT_NAMES[i] for i in zero_var.tolist()]
                    _print(f"  Warning: zero-variance stats in this batch: "
                           f"{names} — these will be passed through unscaled")

            self._inference.append_simulations(thetas, xs, proposal=proposal)
            density_estimator = self._inference.train(
                show_train_summary=(rnd == n_rounds - 1),
            )
            self.posterior_ = self._inference.build_posterior(density_estimator)

            # Condition on observed data (standardized with same transform)
            self._x_obs_t = self._standardize_x(x_obs_raw.unsqueeze(0)).squeeze(0)
            self.posterior_.set_default_x(self._x_obs_t)

            if verbose:
                _print(f"  Round {rnd+1} done in {time.time()-t0:.1f}s")

            if rnd < n_rounds - 1:
                proposal = self.posterior_

        return self

    def sample_posterior(self, n_samples: Optional[int] = None) -> np.ndarray:
        if self.posterior_ is None:
            raise RuntimeError("Call train() first.")
        n = n_samples or self.inf_config.abc_n_posterior_samples
        return self.posterior_.sample((n,)).numpy()

    def posterior_summary(
        self, samples: Optional[np.ndarray] = None
    ) -> Dict[str, dict]:
        if samples is None:
            samples = self.sample_posterior()
        names = self.inf_config.param_names
        return {
            name: {
                "mean":   float(samples[:, i].mean()),
                "std":    float(samples[:, i].std()),
                "ci_95":  (float(np.percentile(samples[:, i], 2.5)),
                           float(np.percentile(samples[:, i], 97.5))),
                "median": float(np.median(samples[:, i])),
            }
            for i, name in enumerate(names)
        }


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline class
# ─────────────────────────────────────────────────────────────────────────────

class KNDyPipeline:
    """End-to-end inference pipeline for KNDy network structural connectivity."""

    def __init__(
        self,
        sim_config: SimulationConfig,
        inf_config: InferenceConfig,
        output_dir: str = "results",
    ):
        self.sim_config = sim_config
        self.inf_config = inf_config
        self.out        = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        self.spikes_obs:        Optional[np.ndarray] = None
        self.positions_obs:     Optional[np.ndarray] = None
        self.x_obs:             Optional[np.ndarray] = None
        self.snpe:              Optional[SNPEInference] = None
        self.posterior_samples: Optional[np.ndarray] = None
        self.posterior_summary_: Optional[dict]       = None

    # ── Stage 0 — Data loading ───────────────────────────────────────────

    def load_data(
        self,
        spikes:    np.ndarray,
        positions: Optional[np.ndarray] = None,
    ):
        if spikes.ndim != 2:
            raise ValueError(f"spikes must be 2-D, got shape {spikes.shape}.")
        T, N = spikes.shape
        if T == 0 or N == 0:
            raise ValueError(f"spikes has shape {spikes.shape} — both dims > 0.")
        if T < N:
            raise ValueError(
                f"spikes shape {spikes.shape}: {T} time steps, {N} neurons. "
                f"Probably transposed — pass spikes.T."
            )
        firing_rate = float(spikes.mean())
        if firing_rate == 0.0:
            raise ValueError("spikes is entirely zero.")
        unique_vals = np.unique(spikes)
        if unique_vals.max() > 1.0 or (len(unique_vals) > 10 and unique_vals.max() > 1.5):
            warnings.warn(
                f"spikes range [{unique_vals.min():.3f}, {unique_vals.max():.3f}]. "
                f"Expected binary (0/1).", UserWarning, stacklevel=2)
        if firing_rate > 0.70:
            warnings.warn(f"Mean firing rate {firing_rate:.2f} > 70%.",
                          UserWarning, stacklevel=2)

        self.spikes_obs    = spikes.astype(np.int8)
        self.positions_obs = None
        self.sim_config.n_neurons = N

        if positions is not None:
            if positions.ndim != 2 or positions.shape != (N, 2):
                raise ValueError(f"positions shape ({N}, 2) expected, got {positions.shape}.")
            self.positions_obs = positions.astype(float)
            _print("[Data] Neuron positions loaded.")

        _print(f"[Data] {T} time steps x {N} neurons  "
               f"|  mean rate = {firing_rate:.4f}")

    # ── Stage 1 — Observed summary statistics ────────────────────────────

    def compute_observed_stats(self) -> np.ndarray:
        _print("\n[Stage 1] Computing observed summary statistics ...")
        _print("  (includes IPL fitting — may take a moment)")
        t0 = time.time()
        self.x_obs = compute_summary_statistics(
            self.spikes_obs, dt=self.sim_config.dt
        )
        _print(f"  Done in {time.time()-t0:.1f}s")
        _print(f"  {'Statistic':<28} {'Value':>10}")
        _print("  " + "-" * 40)
        for name, val in zip(STAT_NAMES, self.x_obs):
            _print(f"  {name:<28} {val:>10.4f}")
        return self.x_obs

    # ── Stage 2 — SBI-SNPE ──────────────────────────────────────────────

    def run_snpe(
        self,
        num_simulations: Optional[int] = None,
        num_rounds:      Optional[int] = None,
    ) -> dict:
        _print("\n[Stage 2] SBI-SNPE posterior inference ...")
        if self.x_obs is None:
            self.compute_observed_stats()

        _print(
            f"  Prior:  sigma in [{self.inf_config.prior_low[0]:.0f}, "
            f"{self.inf_config.prior_high[0]:.0f}] um  |  "
            f"k in [{self.inf_config.prior_low[1]:.1f}, "
            f"{self.inf_config.prior_high[1]:.1f}]  |  "
            f"c in [{self.inf_config.prior_low[2]:.2f}, "
            f"{self.inf_config.prior_high[2]:.2f}]  |  "
            f"beta in [{self.inf_config.prior_low[3]:.2f}, "
            f"{self.inf_config.prior_high[3]:.2f}]"
        )

        self.snpe = SNPEInference(self.sim_config, self.inf_config)
        self.snpe.train(self.x_obs, num_simulations=num_simulations,
                        num_rounds=num_rounds, verbose=True)

        self.posterior_samples  = self.snpe.sample_posterior()
        self.posterior_summary_ = self.snpe.posterior_summary(self.posterior_samples)

        _print("\n  Posterior summary:")
        _print(f"  {'Parameter':<22} {'Mean':>8} {'Std':>8} {'2.5%':>8} {'97.5%':>8}")
        _print("  " + "-" * 56)
        for name, s in self.posterior_summary_.items():
            lo, hi = s["ci_95"]
            _print(f"  {name:<22} {s['mean']:>8.3f} {s['std']:>8.3f} "
                   f"{lo:>8.3f} {hi:>8.3f}")
        return self.posterior_summary_

    # ── Stage 3 — PPC ───────────────────────────────────────────────────

    def run_validation(self) -> dict:
        n = self.inf_config.n_validation_sims
        _print(f"\n[Stage 3] Posterior predictive checks ({n} simulations) ...")
        if self.posterior_samples is None:
            raise RuntimeError("Run Stage 2 first.")

        rng = np.random.default_rng(99)
        n_available = len(self.posterior_samples)
        idx = rng.choice(n_available, size=min(n, n_available), replace=False)
        ppc_stats = []

        for k, i in enumerate(idx):
            theta = self.posterior_samples[i]
            sigma, mean_deg, cluster_str, beta_val = theta
            cfg = SimulationConfig(
                n_neurons=self.sim_config.n_neurons,
                n_clusters=self.sim_config.n_clusters,
                arena_size=self.sim_config.arena_size,
                sigma_spatial=float(sigma), mean_degree=float(mean_deg),
                cluster_strength=float(cluster_str), beta_val=float(beta_val),
                h0=self.sim_config.h0, w_NKB=self.sim_config.w_NKB,
                w_Dyn=self.sim_config.w_Dyn, dt=self.sim_config.dt,
                T=self.sim_config.T, burnin=self.sim_config.burnin,
                seed=int(k),
            )
            positions = generate_neuron_positions(cfg)
            result = simulate_kndy_network(cfg, positions=positions)
            ppc_stats.append(compute_summary_statistics(result["spikes"], dt=cfg.dt))
            if (k + 1) % max(1, n // 4) == 0:
                _print(f"  {k+1}/{n} PPC simulations done")

        ppc = np.array(ppc_stats)

        _print("\n  PPC vs observed:")
        _print(f"  {'Statistic':<26} {'Observed':>10} {'PPC mean':>10} "
               f"{'PPC std':>10} {'z-score':>9}")
        _print("  " + "-" * 68)
        z_scores = {}
        for j, name in enumerate(STAT_NAMES):
            obs = float(self.x_obs[j])
            mu  = float(ppc[:, j].mean())
            sd  = float(ppc[:, j].std()) + 1e-8
            z   = (obs - mu) / sd
            z_scores[name] = z
            flag = "ok" if abs(z) < 2.0 else "!!"
            _print(f"  {name:<26} {obs:>10.4f} {mu:>10.4f} {sd:>10.4f} "
                   f"{z:>8.2f}  {flag}")
        return {"ppc_stats": ppc, "z_scores": z_scores}

    # ── Full pipeline ────────────────────────────────────────────────────

    def run(self, skip_inference=False, num_simulations=None, num_rounds=None):
        _print("=" * 66)
        _print("  KNDy Network Inference Pipeline  (SBI-SNPE)")
        _print("=" * 66)
        t_total = time.time()
        results = {}
        results["x_obs"] = self.compute_observed_stats()

        if not skip_inference:
            results["posterior_summary"] = self.run_snpe(
                num_simulations=num_simulations, num_rounds=num_rounds)
            results["validation"] = self.run_validation()

        _print(f"\n{'=' * 66}")
        _print(f"  Pipeline complete in {(time.time()-t_total)/60:.2f} min")
        _print(f"  Output directory: {self.out.resolve()}")
        self._save_arrays()
        self._make_figures(results, skip_inference)
        return results

    # ── Saving ───────────────────────────────────────────────────────────

    def _save_arrays(self):
        if self.x_obs is not None:
            np.save(self.out / "x_observed.npy", self.x_obs)
        if self.posterior_samples is not None:
            np.save(self.out / "posterior_samples.npy", self.posterior_samples)
        _print(f"  Arrays saved to {self.out}/")

    # ── Figures ──────────────────────────────────────────────────────────

    def _make_figures(self, results, skip_inference):
        _print("\n[Figures] Generating summary figure ...")
        has_posterior = not skip_inference and self.posterior_samples is not None
        n_rows = 2 if has_posterior else 1
        fig = plt.figure(figsize=(20, n_rows * 5 + 1))
        gs  = gridspec.GridSpec(n_rows, 4, figure=fig, hspace=0.45, wspace=0.38)

        T_show = min(600, self.spikes_obs.shape[0])
        t_ax   = np.arange(T_show) * self.sim_config.dt
        pop    = self.spikes_obs[:T_show].mean(axis=1)
        N      = self.sim_config.n_neurons

        # Row 0: raster + pop + bar chart
        ax_raster = fig.add_subplot(gs[0, :2])
        ax_raster.imshow(self.spikes_obs[:T_show].T, aspect="auto",
                         cmap="binary", interpolation="none",
                         extent=[0, t_ax[-1], -0.5, N - 0.5])
        _sync = detect_sync_events(self.spikes_obs[:T_show])
        for et in _sync["full_event_times"]:
            ax_raster.axvline(et * self.sim_config.dt, color="red", lw=0.6, alpha=0.5)
        ax_raster.set_xlabel("Time (s)"); ax_raster.set_ylabel("Neuron")
        ax_raster.set_title("Spike raster  (red = full sync)")

        ax_pop = fig.add_subplot(gs[0, 2])
        ax_pop.plot(t_ax, pop, color="steelblue", lw=0.9)
        ax_pop.axhline(0.30, ls="--", color="red", lw=0.8, label="full sync")
        ax_pop.axhline(0.12, ls="--", color="orange", lw=0.8, label="partial")
        ax_pop.set_xlabel("Time (s)"); ax_pop.set_ylabel("Pop. activity")
        ax_pop.set_title("Population firing rate"); ax_pop.legend(fontsize=7)

        ax_bar = fig.add_subplot(gs[0, 3])
        if self.x_obs is not None:
            n_sync = 3
            colours = ["steelblue"] * n_sync + ["darkorange"] * (len(STAT_NAMES) - n_sync)
            ax_bar.barh(STAT_NAMES, self.x_obs, color=colours, alpha=0.8)
            ax_bar.set_xlabel("Value")
            ax_bar.set_title("Observed summary statistics")
            from matplotlib.patches import Patch
            ax_bar.legend(handles=[Patch(color="steelblue", label="sync"),
                                   Patch(color="darkorange", label="IPL coupling")],
                          fontsize=7, loc="lower right")

        # Row 1: posterior marginals
        if has_posterior:
            names = self.inf_config.param_names
            units = ["um", "", "", ""]
            for k, (name, unit) in enumerate(zip(names, units)):
                ax_p = fig.add_subplot(gs[1, k])
                col = self.posterior_samples[:, k]
                ax_p.hist(col, bins=45, color="mediumslateblue",
                          alpha=0.8, density=True, edgecolor="none")
                ax_p.axvline(col.mean(), color="red", lw=1.5, ls="--",
                             label=f"mean = {col.mean():.2f}")
                lo, hi = np.percentile(col, [2.5, 97.5])
                ax_p.axvspan(lo, hi, alpha=0.12, color="red", label="95% CI")
                ax_p.set_xlabel(f"{name}{' (' + unit + ')' if unit else ''}")
                ax_p.set_ylabel("Density"); ax_p.set_title(f"Posterior: {name}")
                ax_p.legend(fontsize=7)

        fig.suptitle("KNDy Inference Pipeline  (SBI-SNPE)",
                     fontsize=14, fontweight="bold", y=1.01)
        fig_path = self.out / "pipeline_summary.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        _print(f"  Figure saved to {fig_path}")

        # Separate PPC figure
        if not skip_inference and "validation" in results:
            self._make_ppc_figure(results["validation"]["ppc_stats"])

    def _make_ppc_figure(self, ppc_stats):
        n_stats = len(STAT_NAMES)
        n_cols = 4
        n_rows = (n_stats + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes_flat = axes.flatten()
        for j in range(len(axes_flat)):
            ax = axes_flat[j]
            if j < n_stats:
                obs = float(self.x_obs[j])
                ax.hist(ppc_stats[:, j], bins=25, color="slategrey",
                        alpha=0.75, density=True, edgecolor="none", label="PPC")
                ax.axvline(obs, color="crimson", lw=2, label="observed")
                ax.set_title(STAT_NAMES[j], fontsize=10)
                ax.set_ylabel("Density", fontsize=8); ax.legend(fontsize=7)
            else:
                ax.set_visible(False)
        fig.suptitle("Posterior Predictive Checks", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig_path = self.out / "ppc_checks.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        _print(f"  PPC figure saved to {fig_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _print(msg: str):
    print(msg, flush=True)


def generate_synthetic_data(sim_config: SimulationConfig) -> dict:
    cfg = copy.copy(sim_config)
    _print("[Demo] Generating synthetic observed data ...")
    positions = generate_neuron_positions(cfg)
    result    = simulate_kndy_network(cfg, positions=positions)
    T, N      = result["spikes"].shape
    _print(f"[Demo] {T} x {N}  |  mean rate = {result['spikes'].mean():.4f}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="KNDy inference pipeline (SBI-SNPE)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--spikes",     type=str, default=None)
    p.add_argument("--positions",  type=str, default=None)
    p.add_argument("--output",     type=str, default="results")
    p.add_argument("--num-simulations", type=int, default=500)
    p.add_argument("--num-rounds", type=int, default=3)
    p.add_argument("--skip-inference", action="store_true")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_parser().parse_args()
    sim_config = SimulationConfig(seed=args.seed)
    inf_config = InferenceConfig(
        abc_population_size=50, abc_n_rounds=args.num_rounds,
        abc_n_posterior_samples=500, n_validation_sims=20,
        prior_low=[20.0, 8.0, 0.0, 0.4],
        prior_high=[800.0, 40.0, 0.9, 3.0],
    )
    pipeline = KNDyPipeline(sim_config, inf_config, output_dir=args.output)

    if args.spikes is not None:
        spikes = np.load(args.spikes)
        positions = np.load(args.positions) if args.positions else None
        pipeline.load_data(spikes, positions)
    else:
        if not args.demo:
            _print("No --spikes file; running demo mode.")
        synthetic = generate_synthetic_data(sim_config)
        pipeline.load_data(synthetic["spikes"], synthetic.get("positions"))

    pipeline.run(skip_inference=args.skip_inference,
                 num_simulations=args.num_simulations,
                 num_rounds=args.num_rounds)



if __name__ == "__main__":
    main()

