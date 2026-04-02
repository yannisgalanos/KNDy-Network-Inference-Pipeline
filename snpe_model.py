"""
snpe_model.py
-------------
Amortised SNPE posterior estimator for KNDy network parameters.

This module contains the SNPEInference class which handles:
  - Training from the prior (no observed data needed)
  - Saving / loading the trained model
  - Conditioning on any observation and sampling the posterior

Two companion scripts use this:
  train_model.py  — train and save the model
  infer.py        — load model, condition on data, sample posterior, PPC
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    import torch
    from sbi.inference import SNPE
    from sbi.utils import BoxUniform
    _SBI_AVAILABLE = True
except ImportError:
    _SBI_AVAILABLE = False
    warnings.warn(
        "torch / sbi not installed.  "
        "Install with:  pip install torch sbi"
    )

from config import SimulationConfig, InferenceConfig
from simulation import simulate_kndy_network, generate_neuron_positions
from statistics import compute_summary_statistics, STAT_NAMES
from calcium_model import calcium_forward_model


def _print(msg: str):
    print(msg, flush=True)


# Default calcium forward model parameters (GCaMP6s)
DEFAULT_CALCIUM_PARAMS = {
    "dt_imaging": 0.1,       # imaging frame period (seconds) — match your data
    "amplitude": 0.001,       # peak dF/F per spike
    "baseline": 0.0037,         # baseline fluorescence
    "noise_std": 0.0018,       # observation noise
    "tau_rise": 0.18,        # GCaMP6s rise (s)
    "tau_decay": 2.6,        # GCaMP6s decay (s)
}


def _simulate_worker(args):
    """
    Module-level worker for multiprocessing.

    Runs one KNDy simulation, applies calcium forward model,
    and returns summary stats as a numpy array.
    """
    theta, sim_config, seed, calcium_params = args
    sigma, mean_deg, beta_val = theta

    cfg = SimulationConfig(
        n_neurons        = sim_config.n_neurons,
        n_clusters       = sim_config.n_clusters,
        cluster_size     = sim_config.cluster_size,
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

    # Apply calcium forward model: binary spikes → synthetic dF/F
    calcium = calcium_forward_model(
        result["spikes"], dt_fine=cfg.dt, seed=seed, **calcium_params
    )

    # dt for stats is the imaging frame period (not the fine simulation dt)
    return compute_summary_statistics(calcium, dt=calcium_params["dt_imaging"])


class SNPEInference:
    """
    Amortised Neural Posterior Estimation for KNDy connectivity parameters.

    Inferred parameters theta = [sigma, k, c, beta]:
        sigma  sigma_spatial    : spatial connectivity length-scale (um)
        k      mean_degree      : mean synaptic partners per neuron
        beta   beta_val         : Ising inverse temperature

    Workflow
    ~~~~~~~~
    1. Train from the prior (no data needed):
         snpe = SNPEInference(sim_config, inf_config)
         snpe.train(num_simulations=1500)
         snpe.save("models/")

    2. Load and condition on any observation:
         snpe = SNPEInference.load("models/", sim_config, inf_config)
         snpe.condition_on(x_obs)
         samples = snpe.sample_posterior(1000)
    """

    def __init__(
        self,
        sim_config: SimulationConfig,
        inf_config: InferenceConfig,
        calcium_params: Optional[dict] = None,
    ):
        if not _SBI_AVAILABLE:
            raise ImportError("sbi is not installed.  Run: pip install torch sbi")
        self.sim_config = sim_config
        self.inf_config = inf_config
        self.calcium_params = calcium_params or dict(DEFAULT_CALCIUM_PARAMS)

        low  = torch.tensor(inf_config.prior_low,  dtype=torch.float32)
        high = torch.tensor(inf_config.prior_high, dtype=torch.float32)
        self.prior = BoxUniform(low=low, high=high)

        self._inference = SNPE(prior=self.prior)
        self.posterior_ = None
        self._x_obs_t   = None

        # Manual standardization (populated during train())
        self._x_mean: Optional["torch.Tensor"] = None
        self._x_std:  Optional["torch.Tensor"] = None

    # ── Simulation ───────────────────────────────────────────────────────

    def _simulate_one(self, theta_tensor: "torch.Tensor") -> "torch.Tensor":
        """Simulate one KNDy network, apply calcium model, return summary stats."""
        sigma, mean_deg, beta_val = theta_tensor.numpy()
        cfg = SimulationConfig(
            n_neurons        = self.sim_config.n_neurons,
            n_clusters       = self.sim_config.n_clusters,
            cluster_size     = self.sim_config.cluster_size,
            arena_size       = self.sim_config.arena_size,
            cluster_radius   = self.sim_config.cluster_radius,
            min_cluster_center_distance = self.sim_config.min_cluster_center_distance,
            sigma_spatial    = float(np.clip(sigma,       1.0, 1e4)),
            mean_degree      = float(np.clip(mean_deg,    0.5,
                                             self.sim_config.n_neurons - 1)),
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

        calcium = calcium_forward_model(
            result["spikes"], dt_fine=cfg.dt, **self.calcium_params
        )

        stats = compute_summary_statistics(
            calcium, dt=self.calcium_params["dt_imaging"]
        )
        return torch.tensor(stats, dtype=torch.float32)

    # ── Standardization ──────────────────────────────────────────────────

    def _standardize_x(self, xs: "torch.Tensor") -> "torch.Tensor":
        """Standardize summary stats; floor std to prevent div-by-zero."""
        if self._x_mean is None:
            self._x_mean = xs.mean(dim=0)
            self._x_std  = xs.std(dim=0)
            self._x_std  = torch.where(self._x_std > 1e-8, self._x_std,
                                       torch.ones_like(self._x_std))
        return (xs - self._x_mean) / self._x_std

    # ── Training (from prior only — no data needed) ──────────────────────

    def train(
        self,
        num_simulations: int = 1500,
        verbose: bool = True,
        n_workers: int = 1,
    ) -> "SNPEInference":
        """
        Train the density estimator from the prior.

        Single-round training: all simulations come from the prior,
        so the trained model is valid for any observation (fully amortised).

        Parameters
        ----------
        num_simulations : total number of simulations to run.
        verbose         : print progress.
        n_workers       : number of parallel workers (default 1 = sequential).
                          Set to number of CPU cores for ~linear speedup.
        """
        _print(f"\n=== SNPE Training ({num_simulations} simulations, "
               f"single round from prior, {n_workers} workers) ===")
        _print(f"  Prior:  sigma in [{self.inf_config.prior_low[0]:.0f}, "
               f"{self.inf_config.prior_high[0]:.0f}]  |  "
               f"k in [{self.inf_config.prior_low[1]:.1f}, "
               f"{self.inf_config.prior_high[1]:.1f}]  |  "
               f"beta in [{self.inf_config.prior_low[2]:.2f}, "
               f"{self.inf_config.prior_high[2]:.2f}]")

        t0 = time.time()
        thetas = self.prior.sample((num_simulations,))
        theta_list = [thetas[i].numpy() for i in range(num_simulations)]

        if n_workers > 1:
            from multiprocessing import Pool
            _print(f"  Running {num_simulations} simulations on {n_workers} cores ...")

            # Package config for the worker function
            worker_args = [
                (theta, self.sim_config, i, self.calcium_params)
                for i, theta in enumerate(theta_list)
            ]

            xs_raw_np = []
            with Pool(n_workers) as pool:
                for i, result in enumerate(
                    pool.imap(_simulate_worker, worker_args, chunksize=4)
                ):
                    xs_raw_np.append(result)
                    if verbose and (i + 1) % max(1, num_simulations // 20) == 0:
                        elapsed = time.time() - t0
                        rate = (i + 1) / elapsed
                        eta = (num_simulations - i - 1) / rate
                        _print(f"  {i+1}/{num_simulations} simulations  "
                               f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

            xs_raw = torch.tensor(np.array(xs_raw_np), dtype=torch.float32)
        else:
            xs_raw = []
            for i in range(num_simulations):
                xs_raw.append(self._simulate_one(thetas[i]))
                if verbose and (i + 1) % max(1, num_simulations // 10) == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    eta = (num_simulations - i - 1) / rate
                    _print(f"  {i+1}/{num_simulations} simulations  "
                           f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
            xs_raw = torch.stack(xs_raw)

        # Log zero-variance stats
        if verbose:
            raw_std = xs_raw.std(dim=0)
            zero_var = (raw_std < 1e-8).nonzero(as_tuple=True)[0]
            if len(zero_var) > 0:
                names = [STAT_NAMES[i] for i in zero_var.tolist()]
                _print(f"  Warning: zero-variance stats: {names} "
                       f"— passed through unscaled")

        # Standardize
        xs = self._standardize_x(xs_raw)

        # Train density estimator
        _print(f"\n  Training neural density estimator ...")
        self._inference.append_simulations(thetas, xs)
        density_estimator = self._inference.train(show_train_summary=True)
        self.posterior_ = self._inference.build_posterior(density_estimator)

        total_time = time.time() - t0
        _print(f"\n  Training complete in {total_time/60:.1f} min")

        return self

    # ── Conditioning & sampling ──────────────────────────────────────────

    def condition_on(self, x_observed: np.ndarray):
        """
        Condition the posterior on an observation.

        After calling this, sample_posterior() draws from p(theta | x_obs).

        Parameters
        ----------
        x_observed : (8,) summary statistic vector from real or synthetic data.
        """
        if self.posterior_ is None:
            raise RuntimeError("No trained posterior — call train() or load() first.")
        x_t = torch.tensor(x_observed, dtype=torch.float32)
        self._x_obs_t = self._standardize_x(x_t.unsqueeze(0)).squeeze(0)
        self.posterior_.set_default_x(self._x_obs_t)

    def sample_posterior(self, n_samples: int = 1000) -> np.ndarray:
        """Draw samples from p(theta | x_obs).  Returns (n, 4) array."""
        if self.posterior_ is None:
            raise RuntimeError("No trained posterior — call train() or load() first.")
        if self._x_obs_t is None:
            raise RuntimeError("No observation set — call condition_on(x) first.")
        return self.posterior_.sample((n_samples,)).numpy()

    def posterior_summary(
        self, samples: Optional[np.ndarray] = None, n_samples: int = 1000
    ) -> Dict[str, dict]:
        """Mean, std, 95% CI for each parameter."""
        if samples is None:
            samples = self.sample_posterior(n_samples)
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

    # ── Save / Load ──────────────────────────────────────────────────────

    def save(self, path: str):
        """
        Save the trained model.

        Files written:
          <path>/snpe_posterior.pt      — density estimator
          <path>/snpe_standardize.npz   — mean/std for input standardization
          <path>/snpe_calcium_params.npz — calcium forward model parameters
        """
        import json
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        if self.posterior_ is None:
            raise RuntimeError("Nothing to save — call train() first.")

        torch.save(self.posterior_, out / "snpe_posterior.pt")
        np.savez(
            out / "snpe_standardize.npz",
            x_mean=self._x_mean.numpy() if self._x_mean is not None else np.array([]),
            x_std=self._x_std.numpy()   if self._x_std  is not None else np.array([]),
        )
        with open(out / "snpe_calcium_params.json", "w") as f:
            json.dump(self.calcium_params, f, indent=2)
        _print(f"  Model saved to {out}/")

    @classmethod
    def load(cls, path: str, sim_config: SimulationConfig, inf_config: InferenceConfig) -> "SNPEInference":
        """
        Load a previously trained model.

        Usage
        ~~~~~
            snpe = SNPEInference.load("models/", sim_config, inf_config)
            snpe.condition_on(x_obs)
            samples = snpe.sample_posterior(1000)
        """
        import json
        out = Path(path)
        obj = cls.__new__(cls)
        obj.sim_config = sim_config
        obj.inf_config = inf_config

        obj.posterior_ = torch.load(out / "snpe_posterior.pt", weights_only=False)

        std_data = np.load(out / "snpe_standardize.npz")
        if len(std_data["x_mean"]) > 0:
            obj._x_mean = torch.tensor(std_data["x_mean"], dtype=torch.float32)
            obj._x_std  = torch.tensor(std_data["x_std"],  dtype=torch.float32)
        else:
            obj._x_mean = None
            obj._x_std  = None

        # Load calcium params (fall back to defaults if file missing)
        calcium_path = out / "snpe_calcium_params.json"
        if calcium_path.exists():
            with open(calcium_path) as f:
                obj.calcium_params = json.load(f)
        else:
            obj.calcium_params = dict(DEFAULT_CALCIUM_PARAMS)

        obj._x_obs_t   = None
        obj._inference  = None

        low  = torch.tensor(inf_config.prior_low,  dtype=torch.float32)
        high = torch.tensor(inf_config.prior_high, dtype=torch.float32)
        obj.prior = BoxUniform(low=low, high=high)

        _print(f"  Model loaded from {out}/")
        return obj
