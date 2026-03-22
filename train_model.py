"""
train_model.py
--------------
Train the amortised SNPE posterior estimator from the prior.

The simulator now applies a GCaMP6s calcium forward model so that
simulated summary statistics match the representation of real data.

Usage
~~~~~
  # Default GCaMP6s params, 10Hz imaging:
  python train_model.py --num-simulations 1500 --n-workers 16

  # Match a specific recording's imaging rate and noise:
  python train_model.py --num-simulations 1500 --dt-imaging 6.0 --noise-std 0.03

  # Use calibrated params from calibrate_calcium.py:
  python train_model.py --num-simulations 1500 --calcium-params calcium_params.json

Output
~~~~~~
  <o>/snpe_posterior.pt          — trained density estimator
  <o>/snpe_standardize.npz       — standardization parameters
  <o>/snpe_calcium_params.json   — calcium forward model params used
"""

import argparse
import json
from pathlib import Path
from config import SimulationConfig, InferenceConfig
from snpe_model import SNPEInference, DEFAULT_CALCIUM_PARAMS


def main():
    parser = argparse.ArgumentParser(
        description="Train amortised SNPE model with calcium forward model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-simulations", type=int, default=1500,
                        help="Total simulations from the prior")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Parallel workers (set to number of CPU cores)")
    parser.add_argument("--output", type=str, default="models",
                        help="Directory to save trained model")
    parser.add_argument("--seed", type=int, default=42)

    # Calcium forward model parameters
    parser.add_argument("--calcium-params", type=str, default=None,
                        help="Path to JSON file with calcium params "
                             "(from calibrate_calcium.py)")
    parser.add_argument("--dt-imaging", type=float, default=None,
                        help="Imaging frame period in seconds (overrides JSON)")
    parser.add_argument("--amplitude", type=float, default=None,
                        help="Peak dF/F per spike (overrides JSON)")
    parser.add_argument("--noise-std", type=float, default=None,
                        help="Observation noise std (overrides JSON)")
    args = parser.parse_args()

    # ── Build calcium params ──────────────────────────────────────────────
    if args.calcium_params is not None:
        with open(args.calcium_params) as f:
            calcium_params = json.load(f)
        print(f"Loaded calcium params from {args.calcium_params}")
    else:
        calcium_params = dict(DEFAULT_CALCIUM_PARAMS)

    # CLI overrides
    if args.dt_imaging is not None:
        calcium_params["dt_imaging"] = args.dt_imaging
    if args.amplitude is not None:
        calcium_params["amplitude"] = args.amplitude
    if args.noise_std is not None:
        calcium_params["noise_std"] = args.noise_std

    print(f"Calcium forward model params:")
    for k, v in calcium_params.items():
        print(f"  {k:<16} = {v}")

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

    # ── Train & save ──────────────────────────────────────────────────────
    snpe = SNPEInference(sim_config, inf_config, calcium_params=calcium_params)
    snpe.train(num_simulations=args.num_simulations, n_workers=args.n_workers)
    snpe.save(args.output)

    print(f"\nDone. Model saved to {args.output}/")
    print(f"To infer parameters from data:")
    print(f"  python infer.py --model-dir {args.output} --traces your_data.csv")


if __name__ == "__main__":
    main()
