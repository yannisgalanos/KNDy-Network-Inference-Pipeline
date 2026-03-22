"""
calibrate_calcium.py
--------------------
Estimate calcium forward model parameters from real recordings.

Analyses a real dF/F trace to estimate observation noise and amplitude,
then saves a JSON config to use during SNPE training.

Usage
~~~~~
  python calibrate_calcium.py --input recording.csv --dt-imaging 6.0
  python calibrate_calcium.py --input recording.csv --dt-imaging 0.1 --output calcium_params.json

Output
~~~~~~
  calcium_params.json — parameters for calcium_forward_model()
  calibration_preview.png — diagnostic figure
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from calcium_model import estimate_noise_std, match_noise_to_data


def load_traces(path: str) -> np.ndarray:
    """Load calcium traces from CSV or .npy."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path).astype(np.float64)
    else:
        with open(path, 'r') as f:
            first_line  = f.readline().strip()
            second_line = f.readline().strip()
        skip = 0
        try:
            np.array(first_line.replace(',', ' ').split(), dtype=float)
        except ValueError:
            skip = 1
            try:
                np.array(second_line.replace(',', ' ').split(), dtype=float)
            except ValueError:
                skip = 2
        delimiter = '\t' if path.suffix == '.tsv' else ','
        data = np.loadtxt(path, delimiter=delimiter, skiprows=skip)
        if data.shape[1] > 2:
            first_col = data[:, 0]
            if np.all(np.diff(first_col) > 0):
                data = data[:, 1:]
        return data


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate calcium forward model from real data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to real calcium traces (CSV or .npy)")
    parser.add_argument("--dt-imaging", type=float, required=True,
                        help="Imaging frame period in seconds")
    parser.add_argument("--output", type=str, default="calcium_params.json",
                        help="Output JSON path")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    traces = load_traces(args.input)
    T, N = traces.shape
    duration_min = T * args.dt_imaging / 60.0
    print(f"  {T} frames x {N} neurons, duration = {duration_min:.1f} min")
    print(f"  dF/F range: [{traces.min():.4f}, {traces.max():.4f}]")
    print(f"  dF/F mean:  {traces.mean():.4f}")

    # ── Estimate parameters ───────────────────────────────────────────────
    params = match_noise_to_data(traces)
    params["dt_imaging"] = args.dt_imaging
    params["tau_rise"]   = 0.18    # GCaMP6s fixed
    params["tau_decay"]  = 2.6     # GCaMP6s fixed

    print(f"\nEstimated parameters:")
    for k, v in params.items():
        print(f"  {k:<16} = {v:.4f}" if isinstance(v, float) else f"  {k:<16} = {v}")

    # ── Per-neuron noise analysis ─────────────────────────────────────────
    per_neuron_noise = []
    for j in range(N):
        diffs = np.diff(traces[:, j])
        mad = np.median(np.abs(diffs - np.median(diffs)))
        per_neuron_noise.append(mad / 0.6745 / np.sqrt(2))
    per_neuron_noise = np.array(per_neuron_noise)
    print(f"\nPer-neuron noise std:")
    print(f"  mean = {per_neuron_noise.mean():.4f}")
    print(f"  range = [{per_neuron_noise.min():.4f}, {per_neuron_noise.max():.4f}]")

    # ── Save ──────────────────────────────────────────────────────────────
    with open(args.output, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nSaved to {args.output}")

    # ── Preview figure ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)

    # Population mean trace
    t_min = np.arange(T) * args.dt_imaging / 60.0
    pop = traces.mean(axis=1)
    axes[0, 0].plot(t_min, pop, color="steelblue", lw=0.8)
    axes[0, 0].set_xlabel("Time (min)"); axes[0, 0].set_ylabel("Mean dF/F")
    axes[0, 0].set_title("Population mean trace")

    # A few individual neurons
    n_show = min(5, N)
    for j in range(n_show):
        axes[0, 1].plot(t_min, traces[:, j] + j * 0.5, lw=0.5, label=f"N{j}")
    axes[0, 1].set_xlabel("Time (min)"); axes[0, 1].set_ylabel("dF/F (stacked)")
    axes[0, 1].set_title(f"Example neurons (first {n_show})")

    # Noise distribution
    axes[1, 0].hist(per_neuron_noise, bins=20, color="slategrey",
                    alpha=0.8, edgecolor="none")
    axes[1, 0].axvline(params["noise_std"], color="red", ls="--", lw=1.5,
                       label=f"global = {params['noise_std']:.4f}")
    axes[1, 0].set_xlabel("Noise std"); axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Per-neuron noise estimates"); axes[1, 0].legend()

    # dF/F histogram
    axes[1, 1].hist(traces.ravel(), bins=100, color="slategrey",
                    alpha=0.8, edgecolor="none", density=True)
    axes[1, 1].axvline(params["baseline"], color="orange", ls="--", lw=1.5,
                       label=f"baseline = {params['baseline']:.3f}")
    axes[1, 1].set_xlabel("dF/F"); axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title("dF/F distribution"); axes[1, 1].legend()

    fig.suptitle("Calcium Forward Model Calibration",
                 fontsize=14, fontweight="bold")
    fig_path = args.output.replace(".json", "_preview.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Preview saved to {fig_path}")

    print(f"\nTo train with these parameters:")
    print(f"  python train_model.py --calcium-params {args.output} --num-simulations 1500")


if __name__ == "__main__":
    main()
