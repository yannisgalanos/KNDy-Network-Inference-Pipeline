"""
calcium_model.py
----------------
Forward model converting binary Ising spike trains to synthetic GCaMP6s
calcium fluorescence traces.

This module sits between the simulator and the summary statistics:

    spikes (T_fine, N)  →  calcium_forward_model  →  dF/F (T_coarse, N)

Both simulated and real data are then in the same representation
(continuous calcium traces at the imaging frame rate), so summary
statistics computed on either are directly comparable.

GCaMP6s kinetics (Chen et al. 2013):
    τ_rise  ≈ 0.18 s
    τ_decay ≈ 2.6 s
    Peak ΔF/F per spike ≈ 0.2–0.3 (depends on expression level)

The kernel is a double exponential:
    K(t) = A × (exp(-t/τ_decay) - exp(-t/τ_rise))
normalised so that K peaks at 1.0.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def gcamp6s_kernel(
    dt_fine: float,
    tau_rise: float = 0.18,
    tau_decay: float = 2.6,
    duration: float = 15.0,
) -> np.ndarray:
    """
    Generate GCaMP6s impulse response kernel.

    Parameters
    ----------
    dt_fine   : simulation time step (seconds)
    tau_rise  : rise time constant (s)
    tau_decay : decay time constant (s)
    duration  : kernel length (s) — should be ~5× tau_decay

    Returns
    -------
    kernel : (n_samples,) normalised so peak = 1.0
    """
    t = np.arange(0, duration, dt_fine)
    kernel = np.exp(-t / tau_decay) - np.exp(-t / tau_rise)
    kernel[kernel < 0] = 0.0
    peak = kernel.max()
    if peak > 0:
        kernel /= peak
    return kernel


def calcium_forward_model(
    spikes: np.ndarray,
    dt_fine: float,
    dt_imaging: float = 0.1,
    amplitude: float = 0.01,
    baseline: float = 0.0037,
    noise_std: float = 0.0018,
    tau_rise: float = 0.18,
    tau_decay: float = 2.6,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Convert binary spike trains to synthetic calcium fluorescence traces.

    Pipeline:
        1. Convolve each neuron's spikes with GCaMP6s kernel
        2. Scale by amplitude and add baseline
        3. Downsample to imaging frame rate
        4. Add Gaussian observation noise

    Parameters
    ----------
    spikes     : (T_fine, N) binary int8 array from the Ising simulator
    dt_fine    : simulator time step in seconds
    dt_imaging : imaging frame period in seconds (default 0.1s = 10 Hz)
    amplitude  : peak dF/F per spike (default 0.25)
    baseline   : baseline fluorescence offset (default 0.0)
    noise_std  : std of Gaussian observation noise on dF/F (default 0.02)
    tau_rise   : GCaMP rise time constant in seconds
    tau_decay  : GCaMP decay time constant in seconds
    seed       : random seed for noise generation

    Returns
    -------
    calcium : (T_imaging, N) float64 — synthetic dF/F traces at imaging rate
    """
    T_fine, N = spikes.shape
    rng = np.random.default_rng(seed)

    # ── Step 1: Convolve with GCaMP kernel ────────────────────────────────
    kernel = gcamp6s_kernel(dt_fine, tau_rise, tau_decay)
    n_kernel = len(kernel)

    # Convolve each neuron independently
    # Using 'full' mode then truncating to preserve causal structure
    calcium_fine = np.zeros((T_fine, N), dtype=np.float64)
    for j in range(N):
        conv = np.convolve(spikes[:, j].astype(np.float64), kernel, mode='full')
        calcium_fine[:, j] = conv[:T_fine]

    # ── Step 2: Scale and add baseline ────────────────────────────────────
    calcium_fine = baseline + amplitude * calcium_fine

    # ── Step 3: Downsample to imaging frame rate ──────────────────────────
    bin_size = max(1, int(round(dt_imaging / dt_fine)))
    T_imaging = T_fine // bin_size

    if bin_size > 1:
        # Average within each bin (mimics calcium indicator integration)
        truncated = calcium_fine[:T_imaging * bin_size]
        calcium_ds = truncated.reshape(T_imaging, bin_size, N).mean(axis=1)
    else:
        calcium_ds = calcium_fine[:T_imaging]

    # ── Step 4: Add observation noise ─────────────────────────────────────
    if noise_std > 0:
        calcium_ds += rng.normal(0, noise_std, calcium_ds.shape)

    return calcium_ds


def estimate_noise_std(traces: np.ndarray) -> float:
    """
    Estimate observation noise from real calcium traces.

    Uses the median absolute deviation of the first-difference,
    which is robust to transients and baseline drift:
        σ ≈ MAD(diff(trace)) / 0.6745 / √2

    Parameters
    ----------
    traces : (T, N) real calcium traces

    Returns
    -------
    noise_std : estimated noise standard deviation
    """
    diffs = np.diff(traces, axis=0)
    mad = np.median(np.abs(diffs - np.median(diffs)))
    return float(mad / 0.6745 / np.sqrt(2))


def match_noise_to_data(
    real_traces: np.ndarray,
) -> dict:
    """
    Estimate forward model parameters from real data to improve
    simulation-data agreement.

    Returns a dict of kwargs suitable for calcium_forward_model().

    Parameters
    ----------
    real_traces : (T, N) real calcium dF/F traces

    Returns
    -------
    dict with keys: noise_std, amplitude, baseline
    """
    noise = estimate_noise_std(real_traces)
    baseline = float(np.median(real_traces))

    # Estimate amplitude from peak transients
    # Subtract baseline, take 95th percentile as typical peak
    detrended = real_traces - baseline
    peak_95 = float(np.percentile(detrended, 95))
    # A synchronised burst involves ~10-50 spikes per neuron per event
    # so amplitude_per_spike ≈ peak / typical_burst_spikes
    amplitude = max(peak_95 / 20.0, 0.05)

    return {
        "noise_std": noise,
        "amplitude": amplitude,
        "baseline": baseline,
    }



import matplotlib.pyplot as plt
from simulation import generate_neuron_positions, simulate_kndy_network
from config import SimulationConfig

cfg       = SimulationConfig()
positions = generate_neuron_positions(cfg)
result    = simulate_kndy_network(cfg, positions=positions)
spikes    = result["spikes"]
calcium   = calcium_forward_model(spikes, dt_fine=cfg.dt)

T, N = calcium.shape
dt_imaging = 0.1
t_s = np.arange(T) * dt_imaging

lo     = calcium.min(axis=0, keepdims=True)
hi     = calcium.max(axis=0, keepdims=True)
normed = (calcium - lo) / np.where(hi - lo > 0, hi - lo, 1.0)
offset = 1.3

fig, ax = plt.subplots(figsize=(14, max(4, N * 0.3)))
for i in range(N):
    ax.plot(t_s, normed[:, i] + i * offset,
            lw=0.5, alpha=0.8, color=plt.cm.tab20(i % 20))
ax.set_yticks(np.arange(N) * offset)
ax.set_yticklabels(np.arange(N), fontsize=6)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Neuron")
ax.set_title(f"Synthetic GCaMP6s dF/F  (N={N}, {t_s[-1]:.0f} s)")
fig.tight_layout()
plt.show()
