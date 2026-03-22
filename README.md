# KNDy Network Inference Pipeline

Simulation-based inference of structural connectivity parameters in hypothalamic KNDy (Kisspeptin/NKB/Dynorphin) neuronal networks using amortised neural posterior estimation.

## Overview

KNDy neurons in the arcuate nucleus generate pulsatile GnRH release through synchronised network activity. This pipeline infers the structural connectivity parameters that give rise to observed calcium imaging dynamics, using a combination of:

- An **Ising-type network simulator** with adaptive neuropeptide feedback (C++)
- A **GCaMP6s calcium forward model** bridging simulated spikes to fluorescence traces
- **Ising pseudo-likelihood (IPL)** to extract effective coupling matrices from calcium data
- **Sequential Neural Posterior Estimation (SNPE)** via normalising flows for amortised Bayesian inference

The pipeline is **amortised**: the neural density estimator is trained once from prior simulations (no observed data needed), then conditions on any new calcium recording in milliseconds to produce a full posterior distribution over connectivity parameters.

### Inferred parameters

| Parameter | Symbol | Description | Prior |
|---|---|---|---|
| Spatial length-scale | σ | Gaussian decay of connection probability with distance (µm) | [30, 400] |
| Mean degree | k̄ | Expected synaptic partners per neuron | [8, 25] |
| Cluster strength | c | Fraction of degree budget allocated to intra-cluster edges | [0, 0.9] |
| Inverse temperature | β | Ising coupling strength (controls excitability) | [0.4, 3.0] |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING (once)                          │
│                                                                 │
│  Prior p(θ) ──→ Ising simulator ──→ GCaMP6s forward ──→ x(θ)  │
│       │              (C++)            model                     │
│       │                                 │                       │
│       └──────── (θ, x) pairs ───────────┘                      │
│                       │                                         │
│               Neural density estimator                          │
│              (masked autoregressive flow)                        │
│                       │                                         │
│                  Save model                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE (per recording)                     │
│                                                                 │
│  Calcium recording ──→ Summary statistics ──→ Trained model     │
│       (CSV)              (8-D vector)             │             │
│                                              p(θ | x_obs)      │
│                                           (milliseconds)        │
└─────────────────────────────────────────────────────────────────┘
```

## Summary statistics

The 8-dimensional summary vector combines synchronisation-event features (computed from the population calcium trace) with structural coupling features (computed via IPL on binarised traces):

| Index | Statistic | Type | Primary target |
|---|---|---|---|
| 0 | `sync_frequency` | Sync | β, mean_degree |
| 1 | `log_partial_ratio` | Sync | cluster_strength |
| 2 | `mean_event_fraction` | Sync | mean_degree |
| 3 | `mean_coupling` | IPL | β |
| 4 | `coupling_cv` | IPL | mean_degree, σ |
| 5 | `degree_cv` | IPL | β, mean_degree |
| 6 | `modularity_Q` | IPL | cluster_strength, β |
| 7 | `degree_gini` | IPL | σ |

Summary statistics were selected through iterative calibration (Latin hypercube sampling + Spearman rank correlation analysis) to maximise parameter identifiability while minimising redundancy.

## Installation

### Prerequisites

- Python ≥ 3.9
- C++ compiler (g++ on Linux, MSVC on Windows)
- [pybind11](https://pybind11.readthedocs.io/)

### Setup

```bash
git clone https://github.com/yourusername/kndy-inference.git
cd kndy-inference

python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# CPU-only PyTorch (recommended unless you have a GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sbi numpy scipy scikit-learn matplotlib

# Build the C++ simulator extension
bash build_mycpp.sh
```

### Verify installation

```bash
python -c "import mycpp, torch, sbi; print('All imports OK')"
```

## Quick start

### 1. Calibrate the calcium forward model

Estimate noise and amplitude parameters from a real recording:

```bash
python calibrate_calcium.py \
    --input data/recording.csv \
    --dt-imaging 0.1
```

Edit the output `calcium_params.json` to set `amplitude` and `n_mad` based on the preview figure.

### 2. Run calibration check

Verify that summary statistics discriminate between parameter values before committing to training:

```bash
python calibration_check.py \
    --n-samples 500 \
    --n-workers 16 \
    --calcium-params calcium_params.json
```

Inspect `calibration/calibration_correlation.png` — every parameter should have at least one statistic with |ρ| > 0.25.

### 3. Train the model

Train the amortised posterior from the prior (no data needed):

```bash
python train_model.py \
    --num-simulations 1500 \
    --n-workers 16 \
    --calcium-params calcium_params.json \
    --output models/
```

### 4. Run inference on data

Condition the trained model on a calcium recording and sample the posterior:

```bash
python infer.py \
    --model-dir models/ \
    --traces data/recording.csv \
    --output results/ \
    --n-samples 1000 \
    --n-ppc 30
```

Conditioning + sampling takes milliseconds. Only PPC simulations add compute time.

## Project structure

```
kndy-inference/
├── config.py                  # SimulationConfig, InferenceConfig dataclasses
├── simulation.py              # Ising network simulator (calls C++ via pybind11)
├── simulation.cpp             # C++ Ising model with neuropeptide ODEs
├── calcium_model.py           # GCaMP6s forward model (convolve, downsample, noise)
├── statistics.py              # Summary statistics (sync events + IPL coupling)
├── snpe_model.py              # SNPEInference class (train, save, load, condition)
│
├── train_model.py             # CLI: train amortised model from prior
├── infer.py                   # CLI: load model, condition on data, posterior + PPC
├── calibrate_calcium.py       # CLI: estimate forward model params from real data
├── calibration_check.py       # CLI: verify stat identifiability across prior
├── estimate_clusters.py       # CLI: estimate number of functional clusters in data
├── convert_csv.py             # CLI: convert continuous CSV traces to binary .npy
├── build_mycpp.sh             # Build script for C++ pybind11 extension
│
├── calcium_params.json        # Calibrated forward model parameters
│
├── jobscript.sh               # SGE job script: training
├── jobscript_calibration.sh   # SGE job script: calibration check
├── jobscript_infer.sh         # SGE job script: inference on real data
│
├── models/                    # Trained SNPE model (after training)
│   ├── snpe_posterior.pt
│   ├── snpe_standardize.npz
│   └── snpe_calcium_params.json
│
├── results/                   # Inference outputs (after running infer.py)
│   ├── posterior_samples.npy
│   ├── x_observed.npy
│   ├── inference_summary.png
│   └── ppc_checks.png
│
└── calibration/               # Calibration outputs
    ├── calibration_correlation.png
    ├── calibration_scatter.png
    ├── calibration_histograms.png
    └── calibration_data.npz
```

## The simulator

The KNDy network is modelled as a stochastic Ising system with adaptive neuropeptide feedback:

**Neuron dynamics.** Each neuron *i* is a Bernoulli variable with activation probability:

$$P(s_i = 1) = \sigma\left(\beta \left[h_0 + J_c \sum_j J_{ij} s_j + w_{\text{NKB}} \cdot N(t) - w_{\text{Dyn}} \cdot D(t)\right]\right)$$

where *J* is the coupling matrix, *N(t)* and *D(t)* are NKB and dynorphin concentrations, and σ is the logistic function.

**Neuropeptide dynamics.** NKB (excitatory) and dynorphin (inhibitory) concentrations evolve via coupled nonlinear ODEs integrated with the Euler method. NKB secretion is driven by population activity, providing positive feedback that triggers synchronised bursts. Dynorphin provides delayed negative feedback that terminates each burst and enforces the inter-pulse refractory period.

**Connectivity.** Edge probability between neurons *i* and *j* follows a Gaussian spatial kernel:

$$p_{\text{base}}(i,j) = \exp\left(-\frac{d_{ij}^2}{2\sigma^2}\right)$$

modulated by a cluster boost that allocates a fraction of the total degree budget to intra-cluster connections, controlled by the `cluster_strength` parameter.

## Calcium forward model

Binary Ising spike trains are converted to synthetic GCaMP6s calcium traces to match the representation of real imaging data:

1. **Convolve** each neuron's spike train with a GCaMP6s double-exponential kernel (τ_rise = 0.18s, τ_decay = 2.6s)
2. **Downsample** to the imaging frame rate (e.g. 10 Hz)
3. **Add Gaussian observation noise** calibrated from real recordings

This ensures summary statistics are computed identically on simulated and real data.

## Inference method

The pipeline uses **Sequential Neural Posterior Estimation (SNPE-C)** from the [sbi](https://github.com/sbi-dev/sbi) library. A masked autoregressive flow learns the posterior distribution p(θ|x) by training on (θ, x) pairs simulated from the prior.

**Amortised inference.** Single-round training from the prior produces a model valid for any observation. Once trained, conditioning on a new recording requires only a forward pass through the neural network — no additional simulations.

**Calibration-driven design.** Summary statistics were iteratively refined through calibration analysis:
- Removed statistics with zero variance (constant across the prior)
- Removed redundant statistics (profile ρ > 0.85 between pairs)
- Added scale-invariant coupling statistics to break beta-dominance
- Tuned binarisation threshold (n_MAD) to match real data SNR

## HPC deployment

The training step is parallelised across CPU cores using Python multiprocessing. SGE job scripts are provided for the University of Edinburgh's Eddie cluster, but adapt to any HPC system with a job scheduler.

Typical compute times on 16 cores (N=52 neurons, 100k timesteps):
- Calibration (500 samples): ~1 hour
- Training (1500 simulations): ~2 hours
- Inference (conditioning + 1000 samples): < 1 second
- PPC (30 simulations): ~30 minutes

## Data format

Input calcium recordings should be CSV files with rows as time steps and columns as neurons. The first column may be a time index (auto-detected and dropped). Values should be dF/F (baseline-normalised fluorescence).

The pipeline was validated on GCaMP6s calcium imaging data from KNDy neurons in female mice (10 Hz frame rate, ~60 min recordings, 8–52 neurons).

## Citation

If you use this pipeline in your work, please cite:

```
Galanos, Y. (2026). Amortised neural inference of structural connectivity
in KNDy neuronal networks. PhD Thesis, University of Edinburgh.
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

Supervised by Prof. Margaritis Voliotis (University of Exeter). Calcium imaging data from Moore et al., provided via the [Open Science Framework](https://osf.io/).

## Software credits

This pipeline relies on the following open-source libraries:

- **[sbi](https://github.com/sbi-dev/sbi)** (Tejero-Cantero et al., 2020) — simulation-based inference toolkit implementing SNPE-C. *JOSS*, 5(52), 2505.
- **[PyTorch](https://pytorch.org/)** (Paszke et al., 2019) — neural network backend for the normalising flow. *NeurIPS*.
- **[pybind11](https://github.com/pybind/pybind11)** (Jakob et al., 2017) — C++/Python bindings for the Ising simulator.
- **[scikit-learn](https://scikit-learn.org/)** (Pedregosa et al., 2011) — logistic regression for Ising pseudo-likelihood. *JMLR*, 12, 2825-2830.
- **[NumPy](https://numpy.org/)** (Harris et al., 2020) — array computation. *Nature*, 585, 357-362.
- **[SciPy](https://scipy.org/)** (Virtanen et al., 2020) — statistical tests and spatial distances. *Nature Methods*, 17, 261-272.
- **[Matplotlib](https://matplotlib.org/)** (Hunter, 2007) — visualisation. *Computing in Science & Engineering*, 9(3), 90-95.
