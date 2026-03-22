# Methods

Technical details on each pipeline component. For a high-level overview, see [README.md](README.md).

## 1. Network simulator

### Ising model with neuropeptide feedback

Each neuron is modelled as a binary (0/1) variable updated asynchronously via Glauber dynamics. The activation probability of neuron *i* at each micro-timestep depends on a local field:

```
h_i(t) = h_0 + J_c * Σ_j J_ij * s_j(t) + w_NKB * N(t) - w_Dyn * D(t)
P(s_i = 1) = σ(β * h_i)
```

where:
- `h_0 < 0` is a tonic inhibitory field ensuring sparse baseline activity
- `J_ij` is the coupling matrix encoding synaptic connectivity
- `N(t)`, `D(t)` are NKB and dynorphin concentrations (see below)
- `β` is the inverse temperature controlling sensitivity to the local field

### Neuropeptide ODEs

NKB and dynorphin evolve via coupled nonlinear ODEs (Euler method):

```
dN/dt = (k_N * M^n2) / (K_m2^n2 + M^n2) + k_N0 - d_N * N
dD/dt = (k_D * M^n1) / (K_m1^n1 + M^n1) + k_D0 - d_D * D
```

where `M(t) = (1/N_neurons) * Σ_i s_i(t)` is the population activity. NKB provides positive feedback (excitatory via NK3R), dynorphin provides delayed negative feedback (inhibitory via KOR). The interplay produces relaxation oscillations: NKB-driven synchronised bursts terminated by dynorphin accumulation, with inter-pulse intervals of ~5-15 minutes.

### Connectivity model

Edge probability between neurons *i*, *j* follows a Gaussian spatial kernel, with a cluster boost that allocates a controllable fraction of the total degree budget to intra-cluster edges:

```
p_base(i,j) = exp(-d_ij^2 / (2σ^2))
f_intra = f_natural(σ) + c * (1 - f_natural(σ))
```

where `f_natural` is the fraction of edges the distance kernel would naturally place within clusters, and `c` (cluster_strength) interpolates between purely distance-dependent (c=0) and fully intra-cluster (c=1) connectivity. The probability matrix is rescaled *after* the cluster boost to hit the target mean degree exactly.

Coupling weights are drawn from |N(1, 0.3)| and normalised by √(mean_degree).

## 2. Calcium forward model

Binary spike trains from the Ising simulator are converted to synthetic GCaMP6s fluorescence traces:

1. **Convolution** with a double-exponential GCaMP6s kernel:
   ```
   K(t) = A * (exp(-t/τ_decay) - exp(-t/τ_rise))
   τ_rise = 0.18s, τ_decay = 2.6s  (Chen et al. 2013)
   ```

2. **Downsampling** to the imaging frame rate (typically 10 Hz) by bin-averaging.

3. **Gaussian noise** addition, calibrated from real recordings using the MAD estimator on first-differences.

### Calibration from data

The `calibrate_calcium.py` script estimates forward model parameters from real recordings:
- **noise_std**: median absolute deviation of trace first-differences, scaled by 1/(0.6745 × √2)
- **amplitude**: estimated from the 95th percentile of detrended traces, divided by estimated spikes per burst
- **n_mad**: binarisation threshold for IPL, set to match visually confirmed transient detection

## 3. Summary statistics

### Sync-event statistics

Computed from the population-mean calcium trace (normalised to [0, 1]):

- **sync_frequency**: number of full synchronisation events per unit time. Full events are detected when the normalised population mean exceeds a threshold (0.50 for calcium, 0.30 for binary).
- **log_partial_ratio**: log(1 + n_partial / max(n_full, 1)). Captures partial recruitment behaviour. Log-transformed to compress the heavy tail.
- **mean_event_fraction**: mean normalised population activity during detected full sync events.

### IPL coupling statistics

The Ising pseudo-likelihood method fits per-neuron logistic regressions to binarised traces, recovering an effective coupling matrix J. Five statistics are extracted:

- **mean_coupling**: mean |J_ij| over off-diagonal pairs. Scales with β and network density.
- **coupling_cv**: coefficient of variation of |J_ij|. High when connectivity is heterogeneous (short σ or strong clustering).
- **degree_cv**: CV of the node degree distribution from thresholded |J|. Captures spatial inhomogeneity.
- **modularity_Q**: Newman modularity via spectral bisection of thresholded J. Directly measures modular structure.
- **degree_gini**: Gini coefficient of the degree distribution. High when connectivity is spatially concentrated (short σ).

### Binarisation for IPL

Continuous calcium traces are binarised per-neuron using an adaptive threshold:

```
threshold_j = median_j + n_mad × MAD_j
```

where MAD is the median absolute deviation. The `n_mad` parameter (typically 7.5 for high-SNR GCaMP6s data) must be calibrated to distinguish genuine transients from noise. The same value is used for both simulated and real data.

### Calibration methodology

Summary statistics were selected through iterative calibration:

1. **Latin hypercube sampling** (500 points) from the prior
2. **Simulate + compute stats** at each point
3. **Spearman rank correlation** between each stat and each parameter
4. **Drop** statistics with: zero variance, profile redundancy (|ρ| > 0.85 with another stat), or NaN values
5. **Add** targeted statistics for poorly-identified parameters
6. Repeat until every parameter has |ρ| ≥ 0.25 with at least one statistic

## 4. SNPE inference

### Masked autoregressive flows

The posterior p(θ|x) is represented by a normalising flow that transforms a standard Gaussian into the target distribution through a sequence of invertible autoregressive layers. Each layer applies:

```
θ_i = μ_i(θ_{<i}, x) + σ_i(θ_{<i}, x) × z_i
```

where μ and σ are neural network outputs conditioned on the summary vector x. The autoregressive structure ensures a triangular Jacobian, making density evaluation O(N) instead of O(N³).

The default sbi configuration uses 5 MAF layers with 2 hidden layers of 50 units each — sufficient for the 4-parameter problem.

### Single-round amortised training

All training simulations are drawn from the prior (no observed data involved). This produces a model valid for any observation within the prior support. The tradeoff vs. multi-round training:

| | Single-round | Multi-round |
|---|---|---|
| Training data | From prior | From narrowing proposals |
| Validity | Any x in prior support | Specific x_obs only |
| Efficiency | ~3× more sims needed | Focused, fewer sims |
| Reusability | Condition on new data instantly | Must retrain per dataset |

### Manual standardisation

Summary statistics are standardised (mean=0, std=1) before passing to sbi, with a floor on std to prevent division by zero for statistics that are constant in some parameter regimes (e.g. sync_frequency = 0 when β is below the firing threshold).

## 5. Posterior predictive checks

PPC validates the posterior by simulating from posterior samples and comparing the resulting summary statistics to the observation:

1. Draw n parameter vectors from p(θ|x_obs)
2. Simulate each through the full pipeline (Ising → calcium → stats)
3. Compare simulated stats to observed stats via z-scores

A z-score |z| < 2 for all statistics indicates the posterior is well-calibrated.

## References

- Chen, T.-W., et al. (2013). Ultrasensitive fluorescent proteins for imaging neuronal activity. *Nature*, 499(7458), 295-300.
- Greenberg, D., Nonnenmacher, M., & Macke, J. (2019). Automatic posterior transformation for likelihood-free inference. *ICML*.
- Lehman, M. N., Coolen, L. M., & Goodman, R. L. (2010). KNDy cells of the arcuate nucleus: a central node in the control of GnRH secretion. *Endocrinology*, 151(8), 3479-3489.
- Papamakarios, G., Pavlakou, T., & Murray, I. (2017). Masked autoregressive flow for density estimation. *NeurIPS*.
- Tejero-Cantero, A., et al. (2020). sbi: A toolkit for simulation-based inference. *JOSS*, 5(52), 2505.
