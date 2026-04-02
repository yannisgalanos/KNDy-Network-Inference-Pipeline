"""
statistics.py
-------------
Summary statistics for the KNDy inference pipeline.

Accepts both binary spike arrays and continuous calcium (dF/F) traces.
When the calcium forward model is used, both simulated and real data
arrive as continuous traces, so all functions handle both representations.

The 8-dimensional summary vector:

  Sync-event statistics (from population trace):
  Index | Name                   | Primary target
  ------+------------------------+---------------------
    0   | sync_frequency         | beta
    1   | log_partial_ratio      | cluster_strength
    2   | mean_event_fraction    | mean_degree

  IPL coupling statistics (from binarised trace → J matrix):
  Index | Name                   | Primary target
  ------+------------------------+---------------------
    3   | mean_coupling          | beta
    4   | coupling_cv            | mean_degree
    5   | degree_cv              | beta / structure
    6   | modularity_Q           | cluster_strength
    7   | degree_gini            | sigma_spatial
"""

from __future__ import annotations

import warnings
import numpy as np
from numpy.linalg import eigh
from typing import Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Input handling: detect binary vs continuous, binarise if needed
# ─────────────────────────────────────────────────────────────────────────────

def _is_binary(traces: np.ndarray) -> bool:
    """Check if array contains only 0/1 values."""
    unique = np.unique(traces)
    return len(unique) <= 2 and set(unique).issubset({0, 0.0, 1, 1.0})


def _binarise_traces(
    traces: np.ndarray,
    n_mad: float = 3.0,
) -> np.ndarray:
    """
    Binarise continuous calcium traces per neuron.

    Threshold = median + n_mad × MAD for each neuron.
    MAD (median absolute deviation) is robust to the baseline-dominated
    distribution of calcium signals.

    Parameters
    ----------
    traces : (T, N) continuous array (dF/F)
    n_mad  : number of MADs above median

    Returns
    -------
    binary : (T, N) int8 array
    """
    T, N = traces.shape
    binary = np.zeros((T, N), dtype=np.int8)

    for j in range(N):
        col = traces[:, j]
        med = np.median(col)
        mad = np.median(np.abs(col - med))
        if mad < 1e-12:
            mad = col.std()
        thresh = med + n_mad * mad
        binary[:, j] = (col > thresh).astype(np.int8)

    return binary


# ─────────────────────────────────────────────────────────────────────────────
# Ising Pseudo-Likelihood (IPL)
# ─────────────────────────────────────────────────────────────────────────────

def _fit_ipl(traces: np.ndarray, C: float = 10.0, n_mad: float = 3.0) -> np.ndarray:
    """
    Fit Ising couplings J via pseudo-likelihood (logistic regression).

    If input is continuous, binarises first. IPL requires binary targets
    for logistic regression.

    Parameters
    ----------
    traces : (T, N) binary or continuous array
    C      : inverse L2 regularisation strength
    n_mad  : MAD multiplier for binarisation (only used if continuous)

    Returns
    -------
    J : (N, N) symmetric coupling matrix, zero diagonal.
    """
    from sklearn.linear_model import LogisticRegression

    if _is_binary(traces):
        spikes = traces.astype(float)
    else:
        spikes = _binarise_traces(traces, n_mad=n_mad).astype(float)

    T, N = spikes.shape
    J = np.zeros((N, N))

    for i in range(N):
        y = spikes[:, i]
        if y.std() < 1e-8:
            continue

        mask = np.ones(N, dtype=bool)
        mask[i] = False
        X = spikes[:, mask]

        clf = LogisticRegression(
            C=C, fit_intercept=True,
            solver="lbfgs", max_iter=600, tol=1e-4,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X, y)

        J[i, mask] = clf.coef_[0]

    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    return J


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: Gini coefficient, spectral modularity
# ─────────────────────────────────────────────────────────────────────────────

def _gini(values: np.ndarray) -> float:
    """Gini coefficient. 0 = equal, 1 = maximally unequal."""
    v = np.sort(values.ravel()).astype(float)
    n = len(v)
    if n < 2 or v.sum() < 1e-12:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * v) - (n + 1) * v.sum()) /
                 (n * v.sum()))


def _spectral_modularity(A: np.ndarray) -> float:
    """Newman modularity Q via leading-eigenvector bisection."""
    N = A.shape[0]
    degrees = A.sum(axis=1)
    m2 = degrees.sum()
    if m2 < 1e-12:
        return 0.0
    B = A - np.outer(degrees, degrees) / m2
    eigenvalues, eigenvectors = eigh(B)
    top_vec = eigenvectors[:, -1]
    s = np.where(top_vec >= 0, 1.0, -1.0)
    Q = float(s @ B @ s) / m2
    return max(Q, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# IPL coupling statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_coupling_statistics(
    traces: np.ndarray,
    ipl_C: float = 10.0,
    degree_percentile: float = 70.0,
    n_mad: float = 3.0,
) -> Dict[str, float]:
    """
    Fit IPL and extract 5 structural summary statistics from J.

    Accepts binary or continuous traces (binarises internally for IPL).
    """
    J = _fit_ipl(traces, C=ipl_C, n_mad=n_mad)
    N = J.shape[0]

    idx = np.triu_indices(N, k=1)
    abs_j = np.abs(J[idx])

    mean_coupling = float(abs_j.mean()) if len(abs_j) > 0 else 0.0
    coupling_std  = float(abs_j.std())  if len(abs_j) > 0 else 0.0
    coupling_cv   = coupling_std / (mean_coupling + 1e-12)

    abs_J_full = np.abs(J.copy())
    np.fill_diagonal(abs_J_full, 0.0)
    thresh = float(np.percentile(abs_j, degree_percentile)) if len(abs_j) > 0 else 0.0

    A_thresh = (abs_J_full >= thresh).astype(float)
    np.fill_diagonal(A_thresh, 0.0)
    degrees = A_thresh.sum(axis=1)

    inferred_mean_degree = float(degrees.mean())
    degree_cv = float(degrees.std()) / (inferred_mean_degree + 1e-12)

    degree_gini = _gini(degrees)
    modularity_Q = _spectral_modularity(A_thresh)

    return {
        "mean_coupling":  mean_coupling,
        "coupling_cv":    coupling_cv,
        "degree_cv":      degree_cv,
        "modularity_Q":   modularity_Q,
        "degree_gini":    degree_gini,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Synchronisation event detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_sync_events(
    traces: np.ndarray,
    threshold_full: float    = None,
    partial_low:    float    = None,
    partial_high:   float    = None,
    min_gap_steps:  int      = 5,
    n_mad_binarise: float    = 3.0,
) -> Dict[str, object]:
    """
    Detect full and partial synchronisation events.

    Works on both binary and continuous traces:
    - Binary: thresholds population firing fraction (as before)
    - Continuous: normalises population mean to [0, 1] range and applies
      adaptive thresholds based on the signal distribution

    Parameters
    ----------
    traces          : (T, N) binary or continuous array
    threshold_full  : fraction/level for full sync (None = auto)
    partial_low     : lower bound for partial sync (None = auto)
    partial_high    : upper bound for partial sync (None = auto)
    min_gap_steps   : minimum gap between detected events
    n_mad_binarise  : MAD multiplier if binarising for event detection
    """
    T, N = traces.shape
    binary = _is_binary(traces)

    if binary:
        pop = traces.mean(axis=1)
        # Default thresholds for binary data
        if threshold_full is None:
            threshold_full = 0.30
        if partial_low is None:
            partial_low = 0.12
        if partial_high is None:
            partial_high = 0.28
    else:
        # For continuous calcium traces: compute population mean,
        # then normalise to [0, 1] using robust min/max
        pop_raw = traces.mean(axis=1)
        p_low  = np.percentile(pop_raw, 2)
        p_high = np.percentile(pop_raw, 99.5)
        if p_high - p_low < 1e-12:
            pop = np.zeros(T)
        else:
            pop = (pop_raw - p_low) / (p_high - p_low)
            pop = np.clip(pop, 0.0, 1.0)

        # Adaptive thresholds for normalised calcium signal
        if threshold_full is None:
            threshold_full = 0.50
        if partial_low is None:
            partial_low = 0.20
        if partial_high is None:
            partial_high = 0.45

    full_mask    = pop >= threshold_full
    partial_mask = (pop >= partial_low) & (pop < partial_high)

    full_events    = _extract_onsets(full_mask,    min_gap_steps)
    partial_events = _extract_onsets(partial_mask, min_gap_steps)

    if len(full_events) > 1:
        iei    = np.diff(full_events).astype(float)
        iei_cv = float(iei.std() / (iei.mean() + 1e-12))
    else:
        iei    = np.array([np.nan])
        iei_cv = 0.0

    mean_frac = float(pop[full_events].mean()) if len(full_events) > 0 else 0.0

    return {
        "full_event_times":    full_events,
        "partial_event_times": partial_events,
        "iei":                 iei,
        "n_full_events":       len(full_events),
        "n_partial_events":    len(partial_events),
        "sync_frequency":      len(full_events) / T,
        "iei_cv":              iei_cv,
        "mean_event_fraction": mean_frac,
    }


def _extract_onsets(mask: np.ndarray, min_gap: int) -> np.ndarray:
    """Return onset indices where mask rises True, enforcing a minimum gap."""
    onsets     = []
    in_event   = False
    last_onset = -min_gap - 1
    for t, val in enumerate(mask):
        if val and not in_event:
            if t - last_onset >= min_gap:
                onsets.append(t)
                last_onset = t
            in_event = True
        elif not val:
            in_event = False
    return np.array(onsets, dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic functions (retained, not used in inference)
# ─────────────────────────────────────────────────────────────────────────────

def compute_fc_matrix(traces: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix with zero diagonal."""
    if traces.ndim < 2:
        raise RuntimeError(f"traces is not a 2D array, shape: {traces.shape}")
    fc = np.corrcoef(traces.T)
    np.fill_diagonal(fc, 0.0)
    return np.nan_to_num(fc, nan=0.0)


def spectral_statistics(fc: np.ndarray) -> Dict[str, float]:
    """Eigenspectrum analysis of the FC matrix."""
    N = fc.shape[0]
    eigenvalues, eigenvectors = eigh(fc)
    eigenvalues_desc = eigenvalues[::-1].copy()
    top_vec = eigenvectors[:, -1]
    T_eff    = 4 * N
    mp_upper = (1.0 + np.sqrt(N / T_eff)) ** 2
    n_sig    = int(np.sum(eigenvalues_desc > mp_upper))
    pos_eigs = eigenvalues_desc[eigenvalues_desc > 0.0]
    if len(pos_eigs) > 0:
        p = pos_eigs / pos_eigs.sum()
        spectral_entropy = float(-np.sum(p * np.log(p + 1e-12)))
    else:
        spectral_entropy = 0.0
    pr = 1.0 / (N * float(np.sum(top_vec ** 4)) + 1e-12)
    pr = float(np.clip(pr, 0.0, 1.0))
    return {
        "n_significant_eigs": max(n_sig, 0),
        "top_eigenvalue":     float(eigenvalues_desc[0]),
        "spectral_entropy":   spectral_entropy,
        "participation_ratio": pr,
    }


def compute_avalanches(traces: np.ndarray, bin_size: int = 1) -> Dict[str, object]:
    """Detect neuronal avalanches. Binarises continuous input first."""
    if not _is_binary(traces):
        traces = _binarise_traces(traces)
    T, N = traces.shape
    if bin_size > 1:
        n_bins = T // bin_size
        binned = traces[:n_bins*bin_size].reshape(n_bins, bin_size, N).sum(axis=1)
    else:
        binned = traces
    pop = binned.sum(axis=1)
    sizes, durations = [], []
    in_av, cur_size, cur_dur = False, 0, 0
    for v in pop:
        if v > 0:
            in_av = True; cur_size += int(v); cur_dur += 1
        else:
            if in_av:
                sizes.append(cur_size); durations.append(cur_dur)
                cur_size = 0; cur_dur = 0
            in_av = False
    sizes     = np.array(sizes, dtype=float)
    durations = np.array(durations, dtype=float)
    tau   = _powerlaw_mle(sizes) if len(sizes) > 5 else np.nan
    alpha = _powerlaw_mle(durations) if len(durations) > 5 else np.nan
    return {
        "sizes": sizes, "durations": durations,
        "size_exponent": float(tau), "dur_exponent": float(alpha),
        "mean_size": float(sizes.mean()) if len(sizes) > 0 else 0.0,
        "mean_duration": float(durations.mean()) if len(durations) > 0 else 0.0,
    }


def _powerlaw_mle(data: np.ndarray, x_min: float = 1.0) -> float:
    x = data[data >= x_min]
    if len(x) < 2:
        return np.nan
    return float(1.0 + len(x) / np.sum(np.log(x / (x_min - 0.5))))


# ─────────────────────────────────────────────────────────────────────────────
# Calcium correlation statistics (bypass IPL — work on continuous traces)
# ─────────────────────────────────────────────────────────────────────────────

def _residual_correlation_stats(traces: np.ndarray) -> Dict[str, float]:
    """
    Correlation statistics after regressing out the global (population mean)
    signal from each neuron's calcium trace.

    Removing the global component strips out full-sync events that dominate
    raw Pearson correlations, exposing local co-fluctuation structure that
    reflects spatial connectivity (sigma).

    Returns
    -------
    residual_mean_corr : mean pairwise correlation of residuals
    residual_corr_cv   : CV of pairwise residual correlations
                         (high = block structure = short sigma,
                          low = uniform = long sigma)
    """
    T, N = traces.shape
    if N < 3 or T < 10:
        return {"residual_mean_corr": 0.0, "residual_corr_cv": 0.0}

    pop_mean = traces.mean(axis=1)
    pop_dot = np.dot(pop_mean, pop_mean)

    # Regress out global signal from each neuron
    residuals = np.zeros_like(traces)
    for j in range(N):
        if pop_dot > 1e-12:
            slope = np.dot(traces[:, j], pop_mean) / pop_dot
            residuals[:, j] = traces[:, j] - slope * pop_mean
        else:
            residuals[:, j] = traces[:, j]

    fc = np.corrcoef(residuals.T)
    np.fill_diagonal(fc, np.nan)
    idx = np.triu_indices(N, k=1)
    pairwise = fc[idx]
    pairwise = pairwise[np.isfinite(pairwise)]

    if len(pairwise) < 2:
        return {"residual_mean_corr": 0.0, "residual_corr_cv": 0.0}

    mean_corr = float(np.mean(pairwise))
    std_corr = float(np.std(pairwise))
    cv_corr = std_corr / (abs(mean_corr) + 1e-12)

    return {
        "residual_mean_corr": mean_corr,
        "residual_corr_cv": cv_corr,
    }


def _fc_modularity(traces: np.ndarray) -> float:
    """
    Newman modularity Q computed directly on the calcium correlation matrix.

    Bypasses IPL and binarisation entirely — thresholds positive correlations
    and runs spectral bisection on the resulting graph.

    This preserves spatial structure that the IPL pipeline destroys.
    """
    T, N = traces.shape
    if N < 4 or T < 10:
        return 0.0

    fc = np.corrcoef(traces.T)
    np.fill_diagonal(fc, 0.0)
    fc = np.nan_to_num(fc, nan=0.0)

    # Keep only positive correlations as edges
    A = np.maximum(fc, 0.0)
    return _spectral_modularity(A)


# ─────────────────────────────────────────────────────────────────────────────
# Composite summary statistic vector  (INFERENCE TARGET)
# ─────────────────────────────────────────────────────────────────────────────

STAT_NAMES = [
    # Sync-event statistics  (2)
    "sync_frequency",
    "log_partial_ratio",
    # Calcium correlation statistics  (2) — bypass IPL, target sigma
    "log_residual_corr_cv",
    "fc_modularity_Q",
    # IPL coupling statistics  (4)
    "mean_coupling",
    "coupling_cv",
    "degree_cv",
    "degree_gini",
]


def compute_summary_statistics(
    traces: np.ndarray,
    dt: float = 1.0,
    ipl_C: float = 10.0,
    n_mad: float = 3.0,
) -> np.ndarray:
    """
    Compute the 8-D composite summary statistic vector.

    Combines:
      - 2 sync-event features (from population trace)
      - 2 calcium correlation features (from continuous traces, bypass IPL)
      - 4 IPL coupling features (from binarised traces)

    Parameters
    ----------
    traces : (T, N) binary or continuous array
    dt     : time step in seconds (for converting sync_frequency to Hz)
    ipl_C  : IPL regularisation strength
    n_mad  : MAD multiplier for binarisation of continuous traces

    Returns
    -------
    stats : (8,) float32 array — NaN/Inf replaced with 0.
    """
    sync     = detect_sync_events(traces, n_mad_binarise=n_mad)
    coupling = compute_coupling_statistics(traces, ipl_C=ipl_C, n_mad=n_mad)
    resid    = _residual_correlation_stats(traces)
    fc_mod_Q = _fc_modularity(traces)

    n_full    = sync["n_full_events"]
    n_partial = sync["n_partial_events"]

    raw_ratio = n_partial / max(n_full, 1)
    log_partial_ratio = np.log1p(raw_ratio)

    stats = np.array([
        # Sync-event features (2)
        sync["sync_frequency"] / max(dt, 1e-12),
        log_partial_ratio,
        # Calcium correlation features (2)
        np.log1p(resid["residual_corr_cv"]),
        fc_mod_Q,
        # IPL coupling features (4)
        coupling["mean_coupling"],
        coupling["coupling_cv"],
        coupling["degree_cv"],
        coupling["degree_gini"],
    ], dtype=np.float32)

    return np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
