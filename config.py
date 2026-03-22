"""
config.py
---------
Configuration dataclasses for the KNDy network inference pipeline.

Parameters marked [INFER] are the inferential targets.
Parameters marked [FIXED] are assumed known from biology / held constant.
"""

from dataclasses import dataclass, field
from typing import Optional, List


# ─────────────────────────────────────────────────────────────────────────────
# Simulation configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationConfig:
    """
    Controls the KNDy network simulator.

    Biological notes
    ----------------
    KNDy (Kisspeptin/NKB/Dynorphin) neurons reside in the arcuate nucleus
    (ARC).  They form a 'pulse generator' for GnRH via autocrine NKB
    excitation and dynorphin inhibition.  Synchronised bursts drive
    downstream GnRH / LH pulses with an inter-pulse interval of ~30–90 min
    in female mice.
    """

  # Constants

    





    # Tunable parameters
    beta_val: float = 1.4      # critical range 0.75-0.9
    beta: float = beta_val     # global coupling strength [INFER]
    c: float = 0.75            # feedback strength
    g_N: float = 1.5           # NKB field influence   1.5
    g_D: float = 0.25          # Dyn field influence   0.25
    J_c: float = 1.0           # coupling strength

    # Biological parameters
    
    k_D0: float = 0.175        # basal Dyn secretion
    k_D: float = 9.0           # secretion coeff Dyn
    k_N: float = 320.0         # secretion coeff NKB
    k_N0: float = 0.0          # basal NKB
    K_m1: float = 1200.0       # saturation constant Dyn
    K_m2: float = 1200.0       # saturation constant NKB
    K_D: float = 0.3           # inhibition constant Dyn
    K_N: float = 32.0          # inhibition constant NKB
    n_1: float = 2.0           # Hill coeff Dyn
    n_2: float = 2.0           # Hill coeff NKB
    n_3: float = 2.0           # Hill coeff NKB inhibition
    n_4: float = 2.0           # Hill coeff NKB feedback
    d_D: float = 1.0           # degradation rate Dyn
    d_N: float = 10.0          # degradation rate NKB
    I_0: float = 0.2           # basal neuron activity

    h0: float = -1.8
    """Baseline (tonic) external field.  Strongly negative → neurons are
    silent in the absence of excitatory input, ensuring sparse baseline
    activity between synchronised pulses."""

    w_NKB: float = g_N
    """Weight of NKB-mediated excitatory feedback onto the Ising field.
    NKB acts via NK3R autoreceptors to drive synchronised bursting."""

    w_Dyn: float = g_D
    """Weight of dynorphin-mediated inhibitory feedback.  Dynorphin
    provides slow post-burst inhibition that terminates each pulse and
    enforces the inter-pulse refractory period."""
    

    # ── Network geometry ──────────────────────────────────────────────────
    

    arena_size: float = 360.0
    """Spatial extent of the modelled ARC region (µm)."""

    n_clusters: int = 4
    """Number of putative strongly-coupled sub-clusters."""

    seed: Optional[int] = 44
    """Master random seed (None → random)."""

    min_dist: float = 15.0
    max_attempts: int = 80
    cluster_size: int = 13 # Use // for integer division
    cluster_radius: float = 140.0
    min_cluster_center_distance: float = 200.0

    n_neurons: int = n_clusters * cluster_size
    """Number of modelled KNDy neurons in the ARC slice."""

    

    # ── Connectivity [INFER] ──────────────────────────────────────────────
    sigma_spatial: float = 180.0       #300 is a good value
    """Spatial length-scale of the Gaussian distance-dependent edge
    probability (µm).  Larger → longer-range connectivity."""

    mean_degree: float = 12.0      # 25 is a good value
    """Expected number of synaptic partners per neuron.  Controls
    network density independently of σ."""

    cluster_strength: float = 0.4          #0.7 is a good value
    """Extra within-cluster coupling boost in [0, 1].  0 → purely
    distance-dependent graph;  1 → fully connected clusters."""

    # Integration parameters
    dt: float = 1.0 / n_neurons        # individual spin flip timestep
    DT: float = 0.02           # physical time unit
    stoch_time_step: float = (1.0 / n_neurons) * 0.02 / 60.0
    norm: float = (1.0 / 2 / 0.02) * 60.0
    norm_2: float = 1.0 / n_neurons
    norm_time: float = 60.0 / 0.02

    # Main loop parameters
    total: int = 150000         # shorter run for inference
    wait: int = 8000
    t_end: int = total + wait # total + wait
    k_test: int = n_neurons // n_clusters
    test_radius: float = 200.0

    T: float = 1800.0
    """Recording duration after burn-in (s).  1800 s ≈ 30 min."""

    burnin: float = 600.0
    """Burn-in period discarded before recording (s)."""


   

# ─────────────────────────────────────────────────────────────────────────────
# Inference configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InferenceConfig:
    """
    Hyper-parameters for each inference stage.
    """

    # ── Stage 2: Ising pseudo-likelihood ─────────────────────────────────
    ipl_C: float = 10.0
    """Inverse L2 regularisation for logistic regression (sklearn C).
    Larger C → weaker regularisation."""

    ipl_symmetrise: bool = True
    """Symmetrise J = (J + Jᵀ) / 2 after fitting."""

    # ── Stage 3: Graphical LASSO ──────────────────────────────────────────
    glasso_alpha: Optional[float] = None
    """Sparsity penalty.  None → cross-validated automatically."""

    glasso_cv: int = 5
    """Number of CV folds when alpha is cross-validated."""

    # ── Stage 4: Distance-graph fitting ──────────────────────────────────
    graph_fit_method: str = "curve_fit"
    """Method for fitting exp(-d²/2σ²) to inferred couplings.
    One of: 'curve_fit', 'mle'."""

    # ── Stage 5: SMC-ABC posterior inference ─────────────────────────────
    abc_population_size: int = 50
    """Number of live particles per SMC round."""

    abc_n_rounds: int = 2
    """Number of sequential SMC rounds (adaptive ε schedule)."""

    abc_epsilon_percentile: float = 40.0
    """At each round, ε is set to this percentile of accepted distances,
    progressively tightening the tolerance."""

    abc_max_attempts_per_particle: int = 500
    """Safety cap on simulation attempts per particle before giving up."""

    abc_n_posterior_samples: int = 1000
    """Final posterior samples drawn from the weighted particle cloud."""

    # Prior bounds — [sigma_spatial, mean_degree, cluster_strength, beta]
    prior_low: List[float] = field(
        default_factory=lambda: [30.0, 8.0, 0.0, 0.4]
    )
    prior_high: List[float] = field(
        default_factory=lambda: [400.0, 25.0, 0.9, 3.0]
    )
    param_names: List[str] = field(
        default_factory=lambda: [
            "sigma_spatial", "mean_degree", "cluster_strength", "beta"
        ]
    )

    # ── Stage 6: Validation ───────────────────────────────────────────────
    n_validation_sims: int = 30
    """Number of posterior-predictive simulations for PPC."""
