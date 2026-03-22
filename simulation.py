import sys
sys.path.insert(0, r'C:\ising_model_code')  # Add workspace to path
import mycpp
import numpy as np
from config import SimulationConfig
from scipy.spatial.distance import cdist
from typing import Optional, Tuple, Dict


rng = np.random.default_rng(42)

def poisson_disc_sampling(config: SimulationConfig, m, r_max, r_min, rng):

    points = []
    max_attempts = config.max_attempts
    epsilon = 1e-7
    
    for i in range(m):
        point_placed = False
        attempts = 0

        while not point_placed and attempts < max_attempts:
            attempts += 1
            
            
            rand = rng.random() + epsilon
            angle = 2 * np.pi * np.sqrt(rand)
            r = r_max * rng.random()
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            too_close = False

           
            for j in range(i):
                x_j, y_j = points[j]
                dx = x - x_j
                dy = y - y_j
                if dx**2 + dy**2 < r_min**2:
                    too_close = True
                    break
            
            if not too_close :
                points.append((x, y))
                point_placed = True

            if attempts > max_attempts:
                count = len(points)
                n_points = m
                print(f"Warning: Could only place {count}/{n_points} clusters. Space is too crowded!")
                return points[:count] 
                


        if not point_placed:
            raise RuntimeError("FATAL: could not generate point")
    
    return points

# print(poisson_disc_sampling(SimulationConfig, 5, 500, 250, rng))


def generate_neuron_positions(config: SimulationConfig):

    positions = []

    n_clusters = config.n_clusters
    radius = config.arena_size
    min_cluster_center_distance = config.min_cluster_center_distance
    cluster_centers = poisson_disc_sampling(config, n_clusters, radius, min_cluster_center_distance, rng)
    cluster_size = config.cluster_size
    cluster_radius = config.cluster_radius
    min_distance = config.min_dist

    for i in range(n_clusters):
        positions_cluster_i = poisson_disc_sampling(config, cluster_size, cluster_radius, min_distance, rng)
        center = np.array(cluster_centers[i])
        for j in range(cluster_size):
            rel_pos = np.array(positions_cluster_i[j])
            abs_pos = center + rel_pos
            positions.append((float(abs_pos[0]), float(abs_pos[1])))

    positions = [tuple(p) for p in positions]

    return positions




# ─────────────────────────────────────────────────────────────────────────────
# Connectivity
# ─────────────────────────────────────────────────────────────────────────────

def generate_connectivity(
    positions: np.ndarray,
    config: SimulationConfig,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct the Ising coupling matrix J and binary adjacency matrix A.
 
    Edge probability model
    ~~~~~~~~~~~~~~~~~~~~~~
    1. Base kernel:  p_base(i,j) = exp(-d_ij^2 / (2 sigma^2))
    2. Split the degree budget between intra-cluster and inter-cluster:
 
         f_intra = f_natural + cluster_strength * (1 - f_natural)
 
       where f_natural is the fraction of edges the distance kernel would
       naturally place within clusters.  At cluster_strength=0 this is
       purely distance-dependent; at cluster_strength=1 all edges are
       intra-cluster.
 
    3. Scale intra and inter base probabilities SEPARATELY so that:
         E[intra_degree] = mean_degree * f_intra
         E[inter_degree] = mean_degree * (1 - f_intra)
 
       The distance kernel shape is preserved within each group (closer
       pairs are still more likely to connect), but cluster_strength
       controls the fraction of the total degree budget allocated to
       intra-cluster connections.
 
    Coupling weights ~ |N(1, 0.3)| / sqrt(mean_degree)  (normalised).
 
    Parameters
    ----------
    positions : (N, 2) neuron positions (um).
    config    : SimulationConfig carrying sigma, mean_degree, cluster_strength.
    rng       : optional pre-seeded Generator.
 
    Returns
    -------
    J : (N, N) symmetric coupling matrix, zero diagonal.
    A : (N, N) binary adjacency matrix.
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)
 
    N = len(positions)
    D = cdist(positions, positions)   # (N, N) Euclidean distances
 
    # ── Step 1: Base Gaussian edge-probability kernel ─────────────────────
    base_prob = np.exp(-D ** 2 / (2.0 * config.sigma_spatial ** 2))
    np.fill_diagonal(base_prob, 0.0)
 
    # ── Step 2: Build intra/inter masks ───────────────────────────────────
    cluster_ids = np.array_split(np.arange(N), config.n_clusters)
    intra_mask = np.zeros((N, N), dtype=bool)
    for cids in cluster_ids:
        intra_mask[np.ix_(cids, cids)] = True
    np.fill_diagonal(intra_mask, False)
    inter_mask = ~intra_mask
    np.fill_diagonal(inter_mask, False)
 
    # ── Step 3: Compute natural intra-fraction from distance kernel ───────
    sum_intra = base_prob[intra_mask].sum()
    sum_inter = base_prob[inter_mask].sum()
    sum_total = sum_intra + sum_inter
 
    if sum_total < 1e-12:
        # Degenerate case: all probabilities are zero
        return np.zeros((N, N)), np.zeros((N, N))
 
    f_natural = sum_intra / sum_total
 
    # Interpolate: at c=0 purely distance-based, at c=1 fully intra-cluster
    c = config.cluster_strength
    f_intra = f_natural + c * (1.0 - f_natural)
 
    # ── Step 4: Target degrees for each group ─────────────────────────────
    target_intra_deg = config.mean_degree * f_intra
    target_inter_deg = config.mean_degree * (1.0 - f_intra)
 
    # ── Step 5: Scale each group separately ───────────────────────────────
    # Expected intra degree per neuron from unscaled base:
    #   E[intra_deg] = sum_intra / N
    # Scale factor to hit target:
    #   s_intra = target_intra_deg / (sum_intra / N)
    prob = np.zeros((N, N))
 
    exp_intra_deg = sum_intra / N
    if exp_intra_deg > 1e-12 and target_intra_deg > 1e-12:
        s_intra = target_intra_deg / exp_intra_deg
        prob[intra_mask] = base_prob[intra_mask] * s_intra
 
    exp_inter_deg = sum_inter / N
    if exp_inter_deg > 1e-12 and target_inter_deg > 1e-12:
        s_inter = target_inter_deg / exp_inter_deg
        prob[inter_mask] = base_prob[inter_mask] * s_inter
 
    # Clip to [0, 1] — warn if significant clipping occurs
    n_clipped = np.sum(prob > 1.0)
    prob = np.clip(prob, 0.0, 1.0)
    np.fill_diagonal(prob, 0.0)
 
    if n_clipped > 0:
        actual_expected = prob.sum() / N
        if actual_expected < config.mean_degree * 0.9:
            import warnings
            warnings.warn(
                f"Probability clipping reduced expected mean degree to "
                f"{actual_expected:.1f} (target {config.mean_degree:.1f}).  "
                f"Network is near saturation for this cluster_strength / "
                f"mean_degree combination.",
                UserWarning, stacklevel=2,
            )
 
    # ── Sample adjacency (upper triangle -> symmetrise) ───────────────────
    rand = rng.random((N, N))
    A = (rand < prob).astype(np.float64)
    A = np.triu(A, 1)
    A = A + A.T                        # symmetric, zero diagonal
 
    # ── Coupling weights ──────────────────────────────────────────────────
    W = np.abs(rng.normal(1.0, 0.3, (N, N)))
    W = (W + W.T) / 2.0
    scale = max(np.sqrt(config.mean_degree), 1e-3)
    J = A * W / scale
    J_np = np.array(J, dtype=np.float64)
    A_np = np.array(A, dtype=np.float64)
 
    return J_np, A_np





# ─────────────────────────────────────────────────────────────────────────────
# Main simulation function
# ─────────────────────────────────────────────────────────────────────────────

def simulate_kndy_network(
    config: SimulationConfig,
    J: Optional[np.ndarray] = None,
    positions: Optional[np.ndarray] = None,
    seed_override: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Simulate the KNDy network and return spike trains.

    Parameters
    ----------
    config          : SimulationConfig
    J               : (N, N) coupling matrix.  Generated from config if None.
    positions       : (N, 2) positions in µm.  Generated from config if None.
    return_peptides : if True, record neuropeptide traces.
    seed_override   : replace config.seed for this call only (useful in ABC).

    Returns
    -------
    dict with keys:
        'spikes'    : (T_rec, N) int8 — binary spike trains after burn-in
        'positions' : (N, 2)  float  — neuron positions
        'J'         : (N, N)  float  — coupling matrix used
        'A'         : (N, N)  float  — binary adjacency
        'time'      : (T_rec,) float — time axis in seconds
        'peptides'  : (T_rec, 3) float [NKB, Dyn, Kiss]  (if requested)
    """
    seed = seed_override if seed_override is not None else config.seed
    rng  = np.random.default_rng(seed)
    

    # ── Setup ──────────────────────────────────────────────────────────────
    if positions is None:
        positions = generate_neuron_positions(config)
    if J is None:
        J, A = generate_connectivity(positions, config, rng)
    else:
        A = (J != 0.0).astype(np.float64)
    """
    Diagnostics:

    print(f"DEBUG: type(positions) = {type(positions)}")
    print(f"DEBUG: type(positions[0]) = {type(positions[0])}")
    print(f"DEBUG: A.shape = {A.shape}, A.dtype = {A.dtype}")
    print(f"DEBUG: n_neurons from config = {config.n_neurons}")
    
    print("DEBUG: C++ returned, len(spikes)=", len(spikes))  # if this never prints, C++ crashed
    """

    spikes = mycpp.simulate_kndy_network(config, positions, J, A)
    spikes = np.array(spikes, dtype=np.int8)

    if spikes.ndim != 2:
        raise ValueError(
            f"mycpp returned spikes with shape {spikes.shape} — expected 2D (T, N). "
            f"Check that the C++ function is receiving valid inputs."
        )

    N_actual = len(positions)
    if spikes.shape[0] == N_actual and spikes.shape[1] != N_actual:
        spikes = spikes.T   # transpose (N, T) → (T, N) if needed

    T = spikes.shape[0]
    time = np.arange(T, dtype=np.float32) * config.dt

    


    result = {
        "spikes":    spikes,
        "positions": positions,
        "J":         J,
        "A":         A,
        "time":      np.arange(T, dtype=np.float32) * config.dt,
    }

    result["spikes"] = np.array(result["spikes"])
    result["positions"] = np.array(result["positions"])

    """
     Diagnostics:
    print(f"Lengths: {set(len(r) for r in result["spikes"])}")
    T = result["spikes"].shape[0]
    print([np.mean(result["spikes"][i]) for i in range(T) ])
    """
   
    return result







"""
Diagnostics:
cfg    = SimulationConfig(n_neurons=100, seed=42)
result = simulate_kndy_network(cfg)
print("keys:", list(result.keys()))
print("spikes shape:", result["spikes"].shape)   # must be (T, N), not (N, T)
print("spikes dtype:", result["spikes"].dtype)   # must be int or bool
print("spikes sum: ", result["spikes"].sum())    # must be >> 0
print("spikes ndim:", result["spikes"].ndim)     # must be 2
spikes = result["spikes"]
pop = spikes.mean(axis=1)       # (T,) population firing fraction per time step
print("mean:", pop.mean())      # should be ~0.1–0.4 for pulsatile KNDy activity
print("max: ", pop.max())       # should reach ~0.3–1.0 during sync events
print("fraction of silent timesteps:", (pop == 0).mean())
from statistics import detect_sync_events
sync = detect_sync_events(result["spikes"])
print("full sync events:   ", sync["n_full_events"])
print("partial sync events:", sync["n_partial_events"])
print("IEI CV:             ", sync["iei_cv"])
for mean_deg in [20,25,30]:
    cfg = SimulationConfig(
        n_neurons=100, seed=42,
        mean_degree=mean_deg,
        g_N=1.5, g_D=0.25,
    )
    positions = generate_neuron_positions(cfg)
    result    = simulate_kndy_network(cfg, positions=positions)
    spikes    = result["spikes"]
    sync      = detect_sync_events(spikes)
    print(
        f"mean_deg={mean_deg}: "
        f"rate={spikes.mean():.3f}  "
        f"sync={sync['n_full_events']}  "
        f"partial={sync['n_partial_events']}  "
        f"IEI_CV={sync['iei_cv']:.2f}"
    )
"""



from matplotlib import pyplot as plt
cfg = SimulationConfig(n_neurons=52, seed=50)
spikes = simulate_kndy_network(cfg)["spikes"]  * 60 # convert to spikes/min
time = np.arange(spikes.shape[0]) * cfg.DT /60 # convert to minutes
  
bin_size = 1 / cfg.DT # in seconds, e.g. 100 ms for DT=0.1s
binned_time = time.reshape(-1, int(bin_size)).mean(axis=1)
binned_spikes = spikes.reshape(-1, int(bin_size), spikes.shape[1]).sum(axis=1)  # (T//bin_size, N)
figure, ax = plt.subplots()
ax.plot(binned_time, binned_spikes.mean(axis=1))
plt.plot()
plt.show()
