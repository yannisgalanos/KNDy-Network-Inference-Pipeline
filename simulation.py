import sys
sys.path.insert(0, r'C:\ising_model_code')  # Add workspace to path
import mycpp
import numpy as np
from config import SimulationConfig
from scipy.spatial.distance import cdist
from typing import Optional, Tuple, Dict

pos_seed = SimulationConfig().pos_seed
rng = np.random.default_rng(42)
geo_rng = np.random.default_rng(pos_seed)

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
    cluster_centers = poisson_disc_sampling(config, n_clusters, radius, min_cluster_center_distance, geo_rng)
    cluster_size = config.cluster_size
    cluster_radius = config.cluster_radius
    min_distance = config.min_dist

    for i in range(n_clusters):
        positions_cluster_i = poisson_disc_sampling(config, cluster_size, cluster_radius, min_distance, geo_rng)
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
) -> np.ndarray:
    """
    Construct the Ising binary adjacency matrix A.

    Edges are sampled from a Weibull distance kernel scaled so that the
    expected mean degree equals config.mean_degree.

    """
    if rng is None:
        rng = np.random.default_rng(config.seed)

    N = len(positions)
    D = cdist(positions, positions)

    # Weibull edge-probability kernel
    base_prob = np.exp(-D ** config.alpha / (2.0 * config.sigma_spatial ** config.alpha))
    np.fill_diagonal(base_prob, 0.0)

    # Scale uniformly to enforce mean_degree
    expected_deg = base_prob.sum() / N
    if expected_deg < 1e-12:
        return np.zeros((N, N)), np.zeros((N, N))

    prob = np.clip(base_prob * (config.mean_degree / expected_deg), 0.0, 1.0)
    np.fill_diagonal(prob, 0.0)

    # Sample adjacency (upper triangle -> symmetrise)
    rand = rng.random((N, N))
    A = np.triu(rand < prob, 1).astype(np.float64)
    A = A + A.T


    return np.array(A, dtype=np.float64)





# ─────────────────────────────────────────────────────────────────────────────
# Main simulation function
# ─────────────────────────────────────────────────────────────────────────────

def simulate_kndy_network(
    config: SimulationConfig,
    A: Optional[np.ndarray] = None,
    positions: Optional[np.ndarray] = None,
    seed_override: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Simulate the KNDy network and return spike trains.

    Parameters
    ----------
    config          : SimulationConfig
    A               : (N, N) coupling matrix.  Generated from config if None.
    positions       : (N, 2) positions in µm.  Generated from config if None.
    return_peptides : if True, record neuropeptide traces.
    seed_override   : replace config.seed for this call only (useful in ABC).

    Returns
    -------
    dict with keys:
        'spikes'    : (T_rec, N) int8 — binary spike trains after burn-in
        'positions' : (N, 2)  float  — neuron positions
        'A'         : (N, N)  float  — binary adjacency
        'time'      : (T_rec,) float — time axis in seconds
        'peptides'  : (T_rec, 3) float [NKB, Dyn, Kiss]  (if requested)
    """
    seed = seed_override if seed_override is not None else config.seed
    rng  = np.random.default_rng(seed)
    

    # ── Setup ──────────────────────────────────────────────────────────────
    if positions is None:
        positions = generate_neuron_positions(config)
    if A is None:
        A = generate_connectivity(positions, config, rng)

    """
    Diagnostics:

    print(f"DEBUG: type(positions) = {type(positions)}")
    print(f"DEBUG: type(positions[0]) = {type(positions[0])}")
    print(f"DEBUG: A.shape = {A.shape}, A.dtype = {A.dtype}")
    print(f"DEBUG: n_neurons from config = {config.n_neurons}")
    
    print("DEBUG: C++ returned, len(spikes)=", len(spikes))  # if this never prints, C++ crashed
    """

    spikes = mycpp.simulate_kndy_network(config, positions, A)
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
"""

