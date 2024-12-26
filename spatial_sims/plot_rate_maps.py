# plot_rate_maps.py
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

def compute_spatial_rates(sim_data):
    """
    Given simulation_data (the dict loaded from simulation_data.pkl),
    compute average firing rate per neuron, per tower, for each session.
    
    Returns: 
        rates: np.array of shape (n_sessions, n_neurons, 9),
               where rates[s, n, tower_id-1] = average rate
               for session s, neuron n, tower # tower_id.
    """
    n_sessions = sim_data["n_sessions"]
    n_neurons = sim_data["n_neurons"]
    
    # We'll store total spikes and total bins in each tower.
    # shape: (n_sessions, n_neurons, 9) 
    spike_counts = np.zeros((n_sessions, n_neurons, 9), dtype=float)
    bin_counts   = np.zeros((n_sessions, n_neurons, 9), dtype=float)
    
    # Loop over sessions
    for s_idx, sess_dict in enumerate(sim_data["sessions"]):
        # Each session has a list of trial data
        trials_data = sess_dict["trials_data"]
        # Loop over each trial
        for t_dict in trials_data:
            positions_timebins = t_dict["positions_timebins"]  # length ~ 360 (or 270, etc.)
            spikes_array = t_dict["spikes"]  # shape: (n_neurons, T)
            T = spikes_array.shape[1]        # number of time bins in this trial
            
            for t_bin in range(T):
                tower_id = positions_timebins[t_bin]
                # Convert tower_id (1..9) to an index 0..8
                tower_idx = tower_id - 1
                
                # For each neuron, add spikes
                spike_counts[s_idx, :, tower_idx] += spikes_array[:, t_bin]
                # For each neuron, increment bin_counts
                bin_counts[s_idx, :, tower_idx] += 1

    # Now compute average rates
    # Avoid divide by zero: mask bins that are zero
    with np.errstate(divide='ignore', invalid='ignore'):
        rates = np.where(bin_counts > 0, spike_counts / bin_counts, 0.0)
    
    return rates


def plot_rate_maps(sim_data, output_dir="plots"):
    """
    - Compute each neuron's 3x3 average firing rate for each session.
    - Create a figure per session, with subplots for each neuron.
    - Save each figure to disk in `output_dir`.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1) Compute the spatial rates array of shape (n_sessions, n_neurons, 9)
    rates = compute_spatial_rates(sim_data)
    
    n_sessions = sim_data["n_sessions"]
    n_neurons = sim_data["n_neurons"]
    
    # For each session, we'll make a figure
    for s_idx in range(n_sessions):
        # Decide on subplot layout: e.g. 5x10 if you have 50 neurons, or sqrt-based
        # We'll do a simple square-ish layout:
        ncols = int(np.ceil(np.sqrt(n_neurons)))
        nrows = int(np.ceil(n_neurons / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
        axes = np.ravel(axes)  # Flatten so we can index easily

        for n_idx in range(n_neurons):
            # 2) Extract the 9 values for this session, neuron
            rate_vec = rates[s_idx, n_idx, :]  # shape (9,)
            
            # 3) Reshape to 3x3
            rate_map = rate_vec.reshape(3,3)
            
            # 4) Plot it
            ax = axes[n_idx]
            cax = ax.imshow(rate_map, 
                            cmap='viridis', 
                            origin='upper', 
                            interpolation='nearest')
            
            ax.set_title(f"Neuron {n_idx} (Session {s_idx+1})")
            ax.axis('off')
            fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

        # Adjust layout
        plt.tight_layout()
        
        # 5) Save figure
        plt.savefig(os.path.join(output_dir, f"rate_maps_session_{s_idx+1}.png"))
        plt.close(fig)

def main():
    # Example usage
    # Load the simulation data from a file
    with open("sim_data/simulation_data.pkl", "rb") as f:
        sim_data = pickle.load(f)
    
    # Generate & save the figures
    plot_rate_maps(sim_data, output_dir="plots")

if __name__ == "__main__":
    main()
