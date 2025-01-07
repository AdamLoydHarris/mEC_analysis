# simulate.py

import numpy as np
import os
from environment import TowerEnvironment
from place_cells import generate_population, save_placecell_rate_maps
from session import NavigationSession, build_session_sequences
import pickle  # or use np.save, etc.

def run_simulation(
    n_sessions=5, 
    n_trials=30, 
    n_neurons=50, 
    output_dir="sim_data"
):
    """
    Main function to run the entire simulation of n_sessions, each with n_trials.
    We create place cells, generate paths, and produce spikes.
    """
    # 1) Create environment
    env = TowerEnvironment()

    # 2) Build the place cell population
    place_cells = generate_population(num_neurons=n_neurons)
    save_placecell_rate_maps(place_cells, output_dir="peak_maps")

    # 3) Build sequences for each session
    session_sequences = build_session_sequences(n_sessions)

    # We store the results in a structured manner
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    simulation_data = {
        "n_sessions": n_sessions,
        "n_trials": n_trials,
        "n_neurons": n_neurons,
        "sessions": []
    }

    # 4) Loop over sessions
    for s_idx in range(n_sessions):
        seq = session_sequences[s_idx]
        session_obj = NavigationSession(env, seq)

        session_dict = {
            "session_id": s_idx+1,
            "tower_sequence": seq,
            "trials_data": []
        }

        # 5) For each trial, generate path and simulate spiking
        for t_idx in range(n_trials):
            # (a) generate the positions for this trial
            positions_timebins = session_obj.generate_path_for_trial()
            # positions_timebins is length 360

            # (b) apply speed modulation
            # We said each 90 bins is a run from tower to tower, with middle bins having higher speed
            # Let's define a speed factor from 0.5 at start/end up to 1.5 at middle as an example
            speed_factors = _speed_modulation_profile(positions_timebins)

            # (c) compute spikes
            # We'll store spikes in an array of shape (n_neurons, 360)
            trial_spikes = np.zeros((n_neurons, len(positions_timebins)), dtype=int)

            for neuron_idx, pc in enumerate(place_cells):
                for bin_idx, tower in enumerate(positions_timebins):
                    rate = pc.get_rate(current_tower=tower, speed_factor=speed_factors[bin_idx])
                    # Poisson draw with mean = rate
                    # Here, we assume 1 time-bin = 1 arbitrary time unit
                    # In real simulations, multiply rate * bin_width if needed
                    trial_spikes[neuron_idx, bin_idx] = np.random.poisson(rate)

            # Collect trial data
            trial_dict = {
                "trial_id": t_idx+1,
                "positions_timebins": positions_timebins,
                "spikes": trial_spikes
            }
            session_dict["trials_data"].append(trial_dict)

        simulation_data["sessions"].append(session_dict)

    # 6) Save data
    save_path = os.path.join(output_dir, "simulation_data.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(simulation_data, f)

    print(f"Simulation completed. Data saved to {save_path}")

def _speed_modulation_profile(positions_timebins):
    """
    Returns a list/array of speed factors for each bin (length 360 in each trial).
    We want to reflect a 'U-shaped' speed at the start/end of each 90-bin leg,
    with higher speed in the middle. Let's define something simple:
        - Bins 0-15: speed ramps up from 0.5 to 1.5
        - Bins 15-75: speed stays around 1.5
        - Bins 75-90: speed ramps down from 1.5 to 0.5
    Then repeat for each 90 bin segment.
    """
    n_bins_total = len(positions_timebins)  # 360
    speed_factors = np.zeros(n_bins_total)

    for segment_start in [0, 90, 180, 270]:
        segment_end = segment_start + 90
        for i in range(segment_start, segment_end):
            frac = (i - segment_start) / 90.0
            # We'll do a simple "accelerate-decelerate" pattern:
            if frac < 0.2:
                # accelerate from 0.5 to 1.5
                speed = 0.5 + (1.0 * (frac / 0.2))
            elif frac > 0.8:
                # decelerate from 1.5 to 0.5
                speed = 1.5 - (1.0 * ((frac - 0.8) / 0.2))
            else:
                # cruise at 1.5
                speed = 1.5
            speed_factors[i] = speed

    return speed_factors

if __name__ == "__main__":
    run_simulation()
