# session.py

import numpy as np
from environment import TowerEnvironment

class NavigationSession:
    def __init__(self, env, tower_sequence):
        """
        env: a TowerEnvironment instance
        tower_sequence: list or array of tower IDs of length 4 that 
                        define the repeated loop in this session.
        """
        self.env = env
        # Append the first tower to ensure 4 transitions:
        if len(tower_sequence) == 4:
            tower_sequence = tower_sequence + [tower_sequence[0]]
        self.tower_sequence = tower_sequence

    def generate_path_for_trial(self):
        """
        For one trial, we produce a path that includes going from 
        tower_sequence[0] -> tower_sequence[1] -> tower_sequence[2] -> tower_sequence[3]
        and we time-bin it into 4 segments of 90 bins each (total 360).
        
        We'll return:
         - positions (an array of length 360) of tower IDs or edges.
           For simplicity, let's just store the tower ID if the mouse 
           is exactly at a tower, or the "closest" tower if mid-edge.
        """
        full_path = []
        # Build the actual tower path for the sequence
        for i in range(len(self.tower_sequence) - 1):
            start_t = self.tower_sequence[i]
            end_t = self.tower_sequence[i + 1]
            segment_path = self.env.get_shortest_path(start_t, end_t)
            # The BFS includes the start tower, so remove it from each subsequent segment 
            # so we don't double-count towers at transitions
            if i > 0:
                segment_path = segment_path[1:]
            full_path += segment_path

        # For a single trial, we want exactly 4 segments → 4 legs → each leg is 90 bins.
        # So total 360 bins. We'll do a simplistic mapping:
        # - If the shortest path from tower A to B is L towers, we "stretch" or "compress"
        #   that path over 90 bins.

        positions_timebins = []
        # We'll break it up by each leg separately, to ensure each is 90 bins.

        for i in range(len(self.tower_sequence) - 1):
            start_t = self.tower_sequence[i]
            end_t = self.tower_sequence[i + 1]
            seg_path = self.env.get_shortest_path(start_t, end_t)
            # Remove the first tower if not the first leg
            if i > 0:
                seg_path = seg_path[1:]

            # Now seg_path is a list of towers along this leg
            # We want to map them onto 90 time bins
            seg_positions = self._distribute_path_over_time(seg_path, 90)
            positions_timebins.extend(seg_positions)

        return positions_timebins  # length = 360

    def _distribute_path_over_time(self, tower_list, n_bins):
        """
        Distribute the set of towers tower_list over n_bins, linearly. 
        For example, if tower_list = [1,2,5], we have 3 tower steps. 
        We'll allocate floor(n_bins / (num_steps-1)) bins between transitions, etc.
        Simplistically, we can just repeat each tower for the fraction of bins that 
        correspond to that step. 
        """
        if len(tower_list) == 1:
            # If there's only a single tower (start==end), just fill all bins with that tower
            return [tower_list[0]] * n_bins

        path_bins = []
        num_steps = len(tower_list) - 1
        bins_per_step = n_bins // num_steps  # integer division

        for step in range(num_steps):
            start_tower = tower_list[step]
            end_tower = tower_list[step + 1]
            # We'll fill 'bins_per_step' time bins with the start tower for simplicity 
            # (could interpolate in a more fine-grained manner if you wanted).
            path_bins += [start_tower] * bins_per_step

        # If there's leftover bins (due to integer division rounding), append the final tower
        leftover = n_bins - len(path_bins)
        path_bins += [tower_list[-1]] * leftover

        return path_bins

def build_session_sequences(n_sessions=5):
    """
    Return a list of tower sequences, one for each session.
    Each sequence is a list of 4 unique towers. 
    (You can tailor how these sequences are chosen.)
    """
    sessions = []
    # Example: random sequences, or you define them manually:
    # E.g. session 1: [1,5,7,3], session 2: [2,6,8,9], etc.
    # For demonstration, let’s do random sequences:
    all_towers = range(1, 10)
    for i in range(n_sessions):
        seq = np.random.choice(all_towers, size=4, replace=False).tolist()
        sessions.append(seq)
    return sessions
