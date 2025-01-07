# session.py

import numpy as np
from environment import TowerEnvironment

class NavigationSession:
    def __init__(self, env, tower_sequence):
        """
        env: TowerEnvironment
        tower_sequence: originally 4 towers, e.g. [1, 5, 7, 3]
        We will append the first tower so that the path forms
        a loop with 4 transitions: (1->5, 5->7, 7->3, 3->1).
        """
        self.env = env
        # Make sure the last tower is the same as the first:
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
        We want to fill n_bins by walking sequentially through
        all consecutive pairs in tower_list, distributing time evenly.
        E.g. BFS path might be [1, 2, 5, 8, 9].
        We'll treat that as 4 segments, each assigned ~n_bins/4 bins, etc.
        """
        path_bins = []

        # If there's only one tower, just fill them all:
        if len(tower_list) == 1:
            return [tower_list[0]] * n_bins

        total_steps = len(tower_list) - 1
        bins_per_step = n_bins // total_steps  # integer division
        remainder = n_bins % total_steps

        for step_idx in range(total_steps):
            start_tower = tower_list[step_idx]
            end_tower   = tower_list[step_idx + 1]

            # For more realism, let's create a small linear ramp:
            # e.g. if bins_per_step = 20, we might do 
            # t=0 -> start_tower, t=10 -> end_tower, something in between.
            # But if you only care about tower identity, let's do:
            # half the bins = start tower, half the bins = end tower, or 
            # something more advanced. For simplicity:
            step_bins = bins_per_step
            
            # If we want to fairly distribute the leftover bins, 
            # give 1 extra bin to some steps:
            if step_idx < remainder:
                step_bins += 1

            # For illustration, let's just linearly fade from start_tower to end_tower 
            # in integer steps. But if you only store "tower IDs" (like 1..9),
            # there's no partial tower. Instead, let's store each bin as the "closest" tower 
            # if you want a purely discrete sense.

            # We'll do a simple approach: first half of step_bins = start_tower, 
            # second half = end_tower. 
            # In a real continuous environment, you'd interpolate or step BFS sub-locations.
            half = step_bins // 2
            for i in range(step_bins):
                if i < half:
                    path_bins.append(start_tower)
                else:
                    path_bins.append(end_tower)

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
