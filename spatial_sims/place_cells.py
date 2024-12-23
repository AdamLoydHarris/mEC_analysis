# place_cells.py

import numpy as np

class PlaceCell:
    def __init__(self, neuron_id, peak_locations, peak_rate=10.0, baseline_rate=1.0):
        """
        A simplistic PlaceCell representation.
        neuron_id: an integer identifier
        peak_locations: list of towers or coordinates where this neuron is tuned
        peak_rate: maximal firing rate at each peak location
        baseline_rate: baseline firing rate outside peaks
        """
        self.neuron_id = neuron_id
        self.peak_locations = peak_locations  # list of tower IDs for simplicity
        self.peak_rate = peak_rate
        self.baseline_rate = baseline_rate

    def get_rate(self, current_tower, speed_factor=1.0):
        """
        Returns the firing rate given the current tower and a speed modulation factor.
        If the tower is in the neuron's peak_locations, we return peak_rate * speed_factor,
        else baseline_rate * speed_factor.
        """
        if current_tower in self.peak_locations:
            return self.peak_rate * speed_factor
        else:
            return self.baseline_rate * speed_factor


def generate_population(num_neurons=50):
    """
    Generate a population of place cells. Each neuron can have 1-4 peaks.
    We'll randomly choose which towers they prefer as their 'peak' locations.
    """
    all_place_cells = []
    for i in range(num_neurons):
        # Number of peaks from 1 to 4
        num_peaks = np.random.randint(1, 5)
        # Randomly choose tower IDs (1-9) for these peaks
        peak_locations = np.random.choice(range(1, 10), size=num_peaks, replace=False)
        
        # You can add variability in peak_rate and baseline_rate if desired
        peak_rate = np.random.uniform(5, 15)  # random peak 5-15
        baseline_rate = np.random.uniform(0.5, 2)  # random baseline 0.5-2

        pc = PlaceCell(
            neuron_id=i,
            peak_locations=peak_locations,
            peak_rate=peak_rate,
            baseline_rate=baseline_rate
        )
        all_place_cells.append(pc)

    return all_place_cells
