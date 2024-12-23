# environment.py

import numpy as np
from collections import defaultdict

class TowerEnvironment:
    def __init__(self):
        """
        Define a 3x3 grid of towers, labeled 1-9. We'll store adjacency
        in a dictionary where keys are tower indices (1-9) and values
        are lists of connected neighbors.
        """
        self.adjacency = self._build_adjacency()

    def _build_adjacency(self):
        """
        Returns a dictionary: tower -> list of neighbors
        For the 3x3 grid, we assume direct vertical/horizontal adjacency.
        """
        adjacency = defaultdict(list)

        # Layout (towers):
        # 1  2  3
        # 4  5  6
        # 7  8  9

        # We can manually encode adjacency or do it programmatically.
        # Let's do it manually for clarity:
        adjacency[1] = [2, 4]
        adjacency[2] = [1, 3, 5]
        adjacency[3] = [2, 6]
        adjacency[4] = [1, 5, 7]
        adjacency[5] = [2, 4, 6, 8]
        adjacency[6] = [3, 5, 9]
        adjacency[7] = [4, 8]
        adjacency[8] = [5, 7, 9]
        adjacency[9] = [6, 8]

        return adjacency

    def get_shortest_path(self, start_tower, end_tower):
        """
        Compute the shortest path from start_tower to end_tower using BFS
        (since this is a simple unweighted graph).
        Returns a list of towers from start to end.
        """
        from collections import deque

        visited = set()
        queue = deque([[start_tower]])

        while queue:
            path = queue.popleft()
            node = path[-1]
            if node == end_tower:
                return path  # Found the path
            elif node not in visited:
                visited.add(node)
                for neighbor in self.adjacency[node]:
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)

        # Should never happen if the graph is fully connected
        return None
