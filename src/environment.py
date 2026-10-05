"""
2D Grid Environment and Deterministic Obstacle Generator.

Generates a programmatic 2D grid instance with obstacle cells, start point,
and goal point using the student's roll number as a deterministic random seed.
"""

from typing import Tuple, Set, List, Optional
import random
from collections import deque
import numpy as np


class GridEnvironment:
    """
    Represents a 2D grid navigation space with obstacles, start, and goal points.
    """

    def __init__(
        self,
        grid_size: int = 20,
        seed: int = 73,
        num_obstacles: Optional[int] = 55,
        obstacle_ratio: Optional[float] = None,
    ):
        self.grid_size = grid_size
        self.seed = seed
        if obstacle_ratio is not None:
            self.num_obstacles = int(grid_size * grid_size * obstacle_ratio)
        else:
            self.num_obstacles = num_obstacles if num_obstacles is not None else 55
        self.obstacle_ratio = self.num_obstacles / (grid_size * grid_size)
        self.obstacles: Set[Tuple[int, int]] = set()
        self.start: Tuple[float, float] = (0.0, 0.0)
        self.goal: Tuple[float, float] = (0.0, 0.0)
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        
        self._generate_instance()

    def _generate_instance(self) -> None:
        """
        Deterministically generates obstacles, start point, and goal point using self.seed.
        """
        # Set seeds for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed % (2**32 - 1))

        all_cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)]
        
        # Programmatically generate obstacle cells from seed
        sampled_obstacles = random.sample(all_cells, self.num_obstacles)
        self.obstacles = set(sampled_obstacles)

        # Free non-obstacle cells
        free_cells = [cell for cell in all_cells if cell not in self.obstacles]

        # Programmatically place start and goal points on free cells
        # Select start point
        start_cell = random.choice(free_cells)
        free_cells.remove(start_cell)

        # Select a goal point that is sufficiently distant from start for a meaningful path
        sorted_by_dist = sorted(
            free_cells,
            key=lambda c: (c[0] - start_cell[0]) ** 2 + (c[1] - start_cell[1]) ** 2,
            reverse=True,
        )
        # Pick from top candidates deterministically
        candidate_pool = sorted_by_dist[: max(1, len(sorted_by_dist) // 5)]
        goal_cell = random.choice(candidate_pool)

        # Store coordinates (x = col, y = row) for Cartesian 2D plotting & continuous planning
        # Start and Goal are represented as (x, y) float coordinates
        self.start = (float(start_cell[1]), float(start_cell[0]))
        self.goal = (float(goal_cell[1]), float(goal_cell[0]))

        # Ensure start and goal are connectable; if not, open minimum path bottleneck
        self._ensure_solvability(start_cell, goal_cell)

        # Build numpy binary grid: 0 = Free, 1 = Obstacle
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for r, c in self.obstacles:
            self.grid[r, c] = 1

    def _ensure_solvability(self, start_cell: Tuple[int, int], goal_cell: Tuple[int, int]) -> None:
        """
        Verifies with Breadth-First Search (BFS) that a collision-free corridor exists
        between start and goal. If blocked, selectively carves minimal cells to guarantee
        solvability while preserving the seeded obstacle pattern.
        """
        def is_connected() -> bool:
            queue = deque([start_cell])
            visited = {start_cell}
            while queue:
                curr = queue.popleft()
                if curr == goal_cell:
                    return True
                r, c = curr
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        if (nr, nc) not in visited and (nr, nc) not in self.obstacles:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
            return False

        if not is_connected():
            # Carve a narrow Manhattan corridor between start and goal
            cr, cc = start_cell
            gr, gc = goal_cell
            while (cr, cc) != (gr, gc):
                if cr < gr:
                    cr += 1
                elif cr > gr:
                    cr -= 1
                elif cc < gc:
                    cc += 1
                elif cc > gc:
                    cc -= 1
                if (cr, cc) in self.obstacles and (cr, cc) != goal_cell:
                    self.obstacles.remove((cr, cc))

    def is_obstacle(self, x: float, y: float) -> bool:
        """
        Checks if a continuous coordinate (x, y) falls inside an obstacle cell
        or outside the grid boundaries.
        """
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        col = int(np.floor(x))
        row = int(np.floor(y))
        # Ensure within array index bounds
        col = min(max(col, 0), self.grid_size - 1)
        row = min(max(row, 0), self.grid_size - 1)
        return (row, col) in self.obstacles

    def get_summary(self) -> dict:
        """
        Returns a dictionary summary of the environment instance.
        """
        return {
            "seed": self.seed,
            "grid_size": f"{self.grid_size}x{self.grid_size}",
            "num_obstacles": len(self.obstacles),
            "obstacle_ratio": f"{len(self.obstacles) / (self.grid_size ** 2):.1%}",
            "start_point": self.start,
            "goal_point": self.goal,
            "euclidean_distance": float(
                np.sqrt((self.goal[0] - self.start[0]) ** 2 + (self.goal[1] - self.start[1]) ** 2)
            ),
        }
