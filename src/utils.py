"""
Geometry, Collision Detection, and Fitness Evaluation Utilities.

Provides continuous raycasting, obstacle intersection checking, path length,
smoothness metric calculations, and post-processing line-of-sight shortcutting.
"""

from typing import List, Tuple
import numpy as np
from src.environment import GridEnvironment
from src.config import COLLISION_PENALTY, SMOOTHNESS_WEIGHT


def get_line_points(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    step_size: float = 0.25,
) -> List[Tuple[float, float]]:
    """
    Interpolates points along the line segment from p1 to p2 with fine granularity
    to ensure thorough collision detection across grid cell boundaries.
    """
    dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if dist == 0:
        return [p1]
    
    num_steps = max(int(np.ceil(dist / step_size)), 2)
    xs = np.linspace(p1[0], p2[0], num_steps)
    ys = np.linspace(p1[1], p2[1], num_steps)
    return list(zip(xs, ys))


def check_segment_collision(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    env: GridEnvironment,
) -> int:
    """
    Checks if a straight segment between p1 and p2 crosses obstacles or grid bounds.
    Returns the count of intercepted obstacle cells.
    """
    points = get_line_points(p1, p2, step_size=0.25)
    collided_cells = set()
    for x, y in points:
        if env.is_obstacle(x, y):
            col = int(np.floor(x))
            row = int(np.floor(y))
            collided_cells.add((row, col))
    return len(collided_cells)


def compute_path_length(path: List[Tuple[float, float]]) -> float:
    """
    Calculates the total Euclidean length of a polygonal path.
    """
    total_len = 0.0
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        total_len += np.hypot(dx, dy)
    return float(total_len)


def compute_smoothness_cost(path: List[Tuple[float, float]]) -> float:
    """
    Penalizes abrupt heading changes (sharp corners) between consecutive path segments.
    """
    if len(path) < 3:
        return 0.0
    
    smoothness_cost = 0.0
    for i in range(1, len(path) - 1):
        v1 = np.array([path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]])
        v2 = np.array([path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]])
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 > 1e-6 and norm2 > 1e-6:
            cos_theta = np.dot(v1, v2) / (norm1 * norm2)
            # Clamp to [-1.0, 1.0] to avoid numerical precision errors
            cos_theta = max(min(cos_theta, 1.0), -1.0)
            # (1 - cos_theta) is 0 for straight continuation, up to 2 for 180° reversal
            smoothness_cost += (1.0 - cos_theta)
            
    return float(smoothness_cost)


def evaluate_path_fitness(
    path: List[Tuple[float, float]],
    env: GridEnvironment,
    collision_weight: float = COLLISION_PENALTY,
    smoothness_weight: float = SMOOTHNESS_WEIGHT,
) -> Tuple[float, float, int]:
    """
    Computes fitness cost for a candidate path.
    
    Returns:
        (total_cost, path_length, total_collisions)
        Lower cost is better. Feasible paths have total_collisions == 0.
    """
    total_length = compute_path_length(path)
    total_collisions = 0
    
    for i in range(len(path) - 1):
        total_collisions += check_segment_collision(path[i], path[i + 1], env)
        
    smoothness = compute_smoothness_cost(path)
    
    # Penalize collisions heavily so obstacle-free paths always dominate
    cost = total_length + (collision_weight * total_collisions) + (smoothness_weight * smoothness)
    return cost, total_length, total_collisions


def shortcut_path(
    path: List[Tuple[float, float]],
    env: GridEnvironment,
) -> List[Tuple[float, float]]:
    """
    Raycasting post-processing: streamlines the path by pruning redundant waypoints
    whenever direct line-of-sight between non-adjacent points is obstacle-free.
    """
    if len(path) <= 2:
        return path
        
    smoothed = [path[0]]
    curr_idx = 0
    
    while curr_idx < len(path) - 1:
        farthest_idx = curr_idx + 1
        for next_idx in range(len(path) - 1, curr_idx, -1):
            if check_segment_collision(path[curr_idx], path[next_idx], env) == 0:
                farthest_idx = next_idx
                break
        smoothed.append(path[farthest_idx])
        curr_idx = farthest_idx
        
    return smoothed
