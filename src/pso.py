"""
Particle Swarm Optimization (PSO) Algorithm for 2D Grid Path Planning.

Implements the Particle and PSOPathPlanner classes with continuous waypoint
representation, adaptive inertia weighting, and collision-penalized fitness.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from src.environment import GridEnvironment
from src.utils import evaluate_path_fitness, shortcut_path, compute_path_length
from src.config import (
    DEFAULT_NUM_PARTICLES,
    DEFAULT_NUM_ITERATIONS,
    DEFAULT_NUM_WAYPOINTS,
    W_MAX,
    W_MIN,
    C1,
    C2,
)


class Particle:
    """
    Individual particle in the swarm representing candidate waypoints for a path.
    """

    def __init__(self, num_waypoints: int, grid_size: int):
        self.num_waypoints = num_waypoints
        self.grid_size = grid_size
        
        # Position: 2D coordinates for each intermediate waypoint [D, 2]
        self.position: np.ndarray = np.zeros((num_waypoints, 2), dtype=float)
        # Velocity: Displacement vector for each waypoint [D, 2]
        self.velocity: np.ndarray = np.zeros((num_waypoints, 2), dtype=float)
        
        # Personal best (pbest)
        self.best_position: np.ndarray = np.zeros((num_waypoints, 2), dtype=float)
        self.best_cost: float = float("inf")
        self.best_length: float = float("inf")
        self.best_collisions: int = 999999
        
        # Current state
        self.current_cost: float = float("inf")
        self.current_length: float = float("inf")
        self.current_collisions: int = 999999

    def get_full_path(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        use_best: bool = False,
    ) -> List[Tuple[float, float]]:
        """
        Constructs the complete polygonal path: Start -> Waypoints -> Goal.
        """
        pos = self.best_position if use_best else self.position
        waypoints = [(float(pos[i, 0]), float(pos[i, 1])) for i in range(self.num_waypoints)]
        return [start] + waypoints + [goal]

    def evaluate(
        self,
        env: GridEnvironment,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> float:
        """
        Evaluates the candidate path cost and updates personal best.
        """
        full_path = self.get_full_path(start, goal, use_best=False)
        cost, length, collisions = evaluate_path_fitness(full_path, env)
        
        self.current_cost = cost
        self.current_length = length
        self.current_collisions = collisions
        
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_length = length
            self.best_collisions = collisions
            self.best_position = np.copy(self.position)
            
        return cost

    def update_velocity(
        self,
        global_best_position: np.ndarray,
        w: float,
        c1: float,
        c2: float,
        v_max: float,
    ) -> None:
        """
        Applies standard PSO velocity update formula with velocity clamping:
        v(t+1) = w * v(t) + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        """
        r1 = np.random.rand(self.num_waypoints, 2)
        r2 = np.random.rand(self.num_waypoints, 2)
        
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        
        self.velocity = (w * self.velocity) + cognitive + social
        
        # Velocity clamping to prevent particles from overshooting bounds
        self.velocity = np.clip(self.velocity, -v_max, v_max)

    def update_position(self) -> None:
        """
        Updates particle position and clamps waypoints within the grid boundaries:
        x(t+1) = x(t) + v(t+1)
        """
        self.position += self.velocity
        self.position = np.clip(self.position, 0.0, float(self.grid_size - 1))


class PSOPathPlanner:
    """
    Particle Swarm Optimization manager for 2D obstacle-avoiding path planning.
    """

    def __init__(
        self,
        env: GridEnvironment,
        num_particles: int = DEFAULT_NUM_PARTICLES,
        num_iterations: int = DEFAULT_NUM_ITERATIONS,
        num_waypoints: int = DEFAULT_NUM_WAYPOINTS,
        w_max: float = W_MAX,
        w_min: float = W_MIN,
        c1: float = C1,
        c2: float = C2,
    ):
        self.env = env
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.num_waypoints = num_waypoints
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.v_max = 0.25 * env.grid_size
        
        self.particles: List[Particle] = []
        self.gbest_position: Optional[np.ndarray] = None
        self.gbest_cost: float = float("inf")
        self.gbest_length: float = float("inf")
        self.gbest_collisions: int = 999999
        
        # Convergence monitoring
        self.convergence_history: List[float] = []
        self.length_history: List[float] = []
        self.collision_history: List[int] = []

    def _initialize_swarm(self) -> None:
        """
        Initializes particles with a hybrid strategy:
        - 60% with baseline interpolation + Gaussian perturbations
        - 40% uniform random distribution for broad exploratory coverage
        """
        self.particles = []
        start = np.array(self.env.start)
        goal = np.array(self.env.goal)
        grid_sz = float(self.env.grid_size)
        
        num_guided = int(0.6 * self.num_particles)
        
        for p_idx in range(self.num_particles):
            p = Particle(self.num_waypoints, self.env.grid_size)
            
            if p_idx < num_guided:
                # Interpolate evenly between start and goal
                for k in range(self.num_waypoints):
                    t = (k + 1) / (self.num_waypoints + 1)
                    base_point = start + t * (goal - start)
                    # Add normal perturbation perpendicular and along path
                    perturbation = np.random.normal(0.0, 1.5, size=2)
                    p.position[k] = np.clip(base_point + perturbation, 0.0, grid_sz - 1.0)
            else:
                # Uniform random across grid
                p.position = np.random.uniform(0.0, grid_sz - 1.0, size=(self.num_waypoints, 2))
                
            # Random initial velocity
            p.velocity = np.random.uniform(-self.v_max / 2, self.v_max / 2, size=(self.num_waypoints, 2))
            
            # Initial evaluation
            p.evaluate(self.env, self.env.start, self.env.goal)
            self.particles.append(p)

        # Initialize global best
        best_p = min(self.particles, key=lambda p: p.best_cost)
        self.gbest_position = np.copy(best_p.best_position)
        self.gbest_cost = best_p.best_cost
        self.gbest_length = best_p.best_length
        self.gbest_collisions = best_p.best_collisions

    def optimize(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Runs the PSO optimization loop.
        """
        self._initialize_swarm()
        
        self.convergence_history.append(self.gbest_cost)
        self.length_history.append(self.gbest_length)
        self.collision_history.append(self.gbest_collisions)
        
        if verbose:
            print(f"[PSO INIT] Best Cost: {self.gbest_cost:.2f} | "
                  f"Length: {self.gbest_length:.2f} | Collisions: {self.gbest_collisions}")

        for it in range(1, self.num_iterations + 1):
            # Dynamic linearly decreasing inertia weight
            w = self.w_max - ((self.w_max - self.w_min) * (it / self.num_iterations))
            
            for particle in self.particles:
                # Update velocity & position
                particle.update_velocity(self.gbest_position, w, self.c1, self.c2, self.v_max)
                particle.update_position()
                
                # Evaluate new position
                cost = particle.evaluate(self.env, self.env.start, self.env.goal)
                
                # Update global best if better
                if cost < self.gbest_cost:
                    self.gbest_cost = cost
                    self.gbest_length = particle.current_length
                    self.gbest_collisions = particle.current_collisions
                    self.gbest_position = np.copy(particle.position)

            self.convergence_history.append(self.gbest_cost)
            self.length_history.append(self.gbest_length)
            self.collision_history.append(self.gbest_collisions)

            if verbose and (it % 25 == 0 or it == self.num_iterations):
                status = "FEASIBLE" if self.gbest_collisions == 0 else f"COLLISIONS={self.gbest_collisions}"
                print(f"[Iteration {it:3d}/{self.num_iterations}] Best Cost: {self.gbest_cost:7.2f} | "
                      f"Path Length: {self.gbest_length:6.2f} | Status: {status}")

        # Construct final best path
        waypoints = [(float(self.gbest_position[i, 0]), float(self.gbest_position[i, 1]))
                     for i in range(self.num_waypoints)]
        raw_best_path = [self.env.start] + waypoints + [self.env.goal]
        
        # Shortcut / prune redundant waypoints
        shortened_path = shortcut_path(raw_best_path, self.env)
        shortened_len = compute_path_length(shortened_path)

        return {
            "best_path": raw_best_path,
            "best_cost": self.gbest_cost,
            "best_length": self.gbest_length,
            "best_collisions": self.gbest_collisions,
            "is_valid": (self.gbest_collisions == 0),
            "shortened_path": shortened_path,
            "shortened_length": shortened_len,
            "convergence_history": self.convergence_history,
            "length_history": self.length_history,
            "collision_history": self.collision_history,
            "num_particles": self.num_particles,
            "num_iterations": self.num_iterations,
        }
