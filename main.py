"""
Main Entrypoint for Swarm-Based Path Planning with Obstacles using PSO.

Course: Swarm Intelligence (Semester 7) — Lab Assignment 1
Institution: Bahria University, Department of Computer Science
Student: Muhammad Saim Khan
Roll Number: 01-136232-073
"""

import argparse
import sys
from src.config import (
    STUDENT_NAME,
    STUDENT_ROLL_NO,
    DEFAULT_SEED,
    DEFAULT_GRID_SIZE,
    DEFAULT_NUM_OBSTACLES,
    DEFAULT_NUM_PARTICLES,
    DEFAULT_NUM_ITERATIONS,
    DEFAULT_NUM_WAYPOINTS,
    roll_number_to_seed,
)
from src.environment import GridEnvironment
from src.pso import PSOPathPlanner
from src.visualizer import plot_path_planning_results


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Swarm-Based Path Planning with Obstacles using Particle Swarm Optimization (PSO)."
    )
    parser.add_argument(
        "--name",
        type=str,
        default=STUDENT_NAME,
        help=f"Student Name (default: '{STUDENT_NAME}')",
    )
    parser.add_argument(
        "--roll-no",
        type=str,
        default=STUDENT_ROLL_NO,
        help=f"Student Roll Number (default: '{STUDENT_ROLL_NO}')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Custom integer random seed (default: automatically extracted from roll number)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_GRID_SIZE,
        help=f"Dimension of square 2D grid (default: {DEFAULT_GRID_SIZE})",
    )
    parser.add_argument(
        "--obstacles",
        type=int,
        default=DEFAULT_NUM_OBSTACLES,
        help=f"Number of obstacle cells (default: {DEFAULT_NUM_OBSTACLES})",
    )
    parser.add_argument(
        "--obstacle-ratio",
        type=float,
        default=None,
        help="Optional ratio of obstacle cells to total grid cells",
    )
    parser.add_argument(
        "--particles",
        type=int,
        default=DEFAULT_NUM_PARTICLES,
        help=f"Number of swarm particles (default: {DEFAULT_NUM_PARTICLES})",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_NUM_ITERATIONS,
        help=f"Maximum PSO iterations (default: {DEFAULT_NUM_ITERATIONS})",
    )
    parser.add_argument(
        "--waypoints",
        type=int,
        default=DEFAULT_NUM_WAYPOINTS,
        help=f"Number of intermediate path waypoints per particle (default: {DEFAULT_NUM_WAYPOINTS})",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="assets/path_planning_result.png",
        help="File path to save the generated plot (default: 'assets/path_planning_result.png')",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open interactive matplotlib window after optimization",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save plot to disk",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Determine unique seed
    if args.seed is not None:
        seed = args.seed
    else:
        seed = roll_number_to_seed(args.roll_no)

    print("=" * 70)
    print(" SWARM INTELLIGENCE LAB — ASSIGNMENT 1: PSO PATH PLANNING ")
    print("=" * 70)
    print(f" Student Name   : {args.name}")
    print(f" Roll Number    : {args.roll_no}")
    print(f" Problem Seed   : {seed}")
    print(f" Grid Dimension : {args.grid_size} x {args.grid_size}")
    print(f" Obstacles Count: {args.obstacles} cells")
    print(f" Swarm Size     : {args.particles} particles")
    print(f" Iterations     : {args.iterations} cycles")
    print(f" Path Waypoints : {args.waypoints} intermediate points")
    print("=" * 70)

    # 1. Generate Problem Environment
    print("\n[STEP 1] Generating unique 2D grid problem instance...")
    env = GridEnvironment(
        grid_size=args.grid_size,
        seed=seed,
        num_obstacles=args.obstacles,
        obstacle_ratio=args.obstacle_ratio,
    )
    summary = env.get_summary()
    print(f" -> Obstacles Placed    : {summary['num_obstacles']} cells ({summary['obstacle_ratio']})")
    print(f" -> Start Coordinate   : X={env.start[0]:.1f}, Y={env.start[1]:.1f}")
    print(f" -> Goal Coordinate    : X={env.goal[0]:.1f}, Y={env.goal[1]:.1f}")
    print(f" -> Direct Distance    : {summary['euclidean_distance']:.2f} units")

    # 2. Run Particle Swarm Optimization
    print("\n[STEP 2] Running Particle Swarm Optimization (PSO)...")
    planner = PSOPathPlanner(
        env=env,
        num_particles=args.particles,
        num_iterations=args.iterations,
        num_waypoints=args.waypoints,
    )
    results = planner.optimize(verbose=True)

    # 3. Report Results
    print("\n" + "=" * 70)
    print(" OPTIMIZATION RESULTS SUMMARY ")
    print("=" * 70)
    print(f" Status               : {'SUCCESS (Collision-Free Path Found!)' if results['is_valid'] else 'FAILED (Collisions Detected)'}")
    print(f" Best Fitness Cost    : {results['best_cost']:.4f}")
    print(f" Raw PSO Path Length  : {results['best_length']:.4f} units")
    print(f" Smoothed Path Length : {results['shortened_length']:.4f} units")
    print(f" Intercepted Obstacles: {results['best_collisions']} cells")
    print(" Raw Best Waypoints   :")
    for i, pt in enumerate(results['best_path']):
        tag = "START" if i == 0 else ("GOAL" if i == len(results['best_path']) - 1 else f"WP {i}")
        print(f"    [{tag:5s}] (X={pt[0]:.2f}, Y={pt[1]:.2f})")

    # 4. Generate Visualizations
    save_path = None if args.no_save else args.save_path
    print(f"\n[STEP 3] Rendering and saving plots...")
    plot_path_planning_results(
        env=env,
        results=results,
        student_name=args.name,
        roll_number=args.roll_no,
        save_path=save_path,
        show_plot=args.show,
    )

    print("\n[DONE] Execution completed successfully!")


if __name__ == "__main__":
    main()
