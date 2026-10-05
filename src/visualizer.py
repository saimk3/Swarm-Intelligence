"""
Visualization Module for Swarm-Based Path Planning.

Generates high-quality publication-ready plots for:
1. 2D Grid Environment with obstacles, start, goal, and optimized paths.
2. PSO Convergence Curves (Cost and Path Length across iterations).
"""

from typing import List, Tuple, Dict, Any, Optional
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from src.environment import GridEnvironment


def plot_path_planning_results(
    env: GridEnvironment,
    results: Dict[str, Any],
    student_name: str,
    roll_number: str,
    save_path: Optional[str] = "assets/path_planning_result.png",
    show_plot: bool = False,
) -> None:
    """
    Renders a side-by-side figure with:
    Left: 2D Grid Map with obstacles, start, goal, PSO best path, and smoothed shortcut path.
    Right: PSO Convergence History (Fitness Cost and Euclidean Path Length vs Iterations).
    """
    fig, (ax_grid, ax_conv) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.1, 0.9]})
    fig.patch.set_facecolor("#f8f9fa")

    # -------------------------------------------------------------
    # 1. Left Panel: 2D Grid Environment & Trajectories
    # -------------------------------------------------------------
    ax_grid.set_facecolor("#ffffff")
    grid_size = env.grid_size

    # Draw grid cells
    for r in range(grid_size):
        for c in range(grid_size):
            if (r, c) in env.obstacles:
                rect = patches.Rectangle(
                    (c, r), 1, 1,
                    linewidth=0.5,
                    edgecolor="#2c3e50",
                    facecolor="#34495e",
                    alpha=0.9,
                )
                ax_grid.add_patch(rect)
            else:
                rect = patches.Rectangle(
                    (c, r), 1, 1,
                    linewidth=0.3,
                    edgecolor="#e0e0e0",
                    facecolor="#fafafa",
                )
                ax_grid.add_patch(rect)

    # Plot PSO raw best path
    best_path = results["best_path"]
    bx = [p[0] for p in best_path]
    by = [p[1] for p in best_path]
    ax_grid.plot(
        bx, by,
        color="#e67e22",
        linestyle="-",
        linewidth=2.5,
        marker="o",
        markersize=6,
        markerfacecolor="#d35400",
        label=f"PSO Best Path (Len: {results['best_length']:.2f})",
        zorder=4,
    )

    # Plot Post-processed / Shortened Path
    short_path = results["shortened_path"]
    sx = [p[0] for p in short_path]
    sy = [p[1] for p in short_path]
    ax_grid.plot(
        sx, sy,
        color="#2980b9",
        linestyle="--",
        linewidth=3.0,
        marker="s",
        markersize=5,
        markerfacecolor="#1f618d",
        label=f"Smoothed Shortcut (Len: {results['shortened_length']:.2f})",
        zorder=5,
    )

    # Plot Start and Goal points (Centered on grid cell)
    start_x, start_y = env.start
    goal_x, goal_y = env.goal

    ax_grid.scatter(
        [start_x], [start_y],
        color="#27ae60",
        s=250,
        edgecolors="#1e8449",
        linewidths=2,
        zorder=10,
        label=f"Start ({start_x:.1f}, {start_y:.1f})",
    )
    ax_grid.text(
        start_x + 0.3, start_y + 0.3, "START",
        color="#1e8449", fontweight="bold", fontsize=10, zorder=11,
    )

    ax_grid.scatter(
        [goal_x], [goal_y],
        color="#e74c3c",
        s=300,
        marker="*",
        edgecolors="#922b21",
        linewidths=2,
        zorder=10,
        label=f"Goal ({goal_x:.1f}, {goal_y:.1f})",
    )
    ax_grid.text(
        goal_x + 0.3, goal_y + 0.3, "GOAL",
        color="#922b21", fontweight="bold", fontsize=10, zorder=11,
    )

    ax_grid.set_xlim(-0.5, grid_size + 0.5)
    ax_grid.set_ylim(-0.5, grid_size + 0.5)
    ax_grid.set_aspect("equal")
    ax_grid.set_xlabel("X (Columns)", fontsize=11, fontweight="bold")
    ax_grid.set_ylabel("Y (Rows)", fontsize=11, fontweight="bold")
    ax_grid.set_title(
        f"2D Grid Map ({grid_size}x{grid_size}) — Obstacle Avoidance\n"
        f"Student: {student_name} | Roll No: {roll_number} | Seed: {env.seed}",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax_grid.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # -------------------------------------------------------------
    # 2. Right Panel: Convergence Curves
    # -------------------------------------------------------------
    ax_conv.set_facecolor("#ffffff")
    iterations = list(range(len(results["convergence_history"])))
    
    # Plot Cost Curve
    color_cost = "#c0392b"
    ax_conv.plot(
        iterations,
        results["convergence_history"],
        color=color_cost,
        linewidth=2.2,
        label="Fitness Cost (Includes Penalties)",
    )
    ax_conv.set_xlabel("Iteration", fontsize=11, fontweight="bold")
    ax_conv.set_ylabel("Fitness Cost", color=color_cost, fontsize=11, fontweight="bold")
    ax_conv.tick_params(axis="y", labelcolor=color_cost)
    ax_conv.grid(True, linestyle=":", alpha=0.6)

    # Twin axis for Path Length
    ax_len = ax_conv.twinx()
    color_len = "#2980b9"
    ax_len.plot(
        iterations,
        results["length_history"],
        color=color_len,
        linestyle="-.",
        linewidth=2.0,
        label="Path Length (Euclidean)",
    )
    ax_len.set_ylabel("Path Length", color=color_len, fontsize=11, fontweight="bold")
    ax_len.tick_params(axis="y", labelcolor=color_len)

    ax_conv.set_title(
        f"PSO Convergence Dynamics\n"
        f"Particles: {results['num_particles']} | Max Iterations: {results['num_iterations']}",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    # Combined legend
    lines_1, labels_1 = ax_conv.get_legend_handles_labels()
    lines_2, labels_2 = ax_len.get_legend_handles_labels()
    ax_conv.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", framealpha=0.9, fontsize=9)

    plt.tight_layout()

    # Save output if path specified
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] High-resolution visualization saved to: {save_path}")

    if show_plot:
        plt.show()
    plt.close()
