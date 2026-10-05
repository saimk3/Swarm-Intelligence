"""
Configuration and Problem Instance Seeding Module.

This module manages student roll number parsing, deterministic seed generation,
grid dimension specifications, and PSO hyperparameters.
"""

from typing import Union
import hashlib

# Student Information
STUDENT_NAME: str = "Muhammad Saim Khan"
STUDENT_ROLL_NO: str = "01-136232-073"


def roll_number_to_seed(roll_no: Union[str, int]) -> int:
    """
    Transforms a student roll number into an integer random seed.
    Uses the student roll number ID (73 from '01-136232-073') as seed.
    """
    if isinstance(roll_no, int):
        return abs(roll_no)
    
    parts = str(roll_no).strip().split("-")
    if len(parts) > 1 and parts[-1].isdigit():
        return int(parts[-1])
    
    digits = "".join(ch for ch in str(roll_no) if ch.isdigit())
    if digits:
        return int(digits)
    
    return 73


# Default Problem Instance Parameters (Roll Number: 01-136232-073 -> Seed: 73)
DEFAULT_SEED: int = 73
DEFAULT_GRID_SIZE: int = 20
DEFAULT_NUM_OBSTACLES: int = 55  # 55 obstacles as specified in student handwritten flowchart

# PSO Hyperparameters
DEFAULT_NUM_PARTICLES: int = 80
DEFAULT_NUM_ITERATIONS: int = 150
DEFAULT_NUM_WAYPOINTS: int = 5  # Number of intermediate path waypoints

# PSO Inertia and Acceleration Coefficients
W_MAX: float = 0.9      # Initial inertia weight (promotes exploration)
W_MIN: float = 0.4      # Final inertia weight (promotes exploitation)
C1: float = 1.5         # Cognitive (personal best) acceleration coefficient
C2: float = 1.5         # Social (global best) acceleration coefficient

# Fitness Evaluation Penalties
COLLISION_PENALTY: float = 1000.0  # Heavy penalty for hitting an obstacle or grid border
SMOOTHNESS_WEIGHT: float = 0.5     # Weight for penalizing sharp turning angles
