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
    Deterministically transforms a student roll number into an integer random seed.
    
    If roll_no contains digits (e.g. '01-136232-073'), it extracts the numeric
    sequence ('01136232073' -> 1136232073).
    Otherwise, it hashes the string deterministically.
    """
    if isinstance(roll_no, int):
        return abs(roll_no)
    
    digits = "".join(ch for ch in str(roll_no) if ch.isdigit())
    if digits:
        return int(digits)
    
    # Deterministic fallback hashing for non-digit inputs
    return int(hashlib.sha256(str(roll_no).encode("utf-8")).hexdigest()[:8], 16)


# Default Problem Instance Parameters
DEFAULT_SEED: int = roll_number_to_seed(STUDENT_ROLL_NO)
DEFAULT_GRID_SIZE: int = 20
DEFAULT_OBSTACLE_RATIO: float = 0.20  # 20% obstacle coverage (80 obstacles in 20x20)

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
