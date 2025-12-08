import heapq
import math
import time
import numpy as np
from collections import defaultdict
from robot import RobotModel, RobotState, compute_robot_polygon, is_collision_along_arc
from utils import precompute_holonomic, precompute_nonholonomic, get_state_key

# Configuration parameters
GRID_RESOLUTION = 1.0  # meters per cell
RESOLUTIONS = (0.5, 0.5, 5.0, 0.05) # x, y, yaw(degrees), kappa (1/m)
MOTION_STEP = 0.5  # meters per expansion
MAX_KAPPA_RATE = 0.12 # maximum change in kappa per step (steering actuator limit)
REVERSE_PENALTY = 2.0
OBSTACLE_WEIGHT = 3.0

class HybridAStar:
    def __init__(self,grid,robot_model,start_state,goal_state):
        self.grid = grid
        self.robot_model = robot_model
        self.start = start_state
        self.goal = goal_state

        self.dist_2d = precompute_holonomic(goal_state,grid)
        self.dist_nh = precompute_nonholonomic(goal_state)
                
        self.open_set = []
        self.closed = {}
        self.came_from = {}
        self.g_score = {}
        self.explored_nodes = []

    # Combine holonomic and non-holonomic heuristics for path cost estimation
    def path_maker(self,state):
        # Holonomic heuristic with obstacles
        ix, iy = int(state.x), int(state.y)
        # Bounds check
        if not (0 <= iy < self.dist_2d.shape[0] and 0 <= ix < self.dist_2d.shape[1]):
            # Out of bounds
            return float('inf')
        h1 = self.dist_2d[iy, ix]
        # Non-holonomic heuristic with obstacles
        key = get_state_key(state, RESOLUTIONS)
        h2 = self.dist_nh.get(key, 100.0)

        return max(h1, h2)
    
    # Cost to traverse the path considering obstacles
    def obstacle_avoidance(self,state):
        ix, iy = int(state.x), int(state.y)
        # Bounds check
        if not (0 <= iy < self.grid.shape[0] and 0 <= ix < self.grid.shape[1]):
            # Out of bounds
            return 100.0
        # Distance to nearest obstacle
        d = self.dist_2d[iy, ix]
        # Simple Voronoi-based cost
        return OBSTACLE_WEIGHT / (d + 1)
    
    # Check if the goal is reached
    def goal_reached(self,state):
        # Check differences in position and orientation
        dx = abs(state.x - self.goal.x)
        dy = abs(state.y - self.goal.y)
        dyaw = min(abs(state.yaw - self.goal.yaw) % 360, 360 - abs(state.yaw - self.goal.yaw) % 360)

        # Return True if within thresholds
        return dx <= 0.8 and dy <= 0.8 and dyaw <= 10.0