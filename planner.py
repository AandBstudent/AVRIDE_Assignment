import heapq
import math
import time
import numpy as np
from collections import defaultdict
from robot import RobotModel, RobotState, compute_robot_polygon, is_collision_along_arc
from utils import precompute_holonomic, precompute_nonholonomic

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
        self.start_state = start_state
        self.goal_state = goal_state

        self.dist_2d = precompute_holonomic(goal_state,grid)
        self.dist_rs = precompute_nonholonomic(goal_state)

                
        self.open_set = []
        self.closed = {}
        self.came_from = {}
        self.g_score = {}
        self.explored_nodes = []
        