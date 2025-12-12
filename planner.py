import heapq
import math
import time
import numpy as np
from scipy.ndimage import distance_transform_edt
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
        self.model = robot_model
        self.start = start_state
        self.goal = goal_state

        self.dist_2d = precompute_holonomic(goal_state,grid)
        self.dist_nh = precompute_nonholonomic(goal_state)
        
        obstacle_mask = np.zeros_like(grid, dtype=bool)
        obstacle_mask[1:-1, 1:-1] = (grid[1:-1, 1:-1] == 1)  # actual obstacles
        # Add border
        obstacle_mask[:, 0] = True
        obstacle_mask[:, -1] = True
        obstacle_mask[0, :] = True
        obstacle_mask[-1, :] = True
        
        self.dist_to_obs = distance_transform_edt(~obstacle_mask)
                
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
        h2 = self.dist_nh(state)

        return max(h1, h2)
    
    def path_search(self,max_nodes=300000):
        start_key = get_state_key(self.start, RESOLUTIONS)

        heapq.heappush(self.open_set, (0 + self.path_maker(self.start), 0, self.start))
        self.g_score[self.start] = 0.0
        self.came_from[self.start] = None

        nodes_expanded = 0

        while self.open_set and nodes_expanded < max_nodes:
            _, current_g, current = heapq.heappop(self.open_set)
            nodes_expanded += 1
            if nodes_expanded % 1000 == 0:
                print(f"  Expanded {nodes_expanded} nodes...")
            self.explored_nodes.append(current)

            if self.goal_reached(current):
                print(f"Goal reached after {nodes_expanded} nodes expanded.")
                return self.path_reconstruct(current), self.explored_nodes
            
            key = get_state_key(current, RESOLUTIONS)
            if key in self.closed and current_g >= self.closed[key]:
                continue
            self.closed[key] = current_g

            steering_angles_deg = [-30, -24, -18, -12, -6, 0, 6, 12, 18, 24, 30]

            for direction in [1.0, -1.0]:
                rev_penalty = REVERSE_PENALTY if direction < 0 else 1.0
                forward_bias = 0.0 if direction > 0 else 0.5
                rev_penalty += forward_bias

                for steer in steering_angles_deg:
                    kappa = 0.0 if abs(steer) < 1e-3 else math.tan(math.radians(steer)) / self.model.wheelbase
                    successor = current.update(MOTION_STEP * direction, kappa)
                    
                    if is_collision_along_arc(current, successor, self.model, self.grid):
                        continue
                        
                    g = current_g + MOTION_STEP * rev_penalty + self.obstacle_avoidance(successor)
                    
                    if g < self.g_score.get(successor, float('inf')):
                        self.g_score[successor] = g
                        f = g + self.path_maker(successor)
                        heapq.heappush(self.open_set, (f, g, successor))
                        self.came_from[successor] = current
        print("Search Failed")
        return None, self.explored_nodes
            
    def path_reconstruct(self,final):
        path = []
        current = final
        while current is not None:
            path.append(current)
            current = self.came_from.get(current)
        return path[::-1]
    
    # Cost to traverse the path considering obstacles
    def obstacle_avoidance(self,state):
        ix, iy = int(state.x), int(state.y)
        # Bounds check
        if not (0 <= iy < self.grid.shape[0] and 0 <= ix < self.grid.shape[1]):
            # Out of bounds
            return 100.0
        # Distance to nearest obstacle
        d = self.dist_to_obs[iy, ix]
        # Simple Voronoi-based cost
        return OBSTACLE_WEIGHT / (d + 1)
    
    # Check if the goal is reached
    def goal_reached(self,state):
        # Check differences in position and orientation
        dx = abs(state.x - self.goal.x)
        dy = abs(state.y - self.goal.y)
        dyaw = min(abs(state.yaw - self.goal.yaw) % 360, 360 - abs(state.yaw - self.goal.yaw) % 360)

        # Return True if within thresholds
        return dx <= 0.2 and dy <= 0.2 and dyaw <= 30.0