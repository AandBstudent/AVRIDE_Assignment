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
    
    # Perform the path search using A* algorithm
    def path_search(self,min_nodes = 100000, max_nodes=350000):
        
        # Initialize the search
        # Push the start state onto the open set
        heapq.heappush(self.open_set, (0 + self.path_maker(self.start), 0, self.start))
        # Initialize the start state's g-score and parent
        self.g_score[self.start] = 0.0
        self.came_from[self.start] = None

        nodes_expanded = 0

        goal_found = False
        goal_state = None

        # Main search loop, until the goal is reached or the open set is exhausted
        while self.open_set and nodes_expanded < max_nodes:
            # Pop the state with the lowest f-score from the open set
            _, current_g, current = heapq.heappop(self.open_set)
            nodes_expanded += 1
            # Print progress every 1000 nodes
            if nodes_expanded % 1000 == 0:
                print(f"  Expanded {nodes_expanded} nodes...")
            # Mark the current state as explored
            self.explored_nodes.append(current)

            # Check if the goal is reached
            if self.goal_reached(current):
                #print(f"Goal reached after {nodes_expanded} nodes expanded.")
                goal_found = True
                goal_state = current
                if nodes_expanded >= min_nodes:
                    print(f"Min nodes reached, returning path of {nodes_expanded} expansion.")
                    # If the goal is reached, reconstruct and return the path
                    return self.path_reconstruct(goal_state), self.explored_nodes
            
            # Skip if the current state is already in the closed set 
            # and its g-score is not better
            key = get_state_key(current, RESOLUTIONS)
            if key in self.closed and current_g >= self.closed[key]:
                continue
            # Add the current state to the closed set
            self.closed[key] = current_g

            # Generate and evaluate successor states
            # Define possible steering angles in degrees
            # Note: The angles are chosen to be symmetric around 0 degrees
            # and cover a reasonable range for steering
            steering_angles_deg = [-40, -32, -24, -16, -8, 0, 8, 16, 24, 32, 40]

            # Generate successors for forward and reverse directions
            # Note: The direction is 1.0 for forward and -1.0 for reverse
            # Note: The reverse penalty is applied to the cost of moving in reverse
            # Note: A forward bias is added to the reverse penalty to encourage forward motion
            for direction in [1.0, -1.0]:
                # Apply reverse penalty if moving in reverse
                rev_penalty = REVERSE_PENALTY if direction < 0 else 1.0
                # Add forward bias to reverse penalty to encourage forward motion
                forward_bias = 0.0 if direction > 0 else 0.5
                rev_penalty += forward_bias

                # Generate successors for each steering angle
                for steer in steering_angles_deg:
                    # Compute kappa from steering angle
                    kappa = 0.0 if abs(steer) < 1e-3 else math.tan(math.radians(steer)) / self.model.wheelbase
                    # Generate successor state
                    successor = current.update(MOTION_STEP * direction, kappa)
                    
                    # Skip if successor state is in collision
                    if is_collision_along_arc(current, successor, self.model, self.grid):
                        continue
                    # Compute cost to reach successor state
                    g = current_g + MOTION_STEP * rev_penalty + self.obstacle_avoidance(successor)
                    
                    # Update the open set with the new successor state 
                    # if it has a better g-score
                    if g < self.g_score.get(successor, float('inf')):
                        # Update g-score and push to open set
                        self.g_score[successor] = g
                        # Compute f-score and push to open set
                        f = g + self.path_maker(successor)
                        heapq.heappush(self.open_set, (f, g, successor))
                        # Update parent of successor state
                        self.came_from[successor] = current
        if goal_found:
            print(f"Goal was found but minimum nodes not reached, returning best path found.")
            return self.path_reconstruct(goal_state), self.explored_nodes
        
        print("Search Failed")
        # If the goal is not reached, return None and the explored nodes
        return None, self.explored_nodes
    
    # Reconstruct the path from the goal to the start
    def path_reconstruct(self,final):
        path = []
        current = final
        # Trace back through the parent pointers
        while current is not None:
            path.append(current)
            current = self.came_from.get(current)
        # Reverse the path to get from start to goal
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
        return dx <= 0.15 and dy <= 0.15 and dyaw <= 180.0