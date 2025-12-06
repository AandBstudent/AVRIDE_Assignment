import numpy as np
import heapq
import math
from robot import RobotState, RobotModel

RESOLUTIONS = (0.5, 0.5, 5.0, 0.05) # x, y, yaw(degrees), kappa (1/m)

# Holonomic with obstacles, 8-connected grid
def precompute_holonomic(goal_state, grid):
    # Dijkstra's algorithm to compute 2D distance field
    # Initialize distance field
    rows, cols = grid.shape
    # Initialize all distances to infinity
    dist = np.full((rows, cols), np.inf)
    # Set distance at goal to zero
    gx, gy = int(goal_state.x), int(goal_state.y)

    # Dijkstra's algorithm
    # Initialize priority queue with the goal cell
    dist[gy, gx] = 0
    pq = []
    heapq.heappush(pq, (0, gx, gy))
    # Possible 8-connected movements
    dirs = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

    # Dijkstra's algorithm loop
    while pq:
        # Pop the cell with the smallest distance
        d, y, x = heapq.heappop(pq)
        # If this distance is greater than the recorded distance, skip
        if d > dist[y, x]:
            continue
        # Explore neighbors
        for dx, dy in dirs:
            # Compute neighbor coordinates
            ny, nx = y + dy, x + dx
            # Check bounds and obstacles
            if 0 <= ny < rows and 0 <= nx < cols and grid[ny, nx] == 0:
                # Compute new distance
                nd = d + math.hypot(dx, dy)
                # Update distance if new distance is smaller
                if nd < dist[ny, nx]:
                    # Update distance and push to priority queue
                    dist[ny, nx] = nd
                    heapq.heappush(pq, (nd, nx, ny))
    # Return the computed distance field
    return dist

# Non-holonomic without obstacles, analytical cost
def precompute_nonholonomic(goal_state):
    def nh_cost(state):
        dx = state.x - goal_state.x
        dy = state.y - goal_state.y
        dist = math.hypot(dx, dy)

        delta = abs(state.yaw - goal_state.yaw) % 360
        yaw_err = min(delta, 360 - delta)

        kappa_penalty = 10 * abs(state.kappa)

        return dist + 0.5 * yaw_err + kappa_penalty    
    return nh_cost