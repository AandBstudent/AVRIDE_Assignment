import numpy as np
from robot import RobotModel, RobotState
from planner import HybridAStar
from viz import visualize_and_mcap
import time

def load_grid():
    grid = np.zeros((50, 50), dtype=np.uint8)

    # Add obstacles
    grid[25, 35] = 1
    grid[35, 12] = 1
    grid[10, 20] = 1

    # Boundary walls
    grid[:1, :] = 1   # top wall
    grid[-1:, :] = 1  # bottom wall
    grid[:, :1] = 1   # left wall
    grid[:, -1:] = 1  # right wall

    return grid

def main():
    grid = load_grid()

    # Test explored nodes
    explored = [RobotState(i, j, 0, 0) for i in range(0, 20, 2) for j in range(0, 20, 2)]

    # Test path
    path = [RobotState(i, i, 45, 0.1) for i in range(5, 25)]

    # Robot model
    robot_model = RobotModel()

    visualize_and_mcap(grid, path, explored, robot_model, "hybrid_astar_single_tick_main.mcap")

if __name__ == "__main__":
    main()