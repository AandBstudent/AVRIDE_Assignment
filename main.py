import numpy as np
from robot import RobotModel, RobotState
from planner import HybridAStar
from viz import visualize_and_mcap
import time

def load_grid(rows=50, cols=50):
    # grid = np.zeros((rows, cols), dtype=np.uint8)
    # Add obstacles
    grid = np.random.choice([0, 1], size=(rows, cols), p=[0.75, 0.25])
    # Find a free cell to place the robot
    free_indices = np.argwhere(grid == 0)
    if free_indices.size > 0:
        selected = free_indices[np.random.randint(len(free_indices))]
        i, j = selected
        # Center the robot in the cell
        x_coord = j + 0.5
        y_coord = i + 0.5

    # Boundary walls
    grid[:1, :] = 1   # top wall
    grid[-1:, :] = 1  # bottom wall
    grid[:, :1] = 1   # left wall
    grid[:, -1:] = 1  # right wall

    return grid, x_coord, y_coord

def main():
    grid, x_coord, y_coord = load_grid()

    # Test explored nodes
    explored = [RobotState(i, j, 0, 0) for i in range(0, 20, 2) for j in range(0, 20, 2)]

    # Test path
    path = [RobotState(x_coord+ i, y_coord + i, 45, 0.1) for i in range(5, 25)]

    # Robot model
    robot_model = RobotModel(length=0.85, width=0.45)

    visualize_and_mcap(grid, path, explored, robot_model, "hybrid_astar_single_tick_main.mcap")

if __name__ == "__main__":
    main()