import numpy as np
import argparse
from robot import RobotModel, RobotState
from robot import get_goal_direction
from planner import HybridAStar
from viz import visualize_and_mcap
import time

def load_grid(rows=5, cols=5,num_obstacles=5):
    # grid = np.zeros((rows, cols), dtype=np.uint8)
    # Add obstacles
    # Test path finding around obstacles
    grid = np.zeros((rows, cols), dtype=int)

    # Create more obstacles and place them randomly
    for _ in range(num_obstacles+1):
        x, y = np.random.randint(0, 5, 2)
        grid[x, y] = 1  # obstacle

    # Find a free cell to place the robot
    free_indices = np.argwhere(grid == 0)
    if free_indices.size > 0:
        selected = free_indices[np.random.randint(len(free_indices))]
        i, j = selected
        # Center the robot in the cell
        x_coord = j + 0.5
        y_coord = i + 0.5

    # Find a free cell to place the goal, not the same as the robot's cell
    free_indices = np.argwhere(grid == 0)
    free_indices = free_indices[~((free_indices[:, 0] == i) & (free_indices[:, 1] == j))]
    selected = free_indices[np.random.randint(len(free_indices))]
    i, j = selected
    x_coord_goal = j + 0.5
    y_coord_goal = i + 0.5    

    return grid, x_coord, y_coord, x_coord_goal, y_coord_goal

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=5)
    parser.add_argument('--cols', type=int, default=5)
    parser.add_argument('--num_obs', type=int, default=5)
    args = parser.parse_args()

    grid, x_coord, y_coord, x_coord_goal, y_coord_goal = load_grid(rows=args.rows,cols=args.cols,num_obstacles=args.num_obs)
    # Robot model
    robot_model = RobotModel()

    start_angle = get_goal_direction(x_coord, y_coord, x_coord_goal, y_coord_goal)

    start_state = RobotState(x_coord, y_coord, 0, 0)
    goal_state = RobotState(x_coord_goal, y_coord_goal, 0, 0)

    planner = HybridAStar(grid, robot_model, start_state, goal_state)

    start_time = time.time()
    path, explored = planner.path_search()
    end_time = time.time()

    print(f"Planning finished in {end_time - start_time:.2f} seconds | Nodes explored: {len(explored)}")
    if path:
        print(f"Path length: {len(path)} | Path cost: {planner.g_score[path[-1]]:.2f}")
    else:
        print("No path found")

    visualize_and_mcap(grid, path, explored, robot_model, start_state, goal_state)

if __name__ == "__main__":
    main()