import matplotlib.pyplot as plt
import numpy as np
import json
import time
from mcap.writer import Writer
from robot import compute_robot_polygon

def visualize_and_mcap(grid, path, explored, robot_model, start, goal, file_name="hybrid_astar_single_tick.mcap"):
    # Visualization of the grid, path, and explored nodes
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid, cmap="gray_r", origin="lower",
              extent=[0, grid.shape[1], 0, grid.shape[0]], aspect='equal')
    ax.grid(True,color='gray', linewidth=0.5, alpha=0.5)

    # Explored nodes
    if explored:
        ex = [s.x for s in explored]
        ey = [s.y for s in explored]
        ax.scatter(ex, ey, c='pink', s=4, label='Explored Nodes', alpha=1.0)
    
    # Final path and robot footprints
    if path:
        px = [s.x for s in path]
        py = [s.y for s in path]
        ax.plot(px,py, 'b-', linewidth=4, label='Final Path')
        for i, s in enumerate(path[::1]):  # Every state
            poly = compute_robot_polygon(s, robot_model)
            ax.add_patch(plt.Polygon(poly, color='gray', alpha=1, zorder=3))
    
    # Mark start and goal
    if start:
        ax.scatter(start.x, start.y, c='green', s=100, marker='o', label='Start', zorder=4)
    if goal:
        ax.scatter(goal.x, goal.y, c='red', s=100, marker='x', label='Goal', zorder=4)
    
    ax.legend()
    ax.set_title("Hybrid A* Path Planning Visualization")
    plt.tight_layout()
    plt.savefig("hybrid_astar_result_new.png", dpi=300)
    plt.show()

    # https://github.com/foxglove/mcap/tree/main/python/examples/jsonschema