import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

class RobotState:
    def __init__(self, x, y, yaw, kappa):
        self.x = x  # X position in meters
        self.y = y  # Y position in meters
        self.yaw = yaw  # Orientation in degrees
        self.kappa = kappa  # Curvature

class RobotModel:
    def __init__(self, length, width, wheelbase, max_steer_deg=35.0):
        self.length = float(length)  # Length of the robot in meters
        self.width = float(width)    # Width of the robot in meters
        self.wheelbase = float(wheelbase)  # Wheelbase in meters
        self.max_steer_deg = float(max_steer_deg)  # Maximum steering angle in degrees
        self.max_steer_rad = np.deg2rad(self.max_steer_deg)  # Max steering angle in radians

        self.max_curvature = np.tan(self.max_steer_rad) / self.wheelbase  # Max curvature
    
    def __repr__(self):
        return (f"RobotModel(length={self.length}, width={self.width}, "
                f"wheelbase={self.wheelbase}, max_steer_rad={self.max_steer_rad:.3f}, "
                f"max_curvature={self.max_curvature:.4f})")

def draw_robot(state,model):
    L = model.length
    W = model.width

    body = np.array([
        [+L/2, +W/2], #front-left
        [+L/2, -W/2], #front-right
        [-L/2, -W/2], # rear-right
        [-L/2, +W/2], # rear-left
    ])

    c,s = np.cos(np.deg2rad(state.yaw)), np.sin(np.deg2rad(state.yaw))
    R = np.array([[c, -s],
                  [s,  c]])
    body_world = (R @ body.T).T
    body_world[:,0] += state.x
    body_world[:,1] += state.y

    return body_world

rows, cols = 20, 20
grid = np.random.choice([0, 1], size=(rows, cols), p=[0.75, 0.25])

state = RobotState(x=10, y=10, yaw=45, kappa=0.1)
model = RobotModel(length=0.625, width=0.3, wheelbase=0.425)
robot_body = draw_robot(state, model)

robot_place = Polygon(robot_body, closed=True, color='blue', facecolor="none",linewidth=2)

fig, ax = plt.subplots(figsize=(6, 6))

# Map the array to [0, cols] x [0, rows]
ax.imshow(
    grid,
    cmap="gray_r",
    origin="lower",
    extent=[0, cols, 0, rows]  # left, right, bottom, top
)

ax.set_title("2D Occupancy Grid")
ax.set_xlabel("X (meters)")
ax.set_ylabel("Y (meters)")

# Integer ticks from 0 to 20
ax.set_xticks(np.arange(0, cols + 1, 1))
ax.set_yticks(np.arange(0, rows + 1, 1))

# Grid lines exactly on cell boundaries
ax.grid(which="both", color="lightgray", linewidth=0.5)

ax.set_xlim(0, cols)
ax.set_ylim(0, rows)
ax.set_aspect("equal")
ax.add_patch(robot_place)

print(model)

plt.show()