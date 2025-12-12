import numpy as np
import math

# Define robot state classes
class RobotState:
    def __init__(self, x, y, yaw, kappa=0.0):
        self.x = float(x)  # X position in meters
        self.y = float(y)  # Y position in meters
        self.yaw = float(yaw) % 360  # Orientation in degrees, normalized to [0, 360)
        self.kappa = float(kappa)  # Curvature
    
    def update(self,ds, new_kappa):
        # Bicycle model integration over arc segment ds with constant kappa
        if abs(new_kappa) < 1e-6:
            # Straight line motion
            dx = ds * math.cos(math.radians(self.yaw))
            dy = ds * math.sin(math.radians(self.yaw))
            # Return new state
            return RobotState(self.x + dx, self.y + dy, self.yaw, new_kappa)
        else:
            # Turning radius
            R = 1.0 / new_kappa
            # Change in heading
            dtheta = ds / R
            # Calculate new position
            dx = R * (math.sin(math.radians(self.yaw) + dtheta) - math.sin(math.radians(self.yaw)))
            dy = R * (math.cos(math.radians(self.yaw)) - math.cos(math.radians(self.yaw) + dtheta))
            # Calulate new yaw after turn
            new_yaw = (self.yaw + math.degrees(dtheta)) % 360 # Normalize yaw
            # Return new state
            return RobotState(self.x + dx, self.y + dy, new_yaw, new_kappa)

    # Method makes RobotState hashable
    def __hash__(self):
        # Hash based on rounded values to avoid floating point precision issues
        return hash((round(self.x, 3), round(self.y, 3), round(self.yaw, 1), round(self.kappa, 4)))
    
    # Equality check based on hash
    def __eq__(self, other):
        # Verify that other is RobotState and compare hashes
        return isinstance(other, RobotState) and hash(self) == hash(other)
    
    # Less than for heap comparisons
    def __lt__(self, other):
        if not isinstance(other, RobotState):
            return NotImplemented
        return (self.x, self.y, self.yaw, self.kappa) < (other.x, other.y, other.yaw, other.kappa)
    
    # String representation of the robot state
    def __repr__(self):
        return f"RobotState(x={self.x:.2f}, y={self.y:.2f}, yaw={self.yaw:.1f}, kappa={self.kappa:.3f})"

class RobotModel:
    def __init__(self, length=0.3125, width=0.15, wheelbase=0.2125, max_steer_deg=35.0):
        self.length = float(length)  # Length of the robot in meters
        self.width = float(width)    # Width of the robot in meters
        self.wheelbase = float(wheelbase)  # Wheelbase in meters
        self.max_steer_rad = math.radians(max_steer_deg)  # Max steering angle in radians
        self.max_curvature = math.tan(self.max_steer_rad) / self.wheelbase  # Max curvature
    
    def __repr__(self):
        return (f"RobotModel(length={self.length}, width={self.width}, "
                f"wheelbase={self.wheelbase}, max_steer_rad={self.max_steer_rad:.3f}, "
                f"max_curvature={self.max_curvature:.4f})")
    
def compute_robot_polygon(state: RobotState, model: RobotModel):
    # Returns 4 corners of the robot polygon in world coordinates
    L, W = model.length, model.width
    # Define robot body corners in local frame
    body_local = np.array([
        [+L/2, +W/2], # front-left
        [+L/2, -W/2], # front-right
        [-L/2, -W/2], # rear-right
        [-L/2, +W/2], # rear-left
    ])
    # Rotation matrix from local to world frame
    theta = math.radians(state.yaw)
    # 2D rotation matrix
    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta),  math.cos(theta)]])
    # Transform body corners to world frame
    body_world = body_local @ R.T
    # Translate to world position
    body_world[:,0] += state.x
    body_world[:,1] += state.y
    # Return the polygon corners in world coordinates
    return body_world

# Function to check collision along an arc from start to end state
def is_collision_along_arc(start: RobotState, end: RobotState, model: RobotModel, grid):
    # Sample 6 intermediate poses along the arc and check for collisions
    num_samples = 6
    for i in np.linspace(0.1,0.9,num_samples):
        # Linearly interpolate kappa (conservative)
        kappa_interp = start.kappa + i * (end.kappa - start.kappa)
        ds_interp = i * 0.5 # assuming MOTION_STEP = 0.5
        intermediate = start.update(ds_interp, kappa_interp)
        if _is_collision_single(intermediate,model,grid):
            return True
    return False

# Helper function to check collision at a single robot state
def _is_collision_single(state: RobotState, model: RobotModel, grid):
    # Check if robot at given state collides with occupied cells in grid
    poly = compute_robot_polygon(state, model)
    # Axis-aligned bounding box of the robot polygon
    min_x, max_x = poly[:,0].min(), poly[:,0].max()
    min_y, max_y = poly[:,1].min(), poly[:,1].max()

    # Check grid cells overlapping with AABB
    for i in range(int(math.floor(min_y)), int(math.ceil(max_y))):
        for j in range(int(math.floor(min_x)), int(math.ceil(max_x))):
            # Check grid bounds
            if not (0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]):
                # Out of bounds implies collision
                return True
            if grid[i,j] == 1: # occupied cell
                # Quick Axis Aligned Bounding Box check
                if (j <= max_x and j+1 >= min_x and
                    i <= max_y and i+1 >= min_y):
                    return True # Collision detected
    return False