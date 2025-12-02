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
    
    # String representation of the robot state
    def __repr__(self):
        return f"RobotState(x={self.x:.2f}, y={self.y:.2f}, yaw={self.yaw:.1f}, kappa={self.kappa:.3f})"

class RobotModel:
    def __init__(self, length=0.625, width=0.3, wheelbase=0.425, max_steer_deg=35.0):
        self.length = float(length)  # Length of the robot in meters
        self.width = float(width)    # Width of the robot in meters
        self.wheelbase = float(wheelbase)  # Wheelbase in meters
        self.max_steer_rad = math.radians(self.max_steer_deg)  # Max steering angle in radians
        self.max_curvature = math.tan(self.max_steer_rad) / self.wheelbase  # Max curvature
    
    def __repr__(self):
        return (f"RobotModel(length={self.length}, width={self.width}, "
                f"wheelbase={self.wheelbase}, max_steer_rad={self.max_steer_rad:.3f}, "
                f"max_curvature={self.max_curvature:.4f})")