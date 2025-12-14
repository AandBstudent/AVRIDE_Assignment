# AVRIDE_Assignment
Technical Take Home Assignment given by AVRIDE

Goals to Achieve

1. State representation and hashing

    Describe how you will represent the state: (x, y, yaw, curvature).
    
    Design buckets / a hash function for this state space.

2. Occupancy grid and collision checking

    Generate some occupancy grid (for example, a 2D grid with free and occupied cells).
    
    Propose a method to check for collisions with this grid efficiently.
    
    Assume the robot shape is a rectangle.

3. Heuristic and hybrid search

    Propose a heuristic that at least approximately accounts for collisions with the occupancy grid.
    
    How can we account for the fact that the trajectory curvature cannot change instantaneously?
    
    Implement a search algorithm (Hybrid A*) that uses your heuristic and avoids collisions with the grid.
