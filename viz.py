import matplotlib.pyplot as plt
import foxglove
from foxglove.channels import SceneUpdateChannel
from foxglove import Context
from foxglove.schemas import SceneUpdate, SceneEntity, CubePrimitive, TriangleListPrimitive, Pose, Vector3, Quaternion, Color, Duration, Point3
from robot import compute_robot_polygon, _is_collision_single
import os

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
        ax.scatter(ex, ey, c='lightblue', s=4, label='Explored Nodes', alpha=1.0)
    
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

    # MCAP file creation
    # https://docs.foxglove.dev/docs/getting-started/python
    # check if the file exists and delete it if necessary
    if os.path.exists(file_name):
        os.remove(file_name)

    with foxglove.open_mcap(file_name) as mcap:
        scene_channel = SceneUpdateChannel("/scene", context=Context().default())

        entities = []

        for i, state in enumerate(explored[::10]):  # Every 10th state
            entities.append(SceneEntity(
                id=f"explored_{i}",
                frame_id="map",
                lifetime=Duration(sec=0, nsec=0),
                cubes=[CubePrimitive(
                    pose=Pose(
                        position=Vector3(x=state.x, y=state.y, z=0.5),
                        orientation=Quaternion(x=0, y=0, z=0, w=1)
                    ),
                    size=Vector3(x=.05, y=.05, z=0.09),
                    color=Color(r=0.0, g=1.0, b=1.0, a=1.0)
                )]
            ))

        # Path of robot footprints
        if path:
            for i, state in enumerate(path):
                poly = compute_robot_polygon(state, robot_model).tolist()[::-1]
                points = [Point3(x=p[0], y=p[1], z=0.67) for p in poly]
                entities.append(SceneEntity(
                    id=f"path_robot_{i}",
                    frame_id="map",
                    lifetime=Duration(sec=0, nsec=0),
                    triangles=[TriangleListPrimitive(
                        points=points,
                        color=Color(r=1.0, g=0.2, b=0.2, a=1.0)
                    )]
                ))
                # Check for collision
                if _is_collision_single(state, robot_model, grid):
                    entities.append(SceneEntity(
                        id=f"collision_{i}",
                        frame_id="map",
                        lifetime=Duration(sec=0, nsec=0),
                        cubes=[CubePrimitive(
                            pose=Pose(
                                position=Vector3(x=state.x, y=state.y, z=1.5),
                                orientation=Quaternion(w=1)
                            ),
                            size=Vector3(x=0.2, y=0.2, z=0.2),
                            color=Color(r=1.0, g=0.0, b=0.0, a=1.0)
                        )]
                    ))
        
        # Start and goal markers
        entities += [
            SceneEntity(
            id="start",
            frame_id="map",
            lifetime=Duration(sec=0, nsec=0),
            cubes=[CubePrimitive(
                pose=Pose(
                    position=Vector3(x=start.x, y=start.y, z=1.0),
                    orientation=Quaternion(w=1)
                ),
                size=Vector3(x=.3, y=.3, z=.6),
                color=Color(r=0.0, g=1.0, b=0.0, a=1.0)
            )]
        ), 
        SceneEntity(
            id="goal",
            frame_id="map",
            lifetime=Duration(sec=0, nsec=0),
            cubes=[CubePrimitive(
                pose=Pose(
                    position=Vector3(x=goal.x, y=goal.y, z=1.0),
                    orientation=Quaternion(w=1)
                ),
                size=Vector3(x=.3, y=.3, z=.6),
                color=Color(r=1.0, g=0.0, b=0.0, a=1.0)
            )]
        )]
                
        scene = SceneUpdate(entities=entities)
        scene_channel.log(scene)
                

    print(f"MCAP file '{file_name}' created successfully.")