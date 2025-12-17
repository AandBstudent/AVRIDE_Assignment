# Hybrid A* Path Planning (Hybrid A-Star)

This repository contains a **Hybrid A\*** planner for 2D occupancy-grid path planning with a non-holonomic robot state:
\(x, y, yaw, kappa\).

The implementation supports:
- Continuous-state motion primitives (bicycle-model integration)
- Discretized state bucketing for closed-set pruning
- Obstacle-aware heuristic guidance (holonomic + non-holonomic)
- Rectangle footprint collision checking along motion arcs
- Visualization (PNG) and optional **MCAP** export for Foxglove

---

## Project Structure

- `main.py` — CLI entrypoint; creates a random occupancy grid, runs planning, and visualizes results.
- `planner.py` — Hybrid A\* search (open/closed sets, successors, heuristic, obstacle cost).
- `robot.py` — Robot model/state, bicycle-model propagation, and collision checking utilities.
- `utils.py` — Heuristic precomputation and state discretization keying.
- `viz.py` — Matplotlib visualization + optional MCAP export (Foxglove).

---

## Requirements

- **Python:** 3.10+ recommended
- **OS:** Windows / macOS / Linux
- **Hardware:** CPU-only (single-threaded)

> Note: MCAP export requires Foxglove SDK dependencies. If you only need the PNG plot, you can comment out the MCAP section in `viz.py`.

---

## Quickstart

### 1) Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Run the planner
```bash
python main.py --rows 5 --cols 5 --num_obs 5
```

Arguments:
- `--rows` grid rows (default: 5)
- `--cols` grid cols (default: 5)
- `--num_obs` number of randomly placed obstacles (default: 5)

---

## Outputs

After a run, you should see:
- Console timing + node expansion count
- A plotted figure window
- `hybrid_astar_result_new.png` (saved image)
- `hybrid_astar_single_tick.mcap` (MCAP log; default name)

---

## How It Works (High Level)

### State and bucketing
The planner searches in continuous state space \((x, y, yaw, kappa)\) but uses discretized buckets for closed-set checks:
- \(x, y\) bucketed at 0.5 m
- \(yaw\) bucketed at 5 degrees
- \(kappa\) bucketed at 0.05 1/m

This reduces redundant expansions while retaining continuous motion primitives.

### Successor generation
Each expansion evaluates a fixed set of steering angles (both forward and reverse). Motion is propagated with a bicycle model over a fixed step distance.

### Collision checking
Each candidate motion is checked by sampling intermediate poses along the arc, evaluating the robot's rectangular footprint against occupied grid cells.

### Heuristic
The heuristic combines:
- **Holonomic distance-to-go** in the grid (2D obstacle-aware Dijkstra field)
- **Non-holonomic penalty** (distance + yaw mismatch + curvature penalty)

The planner uses the maximum of the two to guide search effectively.

---

## Tuning Tips

You can tune performance/quality tradeoffs via constants in `planner.py`:
- `MOTION_STEP` — smaller = smoother but more nodes
- `RESOLUTIONS` — coarser = faster but less precise pruning
- `steering_angles_deg` — fewer angles = faster but can reduce solution quality
- `REVERSE_PENALTY` — higher = prefer forward motion
- `OBSTACLE_WEIGHT` — higher = keep more clearance

The search call supports:
- `min_nodes` (default 100,000) and `max_nodes` (default 350,000)

---

## Troubleshooting

- **`ModuleNotFoundError: scipy`**  
  Ensure you installed requirements: `pip install -r requirements.txt`

- **MCAP export issues** (Foxglove not installed / import errors)  
  Install requirements (includes `foxglove-sdk` + `mcap`), or comment out the MCAP block in `viz.py`.

- **No path found**  
  Random grids can be unsolvable depending on obstacle placement. Try fewer obstacles or a larger grid:
  ```bash
  python main.py --rows 20 --cols 20 --num_obs 40
  ```

---

## Notes on Reproducibility

The grid in `main.py` is randomized. For comparable benchmarking across runs, consider adding a fixed seed:
```python
np.random.seed(0)
```

---

## License

This code is provided for educational and evaluation purposes as part of a technical take-home assignment.
