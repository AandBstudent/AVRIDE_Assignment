import numpy as np
import matplotlib.pyplot as plt

rows, cols = 20, 20
grid = np.random.choice([0, 1], size=(rows, cols), p=[0.75, 0.25])

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

plt.show()