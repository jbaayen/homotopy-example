import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import ListedColormap, Normalize

from example import results, times

theta_values = list(results.keys())
variable_names = results[theta_values[0]].keys()

# Use greyscale style for plots
plt.style.use("grayscale")

file_type = "pdf"

# Generate Aggregated Plot
n_subplots = 2
width = 4
height = 4
time_hrs = times / 3600
fig, axarr = plt.subplots(n_subplots, sharex=True, figsize=(width, height))
theta = 1.0
vars_to_plot = "H_1", "H_4", "H_7", "H_10", "Q_1", "Q_4", "Q_7", "Q_11"
for var in vars_to_plot:
    axarr[0 if var.startswith("Q") else 1].step(
        time_hrs, results[theta][var], where="mid", label=f"{var}"
    )

axarr[0].set_ylabel("Flow Rate [m³/s]")
axarr[1].set_ylabel("Water Level [m]")
axarr[1].set_xlabel("Time [hrs]")

# Shrink margins
plt.autoscale(enable=True, axis="x", tight=True)
fig.tight_layout()

# Shrink each axis and put a legend to the right of the axis
for i in range(n_subplots):
    box = axarr[i].get_position()
    axarr[i].set_position([box.x0, box.y0, box.width * 0.85, box.height])
    axarr[i].legend(
        loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, prop={"size": 8}
    )

# Output Plot
plt.savefig(f"final_results.{file_type}")

# Generate Individual Deformation Plots
width = 4
height = 2
lightest_grey = 0.8
for var in variable_names:
    fig, ax = plt.subplots(1, figsize=(width, height))
    ax.set_title(var)
    for theta in theta_values:
        ax.step(
            time_hrs,
            results[theta][var],
            where="mid",
            color=str(lightest_grey - theta * lightest_grey),
        )

    # Shrink margins
    ax.set_ylim((90, 310) if var.startswith("Q") else (-2, 2))
    ax.set_ylabel("Flow Rate [m³/s]" if var.startswith("Q") else "Water Level [m]")
    ax.set_xlabel("Time [hrs]")

    fig.tight_layout()

    # Output Plot
    plt.savefig(f"{var}.{file_type}")

# Generate a Bar Scale Legend
width = 4
height = 2
fig, axarr = plt.subplots(1, 5, figsize=(width, height))

for i, ax in enumerate(axarr):
    if i != 2:
        ax.set_axis_off()

cmap = ListedColormap(np.linspace(lightest_grey, 0.0, 256, dtype=str))
norm = Normalize(vmin=0.0, vmax=1.0)
cb = ColorbarBase(axarr[2], cmap=cmap, norm=norm, orientation="vertical")

plt.savefig(f"colorbar.{file_type}")
