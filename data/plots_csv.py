import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load CSV data
csv_file = "data_flops_traces.csv"  # Replace with your actual file name
df = pd.read_csv(csv_file)

# Extract relevant data
x_labels = df["flops_log10"].astype(float).round(decimals=3).astype(str)  # Rounded FLOPs on x-axis
timing_methods = ["custom", "np_mm", "torch"]  # Bars only for these
numpy_times = df["numpy"].values  # Numpy values for text labels
times = df[timing_methods].values  # Extract only bar values

# Define bar width and positions
x = np.arange(len(df))  # X locations for FLOPs values
bar_width = 0.2  # Width of each bar in the group

# Define colors and hatch patterns for accessibility
colors = ["blue", "green", "purple"]
hatch_patterns = ["//", "xx", ".."]  # Different textures

# Create the plot
plt.figure(figsize=(10, 5))

# Plot bars for non-numpy methods
for i, (method, hatch) in enumerate(zip(timing_methods, hatch_patterns)):
    plt.bar(
        x + i * bar_width, times[:, i], width=bar_width,
        label=method, color=colors[i], hatch=hatch, edgecolor="black"
    )

# Add numpy results as text above the bars

fixed_offset = 0.02 * max(times.flatten())  # 2% of max value in dataset

for i in range(len(df)):
    plt.text(
        x[i] + bar_width * 1.5,  # Position in the center of bars
        max(times[i]) + fixed_offset,  # Slightly above the highest bar
        f"{numpy_times[i]:.4f}s",  # Format text
        color="red", fontsize=10, ha="center", fontweight="bold"
    )
# Labels and title
plt.scatter([], [], color="red", label="numpy (text)")
plt.xlabel("flops_log10")
plt.ylabel("Time (seconds)")
plt.title("Execution Times for aabcd,adeef->bcf")
#floats unopt abcd,adef->cbef
# floats opt aabcd,adeef->dcf
# batch abcd,adef->dbef
# traces aabcd,adeef->bcf
plt.xticks(x + bar_width, x_labels, rotation=90)  # Rotate x-ticks to ensure readability
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Adjust layout to prevent overlap and ensure readability
plt.tight_layout()

# Show the plot
plt.show()
