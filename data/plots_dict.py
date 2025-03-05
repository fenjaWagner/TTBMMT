import matplotlib.pyplot as plt
import numpy as np

# Sample dictionary with multiple instances
data = {
    "mc_2022_167": {
        "custom": 34.73985505104065,
        "flops": 6.8712784027149,
        "size": 6.0,
        "numpy": 2.3334221839904785,
        "np_mm": 9.573548316955566,
        "torch": None
    },
    "mc_2022_168": {
        "custom": 28.5,
        "flops": 5.9,
        "size": 6.2,
        "numpy": 3.1,
        "np_mm": 8.2,
        "torch": 4.3
    },
    "mc_2022_169": {
        "custom": 32.1,
        "flops": 6.2,
        "size": 5.8,
        "numpy": 2.8,
        "np_mm": 9.1,
        "torch": 4.9
    }
}

# Extract instance names and timing methods
instance_names = list(data.keys())
timing_methods = ["custom", "numpy", "np_mm", "torch"]

# Convert data into a 2D list (rows = instances, columns = methods)
times = [
    [data[instance].get(method, 0) for method in timing_methods]
    for instance in instance_names
]

# Replace None values with NaN to avoid plotting errors
times = np.array(times, dtype=np.float32)
times[np.isnan(times)] = 0  # Optional: Set NaN values to 0

# Bar width and positions
x = np.arange(len(instance_names))  # X locations for instances
bar_width = 0.2  # Adjust for spacing

# Colors and hatch patterns for each method
colors = ["blue", "green", "red", "purple"]
hatch_patterns = ["//", "xx", "--", ".."]  # Different hatch patterns

# Create the plot
plt.figure(figsize=(10, 5))

# Plot each method as a separate bar group with unique colors and hatch patterns
for i, (method, hatch) in enumerate(zip(timing_methods, hatch_patterns)):
    plt.bar(
        x + i * bar_width, times[:, i], width=bar_width,
        label=method, color=colors[i], hatch=hatch, edgecolor="black"
    )

# Labels and title
plt.xlabel("Instance Name")
plt.ylabel("Time (seconds)")
plt.title("Execution Times for Different Instances (Accessible)")
plt.xticks(x + bar_width * (len(timing_methods) / 2), instance_names, rotation=45)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the plot
plt.show()
