import matplotlib.pyplot as plt
import numpy as np

# The given data dictionary
data = {
    "mc_2022_167": {"custom": 33.79396629333496, "flops": 6.8712784027149, "size": 6.0, "numpy": 2.380446434020996, "np_mm": 9.500880002975464, "torch": None},
    "mc_2020_017": {"custom": 7.168416976928711, "flops": 9.732320076076736, "size": 22.0, "numpy": 57.27097535133362, "np_mm": 6.858346700668335, "torch": None},
    "wmc_2023_141": {"custom": 33.30284571647644, "flops": 10.483754772311697, "size": 24.0, "numpy": 493.5835590362549, "np_mm": 18.252779483795166, "torch": 14.50853443145752},
    "lm_batch_likelihood_sentence_4_12d": {"custom": 14.271461248397827, "flops": 11.071312602234704, "size": 25.231633706102937, "numpy": 167.6301896572113, "np_mm": 3.216212272644043, "torch": 4.071931600570679},
    "wmc_2023_152": {"custom": 2.26204252243042, "flops": 6.445754390267859, "size": 14.0, "numpy": 0.20770716667175293, "np_mm": 0.545893669128418, "torch": 0.890263557434082},
    "mc_2023_002": {"custom": 1.5341520309448242, "flops": 7.457456230927699, "size": 17.0, "numpy": 0.6097269058227539, "np_mm": 0.45136547088623047, "torch": 0.6765711307525635},
    "mc_2020_arjun_046": {"custom": 2.3599014282226562, "flops": 9.306740271382463, "size": 23.0, "numpy": 69.61855173110962, "np_mm": 2.8472557067871094, "torch": 3.327070474624634},
    "mc_2020_arjun_057": {"custom": 0.7611911296844482, "flops": 9.28492112075342, "size": 23.0, "numpy": 93.81369042396545, "np_mm": 1.6386804580688477, "torch": 1.4209449291229248},
    "rnd_mixed_08": {"custom": 7.473694324493408, "flops": 10.484556140807213, "size": 25.17695226833628, "numpy": 181.35143542289734, "np_mm": 3.5256052017211914, "torch": 2.6147942543029785}
}

# Extract instances and their corresponding FLOPs and timings
instances = list(data.keys())  # Instance names
timing_methods = ["custom", "numpy", "np_mm", "torch"]

# Create arrays to hold the data for plotting
flops_sorted = [data[instance]["flops"] for instance in instances]
times_sorted = {method: [] for method in timing_methods}

# Populate the arrays with the times for each method, ensuring that None is replaced with 0
for instance in instances:
    for method in timing_methods:
        # Explicitly handle None values by setting them to 0
        times_sorted[method].append(data[instance].get(method, 0) if data[instance].get(method) is not None else 0)

# Sort data based on FLOPs
sorted_indices = np.argsort(flops_sorted)
flops_sorted =  np.round(np.array(flops_sorted)[sorted_indices], 3)
sorted_instances = np.array(instances)[sorted_indices]

# Create the plot
plt.figure(figsize=(12, 6))

# Bar width and positions
bar_width = 0.2
x = np.arange(len(flops_sorted))

# Colors for each timing method
colors = ["#6C8B3C", "#A3BFD9", "#D1A14D", "#9C3D3D"]
hatch_patterns = ["//", "xx", "..", "||"]

# Plot each method as a separate bar group with unique colors and hatch patterns
for i, method in enumerate(timing_methods):
    plt.bar(
        x + i * bar_width, np.array(times_sorted[method])[sorted_indices], width=bar_width,
        label=method, color=colors[i], hatch=hatch_patterns[i], edgecolor="black"
    )

# Labels and title
plt.xlabel("FLOPs")
plt.ylabel("Iterations per Second")
plt.title("Iterations per Second for Different Backends Sorted by FLOPs")

# Customize the x-axis ticks with sorted flops values
plt.xticks(x + bar_width * (len(timing_methods) / 2), flops_sorted, rotation=90, ha="right")

# Add a legend and grid
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Adjust layout to prevent overlap and ensure readability
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig("iterations_per_second_sorted_by_flops.png", format="png")

# Show the plot
plt.show()
