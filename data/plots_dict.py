import matplotlib.pyplot as plt
import numpy as np
import json

# Sample dictionary with multiple instances
def load_dictionary(filename):
    """Load a dictionary from a JSON file."""
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Creating a new dictionary.")
        return {}  # Return an empty dictionary if file doesn't exist

def normal_plot():
    data = load_dictionary("dictionary.txt")
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




def numpy_numbers_plot():
    data = load_dictionary("dictionary.txt")
    
    # Extract instance names and timing methods
    instance_names = list(data.keys())
    timing_methods = ["custom", "np_mm", "torch"]  # Bars only for these
    
    # Convert data into a 2D list (rows = instances, columns = methods)
    times = np.array([
        [data[instance].get(method, 0) for method in timing_methods]
        for instance in instance_names
    ], dtype=np.float32)

    # Replace NaNs (if any) with 0
    times[np.isnan(times)] = 0  

    # Extract NumPy times separately
    numpy_times = np.array([
        data[instance].get("numpy", 0) for instance in instance_names
    ])

    # Define bar width and positions
    x = np.arange(len(instance_names))  # X locations for instances
    bar_width = 0.2  # Adjust for spacing

    # Define colors and hatch patterns for accessibility
    colors = ["blue", "green", "purple"]
    hatch_patterns = ["//", "xx", ".."]  

    # Create the plot with a larger figure size
    plt.figure(figsize=(14, 6))

    # Plot bars for non-numpy methods
    for i, (method, hatch) in enumerate(zip(timing_methods, hatch_patterns)):
        plt.bar(
            x + i * bar_width, times[:, i], width=bar_width,
            label=method, color=colors[i], hatch=hatch, edgecolor="black"
        )

    # Set a **fixed height** for NumPy labels above the tallest bar
    fixed_offset = 0.02 * max(times.flatten())  

    # Add NumPy results as text above the bars
    for i in range(len(instance_names)):
        plt.text(
            x[i] + bar_width * 1,  # Center above bar groups
            max(times[i]) + fixed_offset,  # Fixed height offset
            f"{numpy_times[i]:.2f}s",  
            color="red", fontsize=9, ha="center", fontweight="bold"
        )

    # Add NumPy to the legend
    plt.scatter([], [], color="red", label="numpy (text)")

    # Labels and title
    plt.xlabel("Instance Names")
    plt.ylabel("Time (seconds)")
    plt.title("Execution Times for Different Instances (NumPy as Text)")

    # ✅ Fix x-axis labels to prevent overlap
    plt.xticks(x + bar_width * (len(timing_methods) / 2), instance_names, rotation=90, ha="right")

    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Prevent labels from being cut off
    plt.tight_layout()

    # Show the plot
    plt.show()

numpy_numbers_plot()