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



def dynamic_normal_plot(filename):
    data = load_dictionary(filename)
    instance_names = list(data.keys())
    timing_methods = ["custom", "np_mm", "torch", "numpy"]

    colors = ["#6C8B3C", "#A3BFD9", "#D1A14D", "#9C3D3D"]
    hatch_patterns = ["//", "xx", "..", "||"]

    def split_instance_name(name):
        return '_\n'.join(name.split('_'))

    plt.figure(figsize=(12, 6))

    for instance_idx, instance in enumerate(instance_names):
        active_methods = [(i, method) for i, method in enumerate(timing_methods) if (data[instance].get(method) or 0) > 0]
        num_active = len(active_methods)
        for bar_idx, (i, method) in enumerate(active_methods):
            val = data[instance].get(method) or 0
            bar_x = instance_idx - (num_active - 1) * 0.5 * 0.18 + bar_idx * 0.18
            plt.bar(
                bar_x, val, width=0.18,
                label=method if instance_idx == 0 else None,
                color=colors[i], hatch=hatch_patterns[i], edgecolor="black"
            )

    split_names = [split_instance_name(name) for name in instance_names]
    plt.xlabel("Instance Name")
    plt.ylabel("Iterations per Second (log scale)")
    plt.title("Iterations per Second for Different Instances and Methods")
    plt.xticks(range(len(instance_names)), split_names, rotation=0, ha="center")
    plt.yscale('log')
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()

    base_filename = filename.split('.')[0]
    plt.savefig(f"{base_filename}_compact.png", format="png")



def normal_plot(filename):
    data = load_dictionary(filename)
    # Extract instance names and timing methods
    instance_names = list(data.keys())
    timing_methods = ["custom", "np_mm", "numpy",  "torch"]

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
    colors = ["#6C8B3C", "#A3BFD9", "#D1A14D", "#9C3D3D"] 
    hatch_patterns = ["//", "xx", "..", "||"] 

    # Function to split instance names for better readability
    def split_instance_name(name):
        return '_\n'.join(name.split('_'))

    # Create the plot
    plt.figure(figsize=(10, 5))

    # Plot each method as a separate bar group with unique colors and hatch patterns
    for i, (method, hatch) in enumerate(zip(timing_methods, hatch_patterns)):
        plt.bar(
            x + i * bar_width, times[:, i], width=bar_width,
            label=method, color=colors[i], hatch=hatch, edgecolor="black"
        )

    # Split the instance names for readability
    split_names = [split_instance_name(name) for name in instance_names]

    # Labels and title
    plt.xlabel("Instance Name")
    plt.ylabel("Iterations per Second")
    plt.title("Iterations per Second for Different Instances")
    
    # Fix x-axis labels to prevent overlap by rotating and splitting them
    plt.xticks(x + bar_width * (len(timing_methods) / 2), split_names, rotation=0, ha="center")
    plt.yscale('log')
    plt.ylabel("Iterations per Second (log scale)")

    # Adjust layout to prevent overlap and ensure readability
    plt.tight_layout()
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    base_filename = filename.split('.')[0]

    # Save the plot as a PNG file with the same name as the CSV file
    plt.savefig(f"{base_filename}.png", format="png")
    #plt.savefig(f"benchmark.png", format="png")

    # Show the plot
   # plt.show()





def numpy_numbers_plot():
    data = load_dictionary("interesting_einsum_dictionary.txt")
    
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
    y_max = max(times.flatten()) 
    plt.ylim(0, y_max * 1.15)  

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

    # Fix x-axis labels to prevent overlap
    plt.xticks(x + bar_width * (len(timing_methods) / 2), instance_names, rotation=90, ha="right")

    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Prevent labels from being cut off
    plt.tight_layout()

    # Show the plot
    plt.show()


def threads_plot():
    data = load_dictionary("1_threads.txt")
    
    # Prepare data for plotting
    labels = []
    threads = []
    sizes = []
    iterations_per_second = []  # Placeholder for iterations per second (using size/threads as a simple proxy)

    for key, value in data.items():
        labels.append(key)
        threads.append(value["threads"])
        sizes.append(value["size"])
        iterations_per_second.append(value["size"] / value["threads"])  # Simple assumption: size divided by threads

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group data by thread count (x-axis)
    unique_threads = sorted(set(threads))
    bar_width = 0.2
    index = np.arange(len(unique_threads))

    # Plot bars for each thread count
    for i, thread_count in enumerate(unique_threads):
        # Get the data for the current thread count
        current_data = [iterations_per_second[j] for j in range(len(threads)) if threads[j] == thread_count]
        current_labels = [labels[j] for j in range(len(threads)) if threads[j] == thread_count]
        current_sizes = [sizes[j] for j in range(len(threads)) if threads[j] == thread_count]

        ax.bar(index[i] + bar_width * np.arange(len(current_data)), current_data,
            bar_width, label=f"Threads: {thread_count}", alpha=0.7)

    # Adding labels, title, and legend
    ax.set_xlabel("Thread Configurations")
    ax.set_ylabel("Iterations per Second (Size/Threads)")
    ax.set_title("Iterations per Second for Different Thread Configurations")
    ax.set_xticks(index + bar_width * 1.5)
    ax.set_xticklabels([f"Thread {t}" for t in unique_threads], rotation=45)
    ax.legend(title="Thread Count")
    plt.tight_layout()

    plt.savefig(f"threads.png", format="png")

dynamic_normal_plot("e_b_double.txt")