import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import re

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
    instance_names = [i[0] for i in sorted(data.items(), key=lambda x: x[1]['flops'])]
    print(instance_names)
    timing_methods = ["custom", "np_mm", "torch", "numpy"]

    colors = ["#27AEEF", "#F9776C", "#666767", "#FEA301"]
    hatch_patterns = ["", "xx", "--", "//"]

    def split_instance_name(name):
        return '_\n'.join(name.split('_'))
    bar_width = 0.18  # Adjust for spacing

    plt.figure(figsize=(12, 6))

    for instance_idx, instance in enumerate(instance_names):
        active_methods = [(i, method) for i, method in enumerate(timing_methods) if (data[instance].get(method) or 0) > 0]
        num_active = len(active_methods)
        for bar_idx, (i, method) in enumerate(active_methods):
            val = data[instance].get(method) or 0
            bar_x = instance_idx - (num_active - 1) * 0.5 * 0.18 + bar_idx * 0.18
            plt.bar(
                bar_x, val, width=bar_width,
                label=method if instance_idx == 0 else None,
                color=colors[i], hatch=hatch_patterns[i], edgecolor="white"
            )

    split_names = [split_instance_name(name) for name in instance_names]
    fontsize = 11
    plt.xlabel("Instance Name", fontsize = fontsize)
    plt.ylabel("Iterations per Second (log scale)", fontsize = fontsize)
    base_filename = filename.split('.')[0]
    display_name = re.sub(r'[_]', ' ', base_filename)
    plt.title(f"Iterations per Second for Different Instances with {display_name} and Methods")
    plt.xticks(range(len(instance_names)), split_names, rotation=0, ha="center")
    ymax = plt.ylim()[1]
    for i,  instance in enumerate(instance_names):
        flops = round(data[instance]['flops'], 3)
        x_pos = i -0.15 #+ bar_width * (len(timing_methods)/ 2)
        #y_pos = max([data[instance].get(m, 0) or 0 for m in timing_methods])  # get highest bar height
        plt.text(x_pos, ymax * 0.98, f"flops: {flops}", ha='center', fontsize=10, rotation=0)
    plt.yscale('log')
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tick_params(axis='both', labelsize=fontsize) 
    plt.legend(loc='upper right', ncol=1)
    plt.tight_layout()

    
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', base_filename)
    plt.savefig(f"{safe_name}.png", dpi=300)



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
    colors = ["#27AEEF", "#F9776C", "#666767", "#FEA301"] 
    hatch_patterns = ["", "xx", "--", "//"]

    # Function to split instance names for better readability
    def split_instance_name(name):
        return '_\n'.join(name.split('_'))

    # Create the plot
    plt.figure(figsize=(10, 5))

    # Plot each method as a separate bar group with unique colors and hatch patterns
    for i, (method, hatch) in enumerate(zip(timing_methods, hatch_patterns)):
        plt.bar(
            x + i * bar_width, times[:, i], width=bar_width,
            label=method, color=colors[i], hatch=hatch, edgecolor="white"
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



def plot_threads():
    problem_name = "mc_2020_arjun_046"
    data = load_dictionary("threads.txt")
    thread_numbers = [int(thread) for thread in data.keys()]
    backends = list(next(iter(data["1"].values())).keys())  # Extract backend names
    colors = ["#27AEEF", "#F9776C", "#666767"] # Custom colors for different backends
    hatch_patterns = ["", "xx", "--"]  # Patterns for accessibility
    bar_width = 0.2  # Width of each bar

    # Create the plot
    plt.figure(figsize=(8, 6))
    offset = (bar_width * (len(backends) - 1)) / 2
    
    for backend_idx, backend in enumerate(backends):
        backend_values = [data[str(thread_number)].get(problem_name, {}).get(backend, 0) for thread_number in thread_numbers]
        
        # Plotting the bars
        x_positions = np.array(thread_numbers) - offset + backend_idx * bar_width
        plt.bar(

            x_positions, backend_values, width=bar_width,
            label=f"{backend}", color=colors[backend_idx], hatch=hatch_patterns[backend_idx], edgecolor="white"
        )

    # Set labels and title
    plt.xlabel("Number of Threads")
    plt.ylabel("Iterations per Second")
    plt.title(f"Iterations per Second for {problem_name} Across Different Backends and Threads")

    
    # Set log scale for Y-axis
    #plt.yscale('log')
    
    # Add a legend for the backends
    plt.legend()

    # Gridlines
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Set x-ticks to be thread numbers
    plt.xticks(thread_numbers)

    # Adjust layout for better visibility
    plt.tight_layout()
    # Show the plot
    plt.savefig(f"threads.png", format="png")

def plot_flops():

    data = load_dictionary("flop_dict.txt")

    colors = {
    "custom": "#27AEEF",    # greenish
    "np_mm": "#F9776C",     # blueish
    "torch": "#666767",     # reddish
    "numpy": "#FEA301"      # yellowish
}

    hatches = {
        "custom": "",
        "np_mm": "xx",
        "torch": "--",
        "numpy": "//"
    }

    for formatstring, flop_dict in data.items():
        flops = []
        backend_results = {
            "custom": [],
            "np_mm": [],
            "torch": [],
            "numpy": []
        }

        for flop, results in flop_dict.items():
            values = [results[backend] for backend in backend_results]
            if not all(v == 0 for v in values):
                flops.append(float(flop))
                for backend in backend_results:
                    backend_results[backend].append(results[backend])

        sorted_indices = np.argsort(flops)
        flops_sorted = np.array(flops)[sorted_indices]
        for backend in backend_results:
            backend_results[backend] = np.array(backend_results[backend])[sorted_indices]

        bar_width = 0.2
        x = np.arange(len(flops_sorted))
        plt.figure(figsize=(12, 6))

        for i, backend in enumerate(backend_results):
            bars = plt.bar(
                x + i * bar_width,
                backend_results[backend],
                width=bar_width,
                label=backend,
                color=colors[backend],
                hatch=hatches[backend],
                edgecolor="white"
            )

        plt.yscale('log')
        plt.xticks(x + 1.5 * bar_width, [f"{f:.2f}" for f in flops_sorted], rotation=45)
        plt.xlabel("FLOPs")
        plt.ylabel("Iterations per second (log scale)")
        plt.title(f"Iterations per second for '{formatstring}'")
        plt.legend()
        plt.tight_layout()

        # Create a safe filename
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', formatstring)
        plt.savefig(f"{safe_name}.png", dpi=300)


#plot_threads()
#plot_flops()
for file in ["datatype_double.txt", "original_datatypes.txt"]:
    dynamic_normal_plot(file)