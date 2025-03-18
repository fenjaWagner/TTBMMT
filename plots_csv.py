import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot(csv_file):
    # Load CSV data
    #csv_file = "data_opt.csv" 
    df = pd.read_csv(csv_file)

    # Extract relevant data
    x_labels = df["flops_log10"].astype(float).round(decimals=3).astype(str)  # Rounded FLOPs on x-axis
    timing_methods = ["custom", "np_mm", "torch", "numpy"]  # Added numpy to the list of backends
    times = df[timing_methods].values  # Extract bar values for the other backends (without numpy)


    # Define bar width and positions
    x = np.arange(len(df))  # X locations for FLOPs values
    bar_width = 0.2  # Width of each bar in the group

    # Define colors and hatch patterns for accessibility
    colors = ["#6C8B3C", "#A3BFD9", "#D1A14D", "#9C3D3D"] 
    hatch_patterns = ["//", "xx", "..", "||"] 
    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot bars for all methods (log2 values)
    for i, (method, hatch) in enumerate(zip(timing_methods, hatch_patterns)):
        plt.bar(
            x + i * bar_width, times[:, i], width=bar_width,
            label=method, color=colors[i], hatch=hatch, edgecolor="black"
        )

    # Labels and title
    plt.xlabel("flops_log10")
    plt.ylabel("Iterations per second")
    plt.title("aabcd,adeef->dcf")
    plt.xticks(x + bar_width * (len(timing_methods) / 2), x_labels, rotation=90)  # Adjust x-ticks for readability

    # Log scale on the y-axis
    plt.yscale('log')

    # Add the legend
    plt.legend()

    # Add gridlines on the y-axis for better readability
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Adjust layout to prevent overlap and ensure readability
    plt.tight_layout()

    # Extract the base name of the CSV file (without extension)
    base_filename = csv_file.split('.')[0]

    # Save the plot as a PNG file with the same name as the CSV file
    plt.savefig(f"{base_filename}.png", format="png")


for filename in ["data_opt.csv"]:#["data_traces.csv", "data_unopt.csv", "data_batch.csv"]:
    plot(filename)