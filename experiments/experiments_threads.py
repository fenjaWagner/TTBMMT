import json
import sys
import einsum_benchmark
import multi_tc as fo


def load_dictionary(filename):
    """Load a dictionary from a JSON file."""
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Creating a new dictionary.")
        return {}  # Return an empty dictionary if file doesn't exist

def save_dictionary(filename, data):
    """Save a dictionary to a JSON file."""
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def main():
    # Get filename from command-line argument
    if len(sys.argv) < 4:
        print("Usage: python my_script.py <dict_file.json>")
        sys.exit(1)
    
    instance_index = int(sys.argv[1])
    backend_index = int(sys.argv[2])
    thread_number = str(sys.argv[3])
    file_name = "threads.txt"

    # Load existing dictionary
    data = load_dictionary(file_name)
    
    instance_list = ["mc_2020_arjun_046", "lm_batch_likelihood_sentence_4_8d"]
    backend_list = ["custom", "np_mm", "torch"]

    instance_name = instance_list[instance_index]
    backend = backend_list[backend_index]

    instance = einsum_benchmark.instances[instance_name]
    s_opt_size = instance.paths.opt_size
    flops = s_opt_size.flops
    size = s_opt_size.size
    thread_number = str(thread_number)
   
    # Check if the thread_number exists in the dictionary
    if thread_number not in data:
        # If not, create the structure for this thread number
        data[thread_number] = {}

    # Check if the instance_name exists for this thread number
    if instance_name not in data[thread_number]:
        # If not, create the structure for this instance name under the thread number
        data[thread_number][instance_name] = {}
        save_dictionary(file_name, data)
    
    print(backend)
    print(instance_name)

    print(f"Running {backend} with warmup...")
    
    # Warm-up iterations (not timed)
    for _ in range(1):
        C, time, time_fragment = fo.multi_tc(s_opt_size.path, instance.tensors, instance.format_string, backend)
        
    print(f"Warmup complete for {backend}. Starting timed runs...")
    try: 
        total_time = 0
        num_iterations = 0
        while total_time < 10:
            C, run_time, time_fragment = fo.multi_tc(s_opt_size.path, instance.tensors, instance.format_string, backend)
            iteration_time = time_fragment #if backend == "torch" else run_time
            total_time += iteration_time
            num_iterations += 1

        iterations_per_second = num_iterations / total_time
        data[thread_number][instance_name][backend]= iterations_per_second
        save_dictionary(file_name, data)
    except:
        print(backend +" was killed.")
        save_dictionary(file_name, data)
    
       



if __name__ == "__main__":
    main()
