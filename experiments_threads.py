import json
import sys
import einsum_benchmark
import f_path_operation_copy as fo

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
    thread_number = int(sys.argv[3])
    file_name = f"{thread_number}_threads.txt"

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
    if instance_name in data.keys():
        data[instance_name][backend] = None
    else: 
        data[instance_name] = {backend: None,
                               "threads": thread_number,
                               "size": size}
    save_dictionary(file_name, data)
    
    print(backend)
    print(instance_name)
    try: 
        C, time, time_fragment = fo.work_path(s_opt_size.path, instance.tensors, instance.format_string, backend)
        if backend == "torch":
            data[instance_name][backend] = time_fragment
            print(time_fragment)
            
        else:
            data[instance_name][backend] = time
            data[instance_name][backend+"_tf"] = time_fragment
            print(time)
        
        save_dictionary(file_name, data)

    except:
        print(backend +" was killed.")
        save_dictionary(file_name, data)




if __name__ == "__main__":
    main()
