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
    if len(sys.argv) < 3:
        print("Usage: python my_script.py <dict_file.json>")
        sys.exit(1)
    
    instance_index = int(sys.argv[1])
    backend_index = int(sys.argv[2])
    file_name = "insteresting_einsum_dictionary.txt"

    # Load existing dictionary
    data = load_dictionary(file_name)
    
    #instance_list =  ["mc_2022_167", "mc_2022_079", "wmc_2021_130", "wmc_2023_035", "str_matrix_chain_multiplication_1000", "str_mps_varying_inner_product_2000", "wmc_2023_152", "str_mps_varying_inner_product_200", "mc_2023_002", "str_matrix_chain_multiplication_100", "lm_batch_likelihood_sentence_4_4d", "lm_batch_likelihood_brackets_4_4d", "mc_2023_188", "lm_batch_likelihood_sentence_3_12d", "mc_2020_017", "lm_batch_likelihood_sentence_4_8d", "mc_2023_arjun_117", "mc_2021_027",  "mc_rw_blasted_case1_b14_even3", "str_nw_peps_closed_333", "wmc_2023_141", "mc_2020_arjun_046", "mc_2020_arjun_057", "mc_2021_arjun_171", "str_nw_mera_closed_120", "lm_batch_likelihood_sentence_4_12d", "str_nw_mera_open_26", "rnd_mixed_08"]
    instance_list = ["mc_2020_017", "wmc_2023_141", "lm_batch_likelihood_sentence_4_12d", "md_mixed_08", "mc_2022_167", "wmc_2023_152", "mc_2023_002", "mc_2020_arjun_046", "lm_batch_likelihood_sentence_4_12d", "mc_2020_arjun_057"]
    backend_list = ["custom" , "numpy", "np_mm", "torch"]

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
                               "flops": flops,
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
            print(time)
        
        save_dictionary(file_name, data)

    except:
        print(backend +" was killed.")
        save_dictionary(file_name, data)




if __name__ == "__main__":
    main()
