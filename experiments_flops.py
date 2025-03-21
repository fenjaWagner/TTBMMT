import numpy as np
import f_path_operation_copy as fo
import time
import math
import csv
import json
import sys

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

def store_data(dictionary, format_string, flops, backend, iterations = 0):
    # Check if the thread_number exists in the dictionary
    if format_string not in dictionary:
        # If not, create the structure for this thread number
        dictionary[format_string] = {}

    flops = np.round(flops, 3)

    flops_str = str(flops)
    # Check if the instance_name exists for this thread number
    if flops_str not in dictionary[format_string]:
        # If not, create the structure for this instance name under the thread number
        dictionary[format_string][flops_str] = {}
    dictionary[format_string][flops_str][backend] = iterations
    save_dictionary("flop_dict.txt", dictionary)

def initialize_writing(filename):
    """Writes the given data to a CSV file."""
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["flops_log10", "size", "custom", "np_mm", "numpy", "torch"])


def do_contraction(format_string, tensor_1, tensor_2, backend, max_time=4):
    """Run the contraction and calculate the number of iterations per second for each backend."""
    total_time = 0
    num_iterations = 0
    while total_time < max_time:
        tic = time.time()
        C, time_fragment = fo.prepare_contraction(format_string, tensor_1, tensor_2, backend)
        toc = time.time()
        iteration_time = time_fragment #if backend == "torch" else toc - tic
        total_time += iteration_time
        num_iterations +=1
    iterations_per_second = num_iterations / total_time
    print(f"{backend} **** {iterations_per_second}")
    return iterations_per_second

def generate_sparse_tensor(shape, density=0.01, random_seed=None):
    """Generate a sparse tensor for a given shape and density, and return as a dense np.array."""
    if random_seed is not None:
        np.random.seed(random_seed)

    size = np.prod(shape)
    num_nonzero = int(size * density)

    # Generate random indices
    indices = np.unravel_index(
        np.random.choice(size, num_nonzero, replace=False), shape
    )

    # Generate random values
    values = np.random.rand(num_nonzero)

    # Create dense array and fill non-zero values
    dense_tensor = np.zeros(shape)
    dense_tensor[indices] = values

    return dense_tensor

# Modified experiments to calculate the iterations per second
def exp_float_unopt(i, backend, dictionary):
    #initialize_writing("data_unopt.csv")
    
    print(f"************** doing {i} **************")
    
    [A, B, C, D, E, F] = [i, i, i, i, i, i]
    size = np.log2(math.prod([A, B, C, D]) + math.prod([A, D, E, F]))
    flops = np.log10(math.prod([A, B, C, D, E, F]) * 2)
    format_string = "abcd,adef->cbef"
    store_data(dictionary, format_string, flops, backend)
        
    tensor_1 = np.random.rand(A, B, C, D)
    tensor_2 = np.random.rand(A, D, E, F)
    

    iterations_per_second = do_contraction(format_string, tensor_1, tensor_2, backend)
    store_data(dictionary, format_string, flops, backend, iterations_per_second)

def exp_float(i, backend, dictionary):
    #initialize_writing("data_opt.csv")

    print(f"************** doing {i} **************")
    
    [A, B, C, D, E, F] = [i, i, i, i, i, i]
    size = np.log2(math.prod([A, A, B, C, D]) + math.prod([A, D, E, E, F]))
    flops = np.log10(math.prod([A, B, C, D, E, F]) * 2)
    format_string = "aabcd,adeef->dcf"
    store_data(dictionary, format_string, flops, backend)
    
    tensor_1 = np.random.rand(A, A, B, C, D)
    tensor_2 = np.random.rand(A, D, E, E, F)
    

    iterations_per_second = do_contraction(format_string, tensor_1, tensor_2, backend)
    store_data(dictionary, format_string, flops, backend, iterations_per_second)

def exp_float_batch(i, backend, dictionary):
    #initialize_writing("data_batch.csv")
    print(f"************** doing {i} **************")
    
    [A, B, C, D, E, F] = [i, i, i, i, i, i]
    size = np.log2(math.prod([A, B, C, D]) + math.prod([A, D, E, F]))
    flops = np.log10(math.prod([A, B, C, D, E, F]) * 2)
    format_string = "abcd,adef->dbef"
    store_data(dictionary, format_string, flops, backend)
        
    tensor_1 = np.random.rand(A, B, C, D)
    tensor_2 = np.random.rand(A, D, E, F)
    

    iterations_per_second = do_contraction(format_string, tensor_1, tensor_2, backend)
    store_data(dictionary, format_string, flops, backend, iterations_per_second)

def exp_float_traces(i, backend, dictionary):
    #initialize_writing("data_traces.csv")

    print(f"************** doing {i} **************")
    
    [A, B, C, D, E, F] = [i, i, i, i, i, i]
    size = np.log2(math.prod([A, A, B, C, D]) + math.prod([A, D, E, E, F]))
    flops = np.log10(math.prod([A, B, C, D, E, F]) * 2)
    format_string = "aabcd,adeef->bcf"
    store_data(dictionary, format_string, flops, backend)
    
    tensor_1 = np.random.rand(A, A, B, C, D)
    tensor_2 = np.random.rand(A, D, E, E, F)
    

    iterations_per_second = do_contraction(format_string, tensor_1, tensor_2, backend)
    store_data(dictionary, format_string, flops, backend, iterations_per_second)

def exp_float32(i, backend, dictionary):
    #initialize_writing("data_32.csv")

    print(f"************** doing {i} **************")
    
    [A, B, C, D, E, F] = [i, i, i, i, i, i]
    size = np.log2(math.prod([A, A, B, C, D]) + math.prod([A, D, E, E, F]))
    flops = np.log10(math.prod([A, B, C, D, E, F]) * 2)
    format_string = "aabcd,adeef->dcf"
    store_data(dictionary, format_string, flops, backend)
    
    tensor_1 = np.random.rand(A, A, B, C, D).astype(np.float32)
    tensor_2 = np.random.rand(A, D, E, E, F).astype(np.float32)
    

    iterations_per_second = do_contraction(format_string, tensor_1, tensor_2, backend)
    store_data(dictionary, format_string, flops, backend, iterations_per_second)

def exp_sparse(i, backend, dictionary):
    #initialize_writing("data_sparse.csv")

    print(f"************** doing {i} **************")
    
    [A, B, C, D, E, F] = [i, i, i, i, i, i]
    size = np.log2(math.prod([A, A, B, C, D]) + math.prod([A, D, E, E, F]))
    flops = np.log10(math.prod([A, B, C, D, E, F]) * 2)
    format_string = "abcd,adef->dbef"
    store_data(dictionary, format_string, flops, backend)
    
    tensor_1 = generate_sparse_tensor((A, A, B, C, D))
    tensor_2 = generate_sparse_tensor((A, D, E, E, F))
    

    iterations_per_second = do_contraction(format_string, tensor_1, tensor_2, backend)
    store_data(dictionary, format_string, flops, backend, iterations_per_second)

def main():
    if len(sys.argv) < 4:
        print("Usage: python my_script.py <dict_file.json>")
        sys.exit(1)
    
    backend_index = int(sys.argv[1])
    method_index = int(sys.argv[2])
    size = int(sys.argv[3])
    dictionary = load_dictionary("flop_dict.txt")
    backends =  ["custom", "np_mm", "numpy", "torch"]
    methods = [exp_float, exp_float_batch, exp_float_traces, exp_float_unopt]

    backend  = backends[backend_index]
    method = methods[method_index]

    method(size, backend, dictionary)

if __name__ == "__main__":
    main()
