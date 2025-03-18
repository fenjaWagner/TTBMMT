import numpy as np
import f_path_operation_copy as fo
import time
import math
import csv

def initialize_writing(filename):
    """Writes the given data to a CSV file."""
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["flops_log10", "size", "custom", "np_mm", "numpy", "torch"])

def append_results_to_csv(filename, data):
    """Appends the given data to a CSV file without removing previous entries."""
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)

def do_contraction(format_string, tensor_1, tensor_2, times, max_time=4):
    """Run the contraction and calculate the number of iterations per second for each backend."""
    
    # Run contraction for each backend
    for backend in ["custom", "np_mm", "numpy", "torch"]:
        total_time = 0
        num_iterations = 0
        while total_time < max_time:
            tic = time.time()
            C, time_fragment = fo.prepare_contraction(format_string, tensor_1, tensor_2, backend)
            toc = time.time()
            iteration_time = time_fragment if backend == "torch" else toc - tic
            total_time += iteration_time
            num_iterations +=1
        iterations_per_second = num_iterations / total_time
        print(f"{backend} **** {iterations_per_second}")
        times.append(iterations_per_second)
    return times

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
def exp_float_unopt():
    initialize_writing("data_unopt.csv")
    for i in range(14, 54, 5):
        print(f"************** doing {i} **************")
        
        [A, B, C, D, E, F] = [i, i, i, i, i, i]
        size = np.log2(math.prod([A, B, C, D]) + math.prod([A, D, E, F]))
        flops = np.log10(math.prod([A, B, C, D, E, F]) * 2)
        times = [flops, size]
         
        tensor_1 = np.random.rand(A, B, C, D)
        tensor_2 = np.random.rand(A, D, E, F)
        format_string = "abcd,adef->cbef"

        times = do_contraction(format_string, tensor_1, tensor_2, times)
        append_results_to_csv("data_unopt.csv", [times]) 

def exp_float():
    initialize_writing("data_opt.csv")
    for i in range(14, 54, 5):
        print(f"************** doing {i} **************")
        
        [A, B, C, D, E, F] = [i, i, i, i, i, i]
        size = np.log2(math.prod([A, A, B, C, D]) + math.prod([A, D, E, E, F]))
        flops = np.log10(math.prod([A, B, C, D, E, F]) * 2)
        times = [flops, size]
        
        tensor_1 = np.random.rand(A, A, B, C, D)
        tensor_2 = np.random.rand(A, D, E, E, F)
        format_string = "aabcd,adeef->dcf"

        times = do_contraction(format_string, tensor_1, tensor_2, times)
        append_results_to_csv("data_opt.csv", [times])

def exp_float_batch():
    initialize_writing("data_batch.csv")
    for i in range(14, 53, 5):
        print(f"************** doing {i} **************")
        
        [A, B, C, D, E, F] = [i, i, i, i, i, i]
        size = np.log2(math.prod([A, B, C, D]) + math.prod([A, D, E, F]))
        flops = np.log10(math.prod([A, B, C, D, E, F]) * 2)
        times = [flops, size]
         
        tensor_1 = np.random.rand(A, B, C, D)
        tensor_2 = np.random.rand(A, D, E, F)
        format_string = "abcd,adef->dbef"

        times = do_contraction(format_string, tensor_1, tensor_2, times)
        append_results_to_csv("data_batch.csv", [times])

def exp_float_traces():
    initialize_writing("data_traces.csv")
    for i in range(14, 54, 5):
        print(f"************** doing {i} **************")
        
        [A, B, C, D, E, F] = [i, i, i, i, i, i]
        size = np.log2(math.prod([A, A, B, C, D]) + math.prod([A, D, E, E, F]))
        flops = np.log10(math.prod([A, B, C, D, E, F]) * 2)
        times = [flops, size]
        
        tensor_1 = np.random.rand(A, A, B, C, D)
        tensor_2 = np.random.rand(A, D, E, E, F)
        format_string = "aabcd,adeef->bcf"

        times = do_contraction(format_string, tensor_1, tensor_2, times)
        append_results_to_csv("data_traces.csv", [times])

def exp_float32():
    initialize_writing("data_32.csv")
    for i in range(53, 55, 1):
        print(f"************** doing {i} **************")
        
        [A, B, C, D, E, F] = [i, i, i, i, i, i]
        size = np.log2(math.prod([A, A, B, C, D]) + math.prod([A, D, E, E, F]))
        flops = np.log10(math.prod([A, B, C, D, E, F]) * 2)
        times = [flops, size]
        
        tensor_1 = np.random.rand(A, A, B, C, D).astype(np.float32)
        tensor_2 = np.random.rand(A, D, E, E, F).astype(np.float32)
        format_string = "aabcd,adeef->dcf"

        times = do_contraction(format_string, tensor_1, tensor_2, times)
        append_results_to_csv("data_32.csv", [times])

def exp_sparse():
    initialize_writing("data_sparse.csv")
    for i in range(4, 54, 5):
        print(f"************** doing {i} **************")
        
        [A, B, C, D, E, F] = [i, i, i, i, i, i]
        size = np.log2(math.prod([A, A, B, C, D]) + math.prod([A, D, E, E, F]))
        flops = np.log10(math.prod([A, B, C, D, E, F]) * 2)
        times = [flops, size]
        
        tensor_1 = generate_sparse_tensor((A, A, B, C, D))
        tensor_2 = generate_sparse_tensor((A, D, E, E, F))
        format_string = "abcd,adef->dbef"

        times = do_contraction(format_string, tensor_1, tensor_2, times)
        append_results_to_csv("data_sparse.csv", [times])

exp_float()