import numpy as np
import f_path_operation_copy as fo
import time
import math
import csv
import sys

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

def do_contraction(format_string, tensor_1, tensor_2, times):
    # *************** Warm up ****************************************
    C, time_fragment = fo.prepare_contraction(format_string, tensor_1, tensor_2, "torch")

    for backend in ["custom", "np_mm", "numpy", "torch"]:
        tic = time.time()
        C, time_fragment = fo.prepare_contraction(format_string, tensor_1, tensor_2, backend)
        toc = time.time()
        if backend == "torch":
            times.append(time_fragment)
            
        else:
            times.append(toc-tic)
    return times

def exp_float_unopt(num_threads):
    initialize_writing(str(num_threads)+"data_flops_unopt.csv")
    for i in range(50,54,1):
        print(f"************** doing {i} **************")
        
        [A,B,C,D,E,F] = [i,i,i,i,i,i]
        size = np.log2(math.prod([A,B,C,D]) + math.prod([A,D,E,F]))
        flops = np.log10(math.prod([A,B,C,D,E,F])*2)
        times = [flops, size]
         
        tensor_1 = np.random.rand(A, B, C, D)
        tensor_2 = np.random.rand(A, D, E, F)
        format_string = "abcd,adef->dcbef"

        times = do_contraction(format_string, tensor_1, tensor_2, times)
 
        append_results_to_csv(str(num_threads)+"data_flops_unopt.csv",[times]) 
       
def exp_float(num_threads):
    initialize_writing(str(num_threads)+"data_flops.csv")
    for i in range(50,54,1):
        print(f"************** doing {i} **************")
        
        [A,B,C,D,E,F] = [i,i,i,i,i,i]
        size = np.log2(math.prod([A,A,B,C,D]) + math.prod([A,D,E,E,F]))
        flops = np.log10(math.prod([A,B,C,D,E,F])*2)
        times = [flops, size]
        
        tensor_1 = np.random.rand(A, A, B, C, D)
        tensor_2 = np.random.rand(A, D, E, E, F)
        format_string = "aabcd,adeef->dcf"


        times = do_contraction(format_string, tensor_1, tensor_2, times)    
        append_results_to_csv(str(num_threads)+"data_flops.csv",[times])

def exp_float32(num_threads):
    initialize_writing(str(num_threads)+"data_flops32.csv")
    for i in range(50,54,1):
        print(f"************** doing {i} **************")
        
        [A,B,C,D,E,F] = [i,i,i,i,i,i]
        size = np.log2(math.prod([A,A,B,C,D]) + math.prod([A,D,E,E,F]))
        flops = np.log10(math.prod([A,B,C,D,E,F])*2)
        times = [flops, size]
        
        tensor_1 = np.random.rand(A, A, B, C, D).astype(np.float32)
        tensor_2 = np.random.rand(A, D, E, E, F).astype(np.float32)
        format_string = "aabcd,adeef->dcf"


        times = do_contraction(format_string, tensor_1, tensor_2, times)    
        append_results_to_csv(str(num_threads)+"data_flops32.csv",[times])



# Standard Python entry point
if __name__ == "__main__":
    # Check if enough arguments are provided
    if len(sys.argv) < 2:
        print("Usage: python script.py <num1> <num2>")
        sys.exit(1)  # Exit with an error code

    # Convert arguments to float
    num1 = int(sys.argv[1])
    exp_float(num1)
    exp_float32(num1)
    exp_float_unopt(num1)