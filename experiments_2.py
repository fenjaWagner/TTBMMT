import numpy as np
import f_path_operation_copy as fo
import time
import math
import csv

def initialize_writing():
    """Writes the given data to a CSV file."""
    with open("data_flops.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["flops_log10", "custom", "np_mm", "torch", "numpy"])

def append_results_to_csv(data):
    """Appends the given data to a CSV file without removing previous entries."""
    with open("data_flops.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)

def exp_float():
    data = []
    initialize_writing()
    for i in range(14,24,5):
        
        [A,B,C,D,E,F] = [i,i,i,i,i,i]
        size = np.log2(math.prod([A,A,B,C,D]) + math.prod([A,D,E,E,F]))
        flops = np.log10(math.prod([A,B,C,D,E,F])*2)
        times = [flops]
        
        tensor_1 = np.random.rand(A, A, B, C, D)
        tensor_2 = np.random.rand(A, D, E, E, F)
        format_string = "aabcd,adeef->dcf"


        # *************** Warm up *****************************************
        C, time_fragment = fo.prepare_contraction(format_string, tensor_1, tensor_2, "torch")

        for backend in ["custom", "np_mm", "torch", "numpy"]:
            tic = time.time()
            C, time_fragment = fo.prepare_contraction(format_string, tensor_1, tensor_2, backend)
            toc = time.time()
            times.append(toc-tic)
        data.append(times)
    append_results_to_csv(data)    

def size_calc():
    for i in range(14,55,5):
        print(i)
        [A,B,C,D,E,F] = [i,i,i,i,i,i]
        size = math.prod([A,A,B,C,D]) + math.prod([A,D,E,E,F])
        flops = math.prod([A,B,C,D,E,F])*2
        print("size", np.log2(size))
        print("flops", np.log10(flops))

exp_float()