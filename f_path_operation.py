import numpy as np
import torch
import f_map_to_bmm 
import ascii
import try_path as t
from collections import Counter

import time
import opt_einsum as oe
from cgreedy import CGreedy, compute_path
import einsum_benchmark

def build_sizes(term_A, term_B, shape_A, shape_B):
    sizes = {}
    sizes = dict(zip(term_A, shape_A))
    sizes.update(dict(zip(term_B, shape_B)))
    return sizes
    
def remove_double_indices(term, Tensor, histo, sizes):
    s_term = set()
    keep = set()
    double = {}
    
    for char in term:
        if term.count(char) == 1:
            keep.add(char)
        else: 
            double[char] = term.count(char)
    #update histo, invoke single_sum
    if double:
        for char in double:
            histo[char] -= double[char]
        tmp = term
        term = ''.join(keep)+''.join(double.keys())
        #Tensor, term = f_single_trace.single_trace(term, Tensor, sizes)
        Tensor = np.einsum(tmp +"->"+term, Tensor)
    return Tensor, term

def find_single_indices(term_A, term_B, histo):
    set_A = set()
    single_A = set()
    single_B = set()
    keep_A = set()
    keep_B = set()
    contract = set()
    batch = set()

    # build set of chars of term_A, remove them from histogramm, find singles in A
    for char in set(term_A+term_B):
        if histo[char]==1:
            histo[char] -= 1
            if char in term_A: single_A.add(char)      
            else: single_B.add(char)
        elif histo[char] == 2 and char in term_A and char in term_B:
            histo[char] -= 2
            contract.add(char)
        elif histo[char] > 2 and char in term_A and char in term_B:
            histo[char] -= 2
            batch.add(char)
        else:
            histo[char] -= 1
            if char in term_A:
                keep_A.add(char)
            else:
                keep_B.add(char)
            
        
    return single_A, single_B, keep_A, keep_B, contract, batch

def invoke_contraction_torch(term_A, term_B, A, B, contract, batch, keep_A, keep_B, sizes):
    A_t = torch.from_numpy(A)
    B_t = torch.from_numpy(B)
    output = ''.join(batch)+''.join(keep_A)+''.join(keep_B)
    C = (torch.einsum(term_A + ","+term_B+"->"+ output, A_t, B_t)).numpy()
    return C, output

def invoke_contraction_numpy(term_A, term_B, A, B, contract, batch, keep_A, keep_B, sizes):
    output = ''.join(batch)+''.join(keep_A)+''.join(keep_B)
    C = np.einsum(term_A + ","+term_B+"->"+ output, A, B)
    return C, output

def invoke_contraction_custom(term_A, term_B, A, B, contract, batch, keep_A, keep_B, sizes):
    output = ''.join(batch)+''.join(keep_A)+''.join(keep_B)
    #C = np.einsum(term_A + ","+term_B+"->"+ output, A, B)
    return f_map_to_bmm.map_to_bmm(term_A, term_B, A, B, contract, batch, keep_A, keep_B, sizes)



def prepare_contraction(term_A, term_B, A, B, histo, backend = "custom"):
    methods = {"custom": invoke_contraction_custom,
               "torch": invoke_contraction_torch,
               "numpy": invoke_contraction_numpy}
    
    sizes = build_sizes(term_A, term_B, A.shape, B.shape)
    single_A, single_B, keep_A, keep_B, contract, batch = find_single_indices(term_A, term_B, histo)
    
    # remove traces
    A, term_A = remove_double_indices(term_A, A, histo, sizes)
    B, term_B = remove_double_indices(term_B, B, histo, sizes)

    # sum over single indices
    #print(f"single A {single_A} \n single B {single_B} \n contract {contract}\n batch {batch}")
    if single_A:
        new_term = ''.join(keep_A)+''.join(batch)+''.join(contract)
        print(f"numpy single 1: {term_A + "->"+ new_term}")
        A = np.einsum(term_A + "->"+ new_term, A)
        if type(A) != np.ndarray:
            A = np.array([A])
        term_A = new_term
        #A, term_A = f_sum_single_index.remove_single_index(A, term_A, single_A, sizes)
    if single_B:
        new_term = ''.join(keep_B)+''.join(batch)+''.join(contract)
        B = np.einsum(term_B + "->"+ new_term, B)
        if type(B) != np.ndarray:
            B = np.array([B])
        print(f"numpy single 2: {term_B + "->"+ new_term}")
        term_B = new_term
        
        #B, term_B = f_sum_single_index.remove_single_index(B, term_B, single_B, sizes)

    #contract -> Tensor_new, term_new
    print(f"Type A", type(A))
    print("Type B", type(B))
    method = methods[backend]

    return method(term_A, term_B, A, B, contract, batch, keep_A, keep_B, sizes)

    

def work_path(ssa_path, tensors, format_string, backend = "custom"):
    format_string = format_string.replace(" ", "")
    #format_string = ascii.convert_to_ascii(format_string)
    
    if backend == "torch":
        format_string = ascii.convert_to_ascii(format_string)
    histogramm = Counter(format_string)
    input, output = format_string.split("->")
    print(format_string)
    print("output", output)
    return 
    term_strings = input.split(",")
    tic = time.time()
    for t_tuple in ssa_path:
        first = t_tuple[0]
        second = t_tuple[1]
        A = tensors[first]
        B = tensors[second]
        
        term_A = term_strings[first]
        term_B = term_strings[second]
        
        C, term_C = prepare_contraction(term_A, term_B, A, B, histogramm, backend)
        tensors.append(C)
        term_strings.append(term_C)
        for char in term_C:
            histogramm[char] += 1
    toc = time.time()
    if term_C != output:
        print("term_c", term_C)
        print("output", output)
        print(tuple(term_C.index(i) for i in output))
        C = np.transpose(C,tuple(term_C.index(i) for i in output))
    return C, term_C, toc-tic


def test_second():
    format_string, shapes = t.rand_equation(20, 3, seed=7, d_min=2, d_max=7, n_out=4)      
    tensors = [np.random.rand(*shape) for shape in shapes]
    path, size_log2, flops_log10 = compute_path(format_string, *tensors, seed=1, minimize="size", max_repeats=1024,
                                            max_time=1.0, progbar=True, threshold_optimal=12, threads=0, is_linear=True)

    ssa = einsum_benchmark.meta.runtime.to_ssa_path(path)

    #warmup
    C_w, term_C, time = work_path(ssa, tensors, format_string, "custom")

    for backend in ["custom", "numpy", "torch"]:
        C, term_C, time = work_path(ssa, tensors, format_string, "custom")
        print(f"Time {backend}: {time}")
        print(f"Allclose? {np.allclose(C_w, C)}")
        

    #click = time.time()
    ##test_opt_eins = oe.contract(format_string, tensors[0], tensors[1], tensors[2], tensors[3], tensors[4])
    #test_opt_eins = oe.contract(format_string,tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5], tensors[6], tensors[7], tensors[8], tensors[9],
    #                            tensors[10], tensors[11], tensors[12], tensors[13], tensors[14], tensors[15], tensors[16], tensors[17], tensors[18], tensors[19] )
    #clock = time.time()
    #print(f"Time oe: {clock-click}")


def test_ascii():
    format_string = "sidjf, ASdjipj23, ASDOIjd23, ASdoijdijs -> 23ccjijd"
    print(ascii.convert_to_ascii(format_string))

#test_second()